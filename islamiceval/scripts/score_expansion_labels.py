#!/usr/bin/env python3
"""
score_expansion_labels.py
=========================
Analyze the manually-labeled blind pair set to (1) decide which metric separates
same/different hadith matns better, and (2) locate each metric's threshold — defensibly.

Inputs:
  pairs_key.json      (scores, hidden during labeling)
  pair_labels.json    (your labels: [{pid, label: same|different|unsure|null}])

For each metric (char-ratio, jaccard, jaccard-extended):
  * ROC-AUC (rank-based / Mann-Whitney) — threshold-free separation; the honest
    "which metric is better" answer.
  * Threshold sweep — precision/recall/F1/accuracy at every midpoint between sorted
    scores; reports the F1-optimal threshold AND the full range of thresholds that
    achieve the top accuracy (small sample → report a range, not a false point).
Then a head-to-head: on pairs where the two metrics DISAGREE at their best thresholds,
which one matches your label.

'unsure' pairs are excluded from scoring (counted separately).

Usage:
  python score_expansion_labels.py --labels ../outputs/correction/expansion_study/pair_labels.json
"""
import os, sys, json, re, argparse
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def load_labels(path):
    """Tolerant loader: accepts a clean [{pid,label}] array, a {pid:label} dict, or
    Gemini output wrapped in ```json fences / surrounding prose. Returns {pid: label}."""
    txt = open(path, encoding="utf-8").read()
    try:
        data = json.loads(txt)
    except Exception:
        m = re.search(r"\[.*\]", txt, re.S) or re.search(r"\{.*\}", txt, re.S)
        if not m:
            sys.exit(f"could not find JSON in {path}")
        data = json.loads(m.group(0))
    if isinstance(data, dict):
        return {k: v for k, v in data.items()}
    return {r["pid"]: r.get("label") for r in data}
HERE = os.path.dirname(os.path.abspath(__file__))
STUDY = os.path.join(HERE, "..", "outputs", "correction", "expansion_study")
# the 2x2 (metric family x frame-stripping) + the extended-frame Jaccard
METRICS = [("char",         "SEQUENCE + raw      (char-ratio, current)"),
           ("char_content", "SEQUENCE + content  (char-ratio, frame-stripped)"),
           ("jac_raw",      "SET + raw           (jaccard, frame kept)"),
           ("jac",          "SET + content       (jaccard, frame-stripped)"),
           ("jac_ext",      "SET + content+open  (jaccard, extended)")]


def auc(scores_y):
    """Rank-based AUC (Mann-Whitney U with tie-averaged ranks)."""
    pos = [s for s, y in scores_y if y == 1]
    neg = [s for s, y in scores_y if y == 0]
    if not pos or not neg:
        return float("nan"), len(pos), len(neg)
    alls = sorted(s for s, _ in scores_y)
    # average ranks
    ranks = {}
    i = 0
    while i < len(alls):
        j = i
        while j + 1 < len(alls) and alls[j + 1] == alls[i]:
            j += 1
        r = (i + j) / 2 + 1                       # 1-based average rank
        ranks[alls[i]] = r
        i = j + 1
    sum_pos = sum(ranks[s] for s in pos)
    u = sum_pos - len(pos) * (len(pos) + 1) / 2
    return u / (len(pos) * len(neg)), len(pos), len(neg)


def sweep(scores_y):
    """Return list of (threshold, prec, rec, f1, acc) and best-F1 + top-acc range."""
    pos_n = sum(1 for _, y in scores_y if y == 1)
    neg_n = sum(1 for _, y in scores_y if y == 0)
    scs = sorted(set(s for s, _ in scores_y))
    cands = [0.0] + [(scs[i] + scs[i + 1]) / 2 for i in range(len(scs) - 1)] + [1.0]
    rows = []
    for t in cands:
        tp = sum(1 for s, y in scores_y if s >= t and y == 1)
        fp = sum(1 for s, y in scores_y if s >= t and y == 0)
        fn = pos_n - tp
        tn = neg_n - fp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / pos_n if pos_n else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        acc = (tp + tn) / (pos_n + neg_n)
        rows.append((t, prec, rec, f1, acc))
    best_f1 = max(rows, key=lambda r: r[3])
    top_acc = max(r[4] for r in rows)
    acc_ts = [r[0] for r in rows if abs(r[4] - top_acc) < 1e-9]
    return rows, best_f1, (min(acc_ts), max(acc_ts), top_acc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", default=os.path.join(STUDY, "pairs_key_100.json"))
    ap.add_argument("--labels", default=os.path.join(STUDY, "gemini_labels_100.json"))
    args = ap.parse_args()

    key = json.load(open(args.key, encoding="utf-8"))
    labs = load_labels(args.labels)
    Y = {}
    n_uns = n_none = 0
    for pid, lab in labs.items():
        if lab == "unsure": n_uns += 1; continue
        if lab in (None, ""): n_none += 1; continue
        Y[pid] = 1 if lab == "same" else 0
    print(f"labeled: {len(Y)} scored ({sum(Y.values())} same / {len(Y)-sum(Y.values())} different)"
          f" | unsure {n_uns} | unlabeled {n_none}\n")
    if not Y:
        sys.exit("no usable labels yet.")

    results = {}
    for mk, name in METRICS:
        sy = [(key[pid][mk], Y[pid]) for pid in Y if pid in key]
        a, npos, nneg = auc(sy)
        rows, bestf1, (tlo, thi, tacc) = sweep(sy)
        results[mk] = {"auc": a, "bestf1": bestf1, "range": (tlo, thi, tacc), "rows": rows}
        print(f"=== {name} ===")
        print(f"  ROC-AUC = {a:.3f}   (1.0=perfect separation, 0.5=chance)")
        print(f"  best F1 @ th={bestf1[0]:.3f}: prec={bestf1[1]:.2f} rec={bestf1[2]:.2f} "
              f"F1={bestf1[3]:.2f} acc={bestf1[4]:.2f}")
        print(f"  top accuracy {tacc:.2f} holds for thresholds in [{tlo:.3f}, {thi:.3f}]")
        print()

    # winner by AUC
    order = sorted(METRICS, key=lambda m: -results[m[0]]["auc"])
    print("──────── VERDICT ────────")
    for mk, name in order:
        print(f"  {name:26} AUC={results[mk]['auc']:.3f}  rec-threshold≈{results[mk]['bestf1'][0]:.2f}")
    win = order[0]
    print(f"\n  BEST SEPARATOR: {win[1]}  (AUC {results[win[0]]['auc']:.3f})")

    # head-to-head on disagreement pairs (at each metric's best-F1 threshold)
    tj = results["jac"]["bestf1"][0]; tc = results["char"]["bestf1"][0]
    dis = []
    for pid in Y:
        if pid not in key: continue
        cj = 1 if key[pid]["jac"] >= tj else 0
        cc = 1 if key[pid]["char"] >= tc else 0
        if cj != cc:
            dis.append((pid, Y[pid], cc, cj))
    if dis:
        jac_right = sum(1 for _, y, cc, cj in dis if cj == y)
        char_right = sum(1 for _, y, cc, cj in dis if cc == y)
        print(f"\n  Disagreement pairs (jac vs char at their best thresholds): {len(dis)}")
        print(f"    jaccard matches your label : {jac_right}/{len(dis)}")
        print(f"    char-ratio matches your label: {char_right}/{len(dis)}")
        print("    pid    label      char->  jac->")
        for pid, y, cc, cj in dis:
            print(f"    {pid}  {'same' if y else 'diff':9} {cc}      {cj}   "
                  f"(char={key[pid]['char']:.2f} jac={key[pid]['jac']:.2f})")


if __name__ == "__main__":
    main()
