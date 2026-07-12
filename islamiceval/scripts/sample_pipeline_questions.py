#!/usr/bin/env python3
"""
sample_pipeline_questions.py
============================
Pick a stratified subsample of the cleaned questions to push through the pipeline,
designed for an incremental "500 now, +500 later" rollout.

Method
------
* Pool = cleaned+appropriate prompts (`questions_final_cleaned.json`, qid) joined to
  the 3-axis classification (`question_labels_arabic_prompt_full.json`, id →
  category / difficulty / divergence).
* Strata = the joint cell (category, difficulty, divergence).
* PROPORTIONAL INTERLEAVING: within each stratum items are shuffled (seeded) and
  given an even spread key (rank+0.5)/size ∈ (0,1); sorting all items by that key
  makes ANY prefix ≈ proportional to the stratum sizes. So batch-1 = first B, batch-2
  = next B, and every cumulative checkpoint stays representative. Reproducible.
* RARE-CELL FLOOR: joint cells are too fine to floor at B=500 (8×5×5=200 cells), so
  we floor the MARGINS instead — the first batch is nudged so each category, each
  difficulty level and each divergence level has >= --floor items (capped at what the
  pool can supply). Promotion pulls the needed value from later in the order and evicts
  the most "borrowed-from-future" removable item, keeping the batch proportional
  otherwise.

Outputs (default outputs/classification/):
  pipeline_sample_stratified.csv / .json  — every pooled question with
      order | batch | qid | prompt | category | difficulty | divergence
  Take batch==1 to start; batch==2 for the next 500; etc. `--batches N` trims output
  to the first N batches.

Usage
-----
  python sample_pipeline_questions.py                       # seed 42, batch 500, floor 15
  python sample_pipeline_questions.py --batch 500 --floor 20 --batches 4
"""
import os, sys, csv, json, argparse, random
from collections import defaultdict, Counter

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
CLS  = os.path.join(HERE, "..", "outputs", "classification")
AXES = ("category", "difficulty", "divergence")


def load_pool(questions_path, labels_path):
    q = json.load(open(questions_path, encoding="utf-8"))
    q = q if isinstance(q, list) else q.get("questions", q)
    lab = json.load(open(labels_path, encoding="utf-8"))
    lab = lab if isinstance(lab, list) else (lab.get("questions") or lab.get("items") or list(lab.values()))
    labmap = {str(x["id"]): x for x in lab}
    pool, missing = [], 0
    for it in q:
        qid = str(it["qid"])
        lb = labmap.get(qid)
        if not lb:
            missing += 1
            continue
        pool.append({"qid": qid, "prompt": it.get("prompt", ""),
                     "category": str(lb.get("category")), "difficulty": str(lb.get("difficulty")),
                     "divergence": str(lb.get("divergence"))})
    if missing:
        print(f"  warn: {missing} cleaned questions had no label (skipped)")
    return pool


def proportional_order(pool, seed):
    """Even-spread interleaving: any prefix ≈ proportional to joint-cell sizes."""
    rng = random.Random(seed)
    strata = defaultdict(list)
    for it in pool:
        strata[(it["category"], it["difficulty"], it["divergence"])].append(it)
    for items in strata.values():
        rng.shuffle(items)
        m = len(items)
        for r, it in enumerate(items):
            it["_key"] = (r + 0.5) / m
    for it in pool:
        it["_tie"] = rng.random()
    return sorted(pool, key=lambda it: (it["_key"], it["_tie"]))


def margin_targets(pool, floor):
    """floor per margin value, capped at pool availability."""
    tgt = {}
    for ax in AXES:
        for val, n in Counter(it[ax] for it in pool).items():
            tgt[(ax, val)] = min(floor, n)
    return tgt


def enforce_floor(order, batch_size, targets):
    """Nudge the first batch_size items so every margin meets its floor."""
    batch, rest = order[:batch_size], order[batch_size:]

    def counts(items):
        c = Counter()
        for it in items:
            for ax in AXES:
                c[(ax, it[ax])] += 1
        return c

    for _ in range(batch_size * len(AXES)):  # generous safety bound
        cb = counts(batch)
        deficits = [(m, targets[m] - cb[m]) for m in targets if cb[m] < targets[m]]
        if not deficits:
            break
        (ax, val), _need = max(deficits, key=lambda x: x[1])
        # bring in the earliest-ordered rest item having this value
        src_i = next((i for i, it in enumerate(rest) if it[ax] == val), None)
        if src_i is None:
            targets[(ax, val)] = cb[(ax, val)]  # unreachable; freeze so we don't loop
            continue
        incoming = rest.pop(src_i)
        # evict the most "future-borrowed" batch item whose removal breaks no met floor
        cb2 = counts(batch)
        evict_i = None
        for i in sorted(range(len(batch)), key=lambda i: -batch[i]["_key"]):
            it = batch[i]
            if all(cb2[(a, it[a])] - 1 >= targets.get((a, it[a]), 0) for a in AXES):
                evict_i = i
                break
        if evict_i is None:
            batch.append(incoming)          # can't evict cleanly; let batch grow by 1
        else:
            rest.append(batch.pop(evict_i))
            batch.append(incoming)
    # keep each part in order key; rest re-sorted (evicted items reinserted by key)
    batch.sort(key=lambda it: (it["_key"], it["_tie"]))
    rest.sort(key=lambda it: (it["_key"], it["_tie"]))
    return batch + rest


def summarize(order, batch_size, pool):
    N = len(pool)
    b1 = order[:batch_size]
    print(f"\nPool {N} questions | batch {batch_size} | batches ≈ {-(-N//batch_size)}")
    for ax in AXES:
        pop = Counter(it[ax] for it in pool)
        b1c = Counter(it[ax] for it in b1)
        print(f"\n  {ax}:  (batch-1 count | pop share → batch-1 share)")
        for val in sorted(pop, key=lambda v: (-pop[v], v)):
            print(f"    {val:26} {b1c[val]:4d}   {100*pop[val]/N:5.1f}% → {100*b1c[val]/len(b1):5.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default=os.path.join(CLS, "questions_final_cleaned.json"))
    ap.add_argument("--labels", default=os.path.join(CLS, "question_labels_arabic_prompt_full.json"))
    ap.add_argument("--out", default=os.path.join(CLS, "pipeline_sample_stratified"))
    ap.add_argument("--batch", type=int, default=500)
    ap.add_argument("--floor", type=int, default=15, help="min per margin value in batch 1")
    ap.add_argument("--batches", type=int, default=0, help="trim output to first N batches (0=all)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    pool = load_pool(args.questions, args.labels)
    order = proportional_order(pool, args.seed)
    if args.floor > 0:
        order = enforce_floor(order, args.batch, margin_targets(pool, args.floor))

    for i, it in enumerate(order):
        it["order"] = i
        it["batch"] = i // args.batch + 1
    keep = order if not args.batches else [it for it in order if it["batch"] <= args.batches]

    cols = ["order", "batch", "qid", "prompt", "category", "difficulty", "divergence"]
    with open(args.out + ".csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f); w.writerow(cols)
        for it in keep:
            w.writerow([it[c] for c in cols])
    json.dump([{c: it[c] for c in cols} for it in keep],
              open(args.out + ".json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    summarize(order, args.batch, pool)
    print(f"\nwrote {os.path.relpath(args.out, HERE)}.csv/.json  ({len(keep)} rows"
          + (f", first {args.batches} batches)" if args.batches else ")"))


if __name__ == "__main__":
    main()
