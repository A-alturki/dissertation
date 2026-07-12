#!/usr/bin/env python3
"""
validate_train_complete.py
==========================
Release-gate validation + QUALITY analysis of train_corrected.jsonl and its
train_corrected_correction.tsv (plus the raw/snapped sidecars).

PART 1 — HARD INVARIANTS (pass/fail). Structural / byte / cross-file consistency:
  valid jsonl, ids unique + order/answers unchanged vs backup, every incorrect target
  has a well-formed `correction` (list-of-nonempty-str | "خطأ", no null/empty/dupes),
  correct spans carry NO correction, tsv reconciles byte-for-byte with the jsonl,
  no ' ||| ' delimiter collision, no tab/newline in tsv cells, snapped-first, and
  EVERY grounded correction text is genuinely present in the corpus.

PART 2 — QUALITY ANALYSIS (no pass/fail; distributions + flags). The correction call,
the grounding, and the expansion, judged on their merits:
  * correction: verdict + confidence-tier + candidate-count distributions.
  * grounding : grounded/خطأ by ref; snap status mix; SPAN↔CORRECTION fidelity
                (does the correction actually match the span it corrects?);
                cross-ref groundings (quran span -> hadith text, or vice-versa).
  * expansion : list-size distribution; anchor→member coherence (are expanded
                variants really the same matn?); the >=15 per-correction skip.

Writes a JSON report and a FLAGS file (low-fidelity / cross-ref / weak-expansion /
big-list) that the viewer highlights for manual inspection.

Usage:
    python validate_train_complete.py
"""
import os, sys, json, csv, re, argparse
from collections import Counter, defaultdict

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import ground_corrections as G
from hadith_similarity import jaccard, content_tokens

DIR = os.path.join(HERE, "..", "data", "islamiceval2026_RELEASE_correction")
JSONL = os.path.join(DIR, "train_corrected.jsonl")
TSV = os.path.join(DIR, "train_corrected_correction.tsv")
SNAP = os.path.join(DIR, "snapped_model_outputs", "train_corrected_snapped.jsonl")
RAW = os.path.join(DIR, "raw_model_outputs", "train_corrected_raw.jsonl")
BACKUP = os.path.join(HERE, "..", "outputs", "correction", "_verify_src", "train_corrected.jsonl")
REPORT = os.path.join(DIR, "train_corrected_validation_report.json")
FLAGS = os.path.join(DIR, "train_corrected_flags.json")
KHATA = "خطأ"
CORR_SEP = " ||| "
CT = {"Ayah": "quran", "matn": "hadith"}
HADITH_BOOKS = ["البخاري", "مسلم", "داود", "الترمذي", "النسائي", "ماجه"]


def content_seg(ann):
    return next((s for s in ann.get("segments", []) if s.get("type") in CT), None)


def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def contain(span, corr):
    """Fraction of the span's content tokens present in the correction (0..1)."""
    A, B = set(content_tokens(span)), set(content_tokens(corr))
    return (len(A & B) / len(A)) if A else 0.0


def is_hadith_src(sources):
    return any(any(b in s for b in HADITH_BOOKS) for s in (sources or []))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="train_corrected",
                    help="deliverable prefix to validate (e.g. train_corrected, dev_corrected)")
    ap.add_argument("--backup", default=None,
                    help="pristine pre-plug jsonl for the order/answers-unchanged check")
    args = ap.parse_args()
    nm = args.name
    global JSONL, TSV, SNAP, RAW, BACKUP, REPORT, FLAGS
    JSONL = os.path.join(DIR, f"{nm}.jsonl")
    TSV = os.path.join(DIR, f"{nm}_correction.tsv")
    SNAP = os.path.join(DIR, "snapped_model_outputs", f"{nm}_snapped.jsonl")
    RAW = os.path.join(DIR, "raw_model_outputs", f"{nm}_raw.jsonl")
    BACKUP = args.backup or os.path.join(HERE, "..", "outputs", "correction", "_verify_src", f"{nm}.jsonl")
    REPORT = os.path.join(DIR, f"{nm}_validation_report.json")
    FLAGS = os.path.join(DIR, f"{nm}_flags.json")

    checks = []
    def check(name, ok, detail=""):
        checks.append((name, ok, detail))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))

    print(f"validating '{nm}'...", flush=True)
    print("loading corpora...", flush=True)
    quran, mushaf, hadith = G.load_corpus()           # for the concatenated mushaf
    # corpus membership sets (norm_dedup == alif-kept, space-stripped, undiacritized)
    snapc = G.load_snap_corpus()
    hnorm = set(G.norm_dedup(r) for r in snapc["hadith_raw"].values())
    qnorm = set(G.norm_dedup(r) for r in snapc["quran_raw"].values())
    mushaf_norm = mushaf.text                           # concatenated normalized mushaf (norm_ns)

    def grounded_in_corpus(text, ref):
        k = G.norm_dedup(text)
        if k in hnorm or k in qnorm:
            return True
        # quran multi-ayah join: contiguous in the concatenated mushaf
        cn = G.norm_ns(text)
        return bool(cn) and cn in mushaf_norm

    # ── load everything ──────────────────────────────────────────────────────
    print("loading files...", flush=True)
    resp = load_jsonl(JSONL)
    snaps = {(o["response_id"], o["annotation_id"]): o for o in load_jsonl(SNAP)}
    raws = {(o["response_id"], o["annotation_id"]): o for o in load_jsonl(RAW)}
    backup = load_jsonl(BACKUP) if os.path.exists(BACKUP) else None

    # ══════════════════════════════════════════════════════════════════════════
    print("\n================ PART 1 — HARD INVARIANTS ================")
    ids = [r["id"] for r in resp]
    check("response ids unique", len(ids) == len(set(ids)), f"{len(ids)} responses")
    if backup:
        check("order + response ids identical to backup", ids == [r["id"] for r in backup])
        check("generated_answer bytes unchanged vs backup",
              all(a.get("generated_answer") == b.get("generated_answer") for a, b in zip(resp, backup)))

    n_incorrect = n_correct = 0
    corr_field = {}          # (rid,aid) -> correction value (list|"خطأ")
    bad_type = dup_in_list = empty_entry = delim_hit = corr_on_correct = 0
    span_oob = 0
    for r in resp:
        rid = r["id"]; ans = r.get("generated_answer", "") or ""
        for ann in r.get("annotations", []):
            aid = ann["annotation_id"]; cs = content_seg(ann); c = ann.get("correction", "__none__")
            is_incorrect = cs is not None and cs.get("label") == "incorrect"
            if is_incorrect:
                n_incorrect += 1
                corr_field[(rid, aid)] = c
                a, b = cs.get("span_start"), cs.get("span_end")
                if not (isinstance(a, int) and isinstance(b, int) and 0 <= a <= b <= len(ans)):
                    span_oob += 1
                if c == KHATA:
                    pass
                elif isinstance(c, list) and c and all(isinstance(x, str) and x.strip() for x in c):
                    if len(c) != len(set(c)): dup_in_list += 1
                    if any(CORR_SEP in x for x in c): delim_hit += 1
                else:
                    bad_type += 1
                    if isinstance(c, list) and any((not isinstance(x, str) or not x.strip()) for x in c):
                        empty_entry += 1
            else:
                if cs and cs.get("label") == "correct":
                    n_correct += 1
                if "correction" in ann and not is_incorrect:
                    corr_on_correct += 1

    check("every incorrect target has a correction", "__none__" not in corr_field.values(),
          f"{n_incorrect} incorrect targets")
    check("correction is list-of-nonempty-str | خطأ (no null/empty)", bad_type == 0, f"{bad_type} malformed")
    check("no duplicate entry within a correction list", dup_in_list == 0, f"{dup_in_list} lists w/ dupes")
    check("no ' ||| ' delimiter inside any correction text", delim_hit == 0, f"{delim_hit} collisions")
    check("no 'correction' field on non-incorrect annotations", corr_on_correct == 0, f"{corr_on_correct} leaks")
    check("all incorrect span offsets in-bounds", span_oob == 0, f"{span_oob} out-of-bounds")

    grounded = [(k, v) for k, v in corr_field.items() if v != KHATA]
    khata = [k for k, v in corr_field.items() if v == KHATA]
    # corpus membership of EVERY grounded correction text
    ungrounded_txt = []
    for (rid, aid), lst in grounded:
        ref = snaps[(rid, aid)]["ref"]
        for t in lst:
            if not grounded_in_corpus(t, ref):
                ungrounded_txt.append((rid, aid, t[:50]))
    check("every grounded correction text is present in the corpus", not ungrounded_txt,
          f"{len(ungrounded_txt)} not found" + (f" e.g. {ungrounded_txt[0]}" if ungrounded_txt else ""))

    # snapped-first: jsonl[0] == snapped sidecar corrections[0]
    sf_bad = 0
    for (rid, aid), lst in grounded:
        s = snaps.get((rid, aid))
        if s and s.get("corrections") and s["corrections"][0] != lst[0]:
            sf_bad += 1
    check("snapped-anchor is first (jsonl[0]==snapped[0])", sf_bad == 0, f"{sf_bad} mismatches")
    check("snapped sidecar NULL count == خطأ count",
          sum(1 for o in snaps.values() if o.get("corrections") is None) == len(khata),
          f"{sum(1 for o in snaps.values() if o.get('corrections') is None)} vs {len(khata)}")

    # ── TSV reconciliation ────────────────────────────────────────────────────
    with open(TSV, encoding="utf-8", newline="") as f:
        tsv = list(csv.reader(f, delimiter="\t"))
    thdr, tbody = tsv[0], tsv[1:]
    check("tsv header exact", thdr == ["Response_ID", "Annotation_ID", "Segment_Type", "Correction"], str(thdr))
    check("tsv every row has 4 columns", all(len(r) == 4 for r in tbody))
    check("tsv row count == incorrect targets", len(tbody) == n_incorrect, f"{len(tbody)} vs {n_incorrect}")
    tsv_keys = [(r[0], int(r[1])) for r in tbody]
    check("tsv (Response_ID,Annotation_ID) unique", len(tsv_keys) == len(set(tsv_keys)))
    check("tsv Segment_Type in {Ayah,matn}", all(r[2] in ("Ayah", "matn") for r in tbody))
    mismatch = 0
    for r in tbody:
        key = (r[0], int(r[1])); jc = corr_field.get(key)
        cell = KHATA if jc == KHATA else CORR_SEP.join(jc) if isinstance(jc, list) else None
        if cell != r[3]:
            mismatch += 1
    check("tsv Correction cell == jsonl correction (byte-exact)", mismatch == 0, f"{mismatch} mismatches")
    check("tsv no empty Correction cell", all(r[3].strip() for r in tbody))

    # ── encoding ──────────────────────────────────────────────────────────────
    raw_bytes = open(JSONL, encoding="utf-8").read()
    check("no U+FFFD replacement chars in jsonl", "�" not in raw_bytes)

    n_pass = sum(1 for _, ok, _ in checks if ok)
    print(f"\n  {n_pass}/{len(checks)} hard invariants PASSED"
          + ("" if n_pass == len(checks) else "   <-- FAILURES ABOVE"))

    # ══════════════════════════════════════════════════════════════════════════
    print("\n================ PART 2 — QUALITY ANALYSIS ================")
    FRONTIER = lambda rid: rid >= "R004844"      # frontier ids start here (topup)

    # --- correction (raw model) ---
    verdict_by = defaultdict(Counter); conf = Counter(); ncand = Counter()
    for (rid, aid), o in raws.items():
        seg_ref = o["ref"]
        v = o.get("verdict")
        verdict_by[seg_ref][v] += 1
        verdict_by["ALL"][v] += 1
        verdict_by["frontier" if FRONTIER(rid) else "oldtrain"][v] += 1
        cands = o.get("candidates") or []
        ncand[len(cands)] += 1
        for c in cands:
            conf[c.get("confidence")] += 1
    print("\n[correction] GPT verdict (ok/خطأ):")
    for k in ("ALL", "quran", "hadith", "oldtrain", "frontier"):
        d = verdict_by[k]; tot = sum(d.values())
        if tot:
            print(f"   {k:9} ok={d.get('ok',0):5} خطأ={d.get('خطأ',0):5}  ok%={100*d.get('ok',0)/tot:5.1f}")
    print(f"   candidate-count dist: {dict(sorted(ncand.items()))}")
    print(f"   candidate confidence tiers: {dict(conf)}")

    # --- grounding ---
    gr_by_ref = Counter(); kh_by_ref = Counter(); status_mix = Counter(); crossref = []
    fidelity = []            # (rid,aid,ref,containment,span,corr0)
    for (rid, aid), lst in grounded:
        s = snaps[(rid, aid)]; ref = s["ref"]; span = s["span"]
        gr_by_ref[ref] += 1
        for d in s["grounding"]:
            status_mix[d.get("status")] += 1
        top_src = (s["grounding"][0]["sources"] if s["grounding"] else [])
        # cross-ref: span ref vs corpus source type of the anchor
        anchor_is_hadith = is_hadith_src(top_src)
        if (ref == "quran" and anchor_is_hadith) or (ref == "hadith" and not anchor_is_hadith and top_src):
            crossref.append((rid, aid, ref, top_src[:1], span[:60], lst[0][:60]))
        c = contain(span, lst[0])
        fidelity.append((rid, aid, ref, round(c, 3), span, lst[0]))
    for k in khata:
        kh_by_ref[snaps[k]["ref"]] += 1
    print(f"\n[grounding] grounded by ref: {dict(gr_by_ref)} | خطأ by ref: {dict(kh_by_ref)}")
    print(f"   snap status mix (grounded=char-substring, snapped=jaccard/char-ratio): {dict(status_mix)}")
    print(f"   cross-ref groundings (span type != correction corpus type): {len(crossref)}")

    fid_vals = [f[3] for f in fidelity]
    def band(lo, hi): return sum(1 for v in fid_vals if lo <= v < hi)
    hi = sum(1 for v in fid_vals if v >= 0.60); mid = band(0.30, 0.60); lo = sum(1 for v in fid_vals if v < 0.30)
    import statistics
    print(f"\n[grounding] SPAN↔CORRECTION fidelity (span content-tokens covered by correction[0]):")
    print(f"   mean={statistics.mean(fid_vals):.3f} median={statistics.median(fid_vals):.3f}"
          f"  | high(>=.60)={hi}  mid(.30-.60)={mid}  low(<.30)={lo}")
    by_ref_fid = defaultdict(list)
    for f in fidelity: by_ref_fid[f[2]].append(f[3])
    for ref, vs in by_ref_fid.items():
        print(f"     {ref}: mean={statistics.mean(vs):.3f}  low(<.30)={sum(1 for v in vs if v<0.30)}")

    # --- expansion ---
    sizes = Counter(); weak_expand = []; coh_min = []
    for (rid, aid), lst in grounded:
        sizes[len(lst)] += 1
        if len(lst) > 1:
            anchor = lst[0]
            js = [jaccard(anchor, m) for m in lst[1:]]
            mn = min(js) if js else 1.0
            coh_min.append(mn)
            if mn < 0.30:                       # an expansion member weakly related to anchor
                weak_expand.append((rid, aid, round(mn, 3), snaps[(rid, aid)]["ref"], anchor[:50]))
    print(f"\n[expansion] correction-list size dist (grounded): {dict(sorted(sizes.items()))}")
    if coh_min:
        print(f"   anchor→member coherence (min jaccard within a list): mean={statistics.mean(coh_min):.3f}"
              f"  lists with a member <0.30 to anchor: {len(weak_expand)}")
    print(f"   lists >=15 (multi-correction): {sum(v for k,v in sizes.items() if k>=15)}")

    # ── FLAGS for the viewer ──────────────────────────────────────────────────
    low_fid = sorted([f for f in fidelity if f[3] < 0.30], key=lambda x: x[3])
    flags = {
        "low_fidelity": [{"id": f"{r}:a{a}", "ref": ref, "containment": c,
                          "span": sp[:120], "correction": co[:120]} for r, a, ref, c, sp, co in low_fid],
        "cross_ref": [{"id": f"{r}:a{a}", "span_ref": ref, "anchor_source": src,
                       "span": sp, "correction": co} for r, a, ref, src, sp, co in crossref],
        "weak_expansion": [{"id": f"{r}:a{a}", "ref": ref, "min_coherence": mn, "anchor": an}
                           for r, a, mn, ref, an in weak_expand],
    }
    json.dump(flags, open(FLAGS, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    report = {
        "hard_invariants": {"passed": n_pass, "total": len(checks),
                            "failures": [n for n, ok, _ in checks if not ok]},
        "counts": {"responses": len(resp), "incorrect_targets": n_incorrect,
                   "correct_spans": n_correct, "grounded": len(grounded), "khata": len(khata)},
        "verdict": {k: dict(v) for k, v in verdict_by.items()},
        "grounded_by_ref": dict(gr_by_ref), "khata_by_ref": dict(kh_by_ref),
        "snap_status": dict(status_mix),
        "fidelity": {"mean": round(statistics.mean(fid_vals), 3), "high": hi, "mid": mid, "low": lo},
        "expansion_sizes": dict(sorted(sizes.items())),
        "flag_counts": {k: len(v) for k, v in flags.items()},
    }
    json.dump(report, open(REPORT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\nFLAGS for manual review: low-fidelity={len(flags['low_fidelity'])} "
          f"cross-ref={len(flags['cross_ref'])} weak-expansion={len(flags['weak_expansion'])}")
    print(f"  -> {os.path.relpath(FLAGS, HERE)}  +  {os.path.relpath(REPORT, HERE)}")
    print(f"\n{'='*60}\nRELEASE GATE: {'ALL HARD INVARIANTS PASS' if n_pass==len(checks) else 'FAILURES PRESENT — DO NOT RELEASE'}")


if __name__ == "__main__":
    main()
