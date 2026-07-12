#!/usr/bin/env python3
"""
verify_release_deliverables.py
==============================
Independent invariant checks over the folder produced by build_release_deliverables.py.
Re-derives everything from the ORIGINAL RELEASE jsonl + provenance so it cannot inherit
a bug from the builder. Prints PASS/FAIL per check.

Usage:
    python verify_release_deliverables.py                 # train + dev
    python verify_release_deliverables.py --splits dev
"""
import os, sys, json, csv, argparse

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.abspath(__file__))
KHATA = "خطأ"
CONTENT_TYPES = {"Ayah": "quran", "matn": "hadith"}
CORR_SEP = " ||| "


def content_seg(ann):
    for s in ann.get("segments", []):
        if s.get("type") in CONTENT_TYPES:
            return s
    return None


def load_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def read_tsv(path):
    with open(path, encoding="utf-8", newline="") as f:
        r = csv.reader(f, delimiter="\t")
        rows = list(r)
    return rows[0], rows[1:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["train", "dev"])
    ap.add_argument("--release-dir", default=os.path.join(HERE, "..", "data", "islamiceval2026_RELEASE"))
    ap.add_argument("--out-dir", default=os.path.join(HERE, "..", "data", "islamiceval2026_RELEASE_correction"))
    args = ap.parse_args()

    checks = []
    def check(name, ok, detail=""):
        checks.append(ok)
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))

    for split in args.splits:
        print(f"\n=== {split} ===")
        orig = load_jsonl(os.path.join(args.release_dir, f"{split}.jsonl"))
        corr = load_jsonl(os.path.join(args.out_dir, f"{split}.jsonl"))
        orig_by = {r["id"]: r for r in orig}

        # 1. same set of response ids, same order, same answers (only `correction` changed)
        check("response id set + order unchanged", [r["id"] for r in orig] == [r["id"] for r in corr])
        answers_ok = all(o.get("generated_answer") == c.get("generated_answer")
                         for o, c in zip(orig, corr))
        check("generated_answer bytes unchanged", answers_ok)

        # 2. gather incorrect targets from ORIGINAL, tally correction field in CORRECTED
        n_incorrect = n_correct = 0
        for r in orig:
            for ann in r.get("annotations", []):
                cs = content_seg(ann)
                if not cs:
                    continue
                if cs.get("label") == "incorrect":
                    n_incorrect += 1
                elif cs.get("label") == "correct":
                    n_correct += 1

        grounded = khata = null_left = nonlist = other_ann_touched = 0
        snapped_first_ok = True
        snap_map = {}
        # load snapped model outputs for cross-check
        for line in open(os.path.join(args.out_dir, "snapped_model_outputs", f"{split}_snapped.jsonl"),
                         encoding="utf-8"):
            o = json.loads(line)
            snap_map[(o["response_id"], o["annotation_id"])] = o

        for r in corr:
            for ann in r.get("annotations", []):
                cs = content_seg(ann)
                is_target = cs is not None and orig_by[r["id"]] and \
                    any(content_seg(a) and content_seg(a).get("label") == "incorrect" and a["annotation_id"] == ann["annotation_id"]
                        for a in orig_by[r["id"]]["annotations"])
                if is_target:
                    c = ann.get("correction")
                    if c == KHATA:
                        khata += 1
                    elif isinstance(c, list) and c and all(isinstance(x, str) for x in c):
                        grounded += 1
                        # snapped-first cross-check: jsonl[0] == snapped_output.corrections[0]
                        s = snap_map.get((r["id"], ann["annotation_id"]))
                        if s and s.get("corrections"):
                            if s["corrections"][0] != c[0]:
                                snapped_first_ok = False
                    elif c is None:
                        null_left += 1
                    else:
                        nonlist += 1

        check("no NULL left in participant jsonl (null->خطأ)", null_left == 0, f"{null_left} nulls")
        check("every correction is list|خطأ", nonlist == 0, f"{nonlist} malformed")
        check("grounded+خطأ == #incorrect targets", grounded + khata == n_incorrect,
              f"{grounded}+{khata} vs {n_incorrect}")
        check("jsonl[0] == snapped anchor (snapped-first)", snapped_first_ok)

        # 3. snapped model outputs: NULL exactly when jsonl is خطأ
        snap_null = sum(1 for o in snap_map.values() if o.get("corrections") is None)
        check("snapped-output NULL count == خطأ count", snap_null == khata, f"{snap_null} vs {khata}")

        # 4. correction.tsv: one row per incorrect target, matches jsonl
        _, ctr = read_tsv(os.path.join(args.out_dir, f"{split}_correction.tsv"))
        check("correction.tsv row count == #incorrect targets", len(ctr) == n_incorrect,
              f"{len(ctr)} vs {n_incorrect}")
        tsv_khata = sum(1 for row in ctr if row[3] == KHATA)
        check("correction.tsv خطأ count == jsonl خطأ", tsv_khata == khata, f"{tsv_khata} vs {khata}")

        # 5. relevance.tsv: rows == correct + grounded-corrected; no خطأ text; Relevance empty
        _, rtr = read_tsv(os.path.join(args.out_dir, f"{split}_relevance.tsv"))
        rel_correct = sum(1 for row in rtr if row[7] == "correct")
        rel_corrected = sum(1 for row in rtr if row[7] == "corrected")
        check("relevance correct rows == #correct spans", rel_correct == n_correct,
              f"{rel_correct} vs {n_correct}")
        check("relevance corrected rows == grounded", rel_corrected == grounded,
              f"{rel_corrected} vs {grounded}")
        check("relevance Relevance column all empty", all((len(row) < 9 or row[8] == "") for row in rtr))
        check("relevance no خطأ / empty span text", all(row[6].strip() and row[6] != KHATA for row in rtr))
        check("relevance types in {quran,hadith}", all(row[5] in ("quran", "hadith") for row in rtr))

        # 6. no tabs/newlines broke any TSV row width
        for nm, hdr, body in [("correction", *read_tsv(os.path.join(args.out_dir, f"{split}_correction.tsv"))),
                              ("relevance", *read_tsv(os.path.join(args.out_dir, f"{split}_relevance.tsv")))]:
            check(f"{nm}.tsv uniform column count", all(len(row) == len(hdr) for row in body))

    print(f"\n{'='*50}\n{sum(checks)}/{len(checks)} checks PASSED"
          + ("" if all(checks) else "  <-- FAILURES PRESENT"))
    sys.exit(0 if all(checks) else 1)


if __name__ == "__main__":
    main()
