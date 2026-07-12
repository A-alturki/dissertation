#!/usr/bin/env python3
"""
build_task4_relevance.py
========================
(Re)generate the Task 4 (relevance) TSV for a correction deliverable, from the
already-plugged <split>.jsonl + <split>_snapped.jsonl (no re-grounding needed).

An answer contributes rows ONLY if every content span in it is correct OR was
successfully corrected — i.e. it carries NO residual (uncorrectable) hallucination.
If ANY incorrect span is خطأ / null (uncorrectable), the WHOLE answer is dropped.
For a qualifying ("clean-after-correction") answer, ALL of its content spans are
emitted:
  * CORRECT spans    — Span_Text = the verbatim answer slice.
  * CORRECTED spans  — Span_Text = the highest-probability snapped correction.
      Cross-corpus groundings (span typed one corpus, grounded to the other) are
      the one exception: they are EXCLUDED from the relevance rows.

Columns (Span_Type = Ayah|matn):
  Question_ID, Question, Response_ID, Generated_Answer, Annotation_ID,
  Span_Type, Span_Text, Relevance_Label   (Relevance_Label left empty)

Usage:
    python build_task4_relevance.py                       # train_corrected + dev_corrected
    python build_task4_relevance.py --names dev_corrected
"""
import os, sys, json, csv, argparse

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.join(HERE, "..", "data", "islamiceval2026_RELEASE_correction")

CONTENT_TYPES = {"Ayah": "quran", "matn": "hadith"}
HADITH_BOOKS = ("البخاري", "مسلم", "داود", "الترمذي", "النسائي", "ماجه")
HEADER = ["Question_ID", "Question", "Response_ID", "Generated_Answer",
          "Annotation_ID", "Span_Type", "Span_Text", "Relevance_Label"]


def source_corpus(sources):
    s = (sources or [""])[0]
    return "hadith" if any(b in s for b in HADITH_BOOKS) else "quran"


def content_seg(ann):
    return next((s for s in ann.get("segments", []) if s.get("type") in CONTENT_TYPES), None)


def cell(x):
    return (x or "").replace("\t", " ").replace("\r", " ").replace("\n", " ").strip()


def relevance_rows(responses, snap_by_key):
    """responses = plugged jsonl objects; snap_by_key[(rid,aid)] = snapped record.
    Returns (rows, n_correct, n_corrected)."""
    rows = []; n_correct = n_corrected = 0
    for r in responses:
        rid = r["id"]; qid = r.get("question_id", ""); q = r.get("question", "")
        ans = r.get("generated_answer", "") or ""
        segs = [(ann, content_seg(ann)) for ann in r.get("annotations", [])]
        # an answer is "uncorrectable" if ANY incorrect span could not be grounded
        # (correction is خطأ / null / empty) — such answers still carry an unfixable
        # hallucination, so they contribute NO relevance rows at all. An answer whose
        # every content span is correct OR corrected is "clean" -> emit ALL its spans.
        answer_has_uncorrectable = any(
            cs is not None and cs.get("label") == "incorrect"
            and not (isinstance(ann.get("correction"), list) and ann.get("correction"))
            for ann, cs in segs)
        if answer_has_uncorrectable:
            continue
        for ann, cs in segs:
            if cs is None:
                continue
            aid = ann["annotation_id"]; seg_type = cs["type"]           # Ayah | matn
            label = cs.get("label")
            if label == "correct":
                a, b = cs.get("span_start"), cs.get("span_end")
                txt = ans[a:b] if isinstance(a, int) and isinstance(b, int) else ""
                rows.append([qid, cell(q), rid, cell(ans), aid, seg_type, cell(txt), ""])
                n_correct += 1
            elif label == "incorrect":
                corr = ann.get("correction")
                if not isinstance(corr, list) or not corr:              # خطأ / null -> not judgeable
                    continue
                snap = snap_by_key.get((rid, aid))
                g = (snap.get("grounding") or [{}])[0].get("sources") if snap else None
                if g and source_corpus(g) != CONTENT_TYPES[seg_type]:   # cross-corpus -> exclude
                    continue
                rows.append([qid, cell(q), rid, cell(ans), aid, seg_type, cell(corr[0]), ""])
                n_corrected += 1
    return rows, n_correct, n_corrected


def run(name):
    jsonl = os.path.join(DIR, f"{name}.jsonl")
    snapf = os.path.join(DIR, "snapped_model_outputs", f"{name}_snapped.jsonl")
    out = os.path.join(DIR, f"{name}_relevance.tsv")
    responses = [json.loads(l) for l in open(jsonl, encoding="utf-8") if l.strip()]
    snap_by_key = {(o["response_id"], o["annotation_id"]): o
                   for o in (json.loads(l) for l in open(snapf, encoding="utf-8") if l.strip())}
    rows, n_correct, n_corrected = relevance_rows(responses, snap_by_key)
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(HEADER)
        w.writerows(rows)
    print(f"  {name}_relevance.tsv: {len(rows)} rows "
          f"({n_correct} correct-from-clean-answers + {n_corrected} corrected) -> {os.path.relpath(out, HERE)}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", nargs="+", default=["train_corrected", "dev_corrected"])
    args = ap.parse_args()
    for nm in args.names:
        run(nm)


if __name__ == "__main__":
    main()
