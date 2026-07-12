#!/usr/bin/env python3
"""
extract_release_incorrect_spans.py
===================================
Pull every INCORRECT citation span from the IslamicEval2026 RELEASE jsonl files
(train / dev / test) so the correction one-shot pipeline can run on them, and keep
a PROVENANCE map so each correction plugs back into the exact annotation later.

RELEASE jsonl schema (one object per response):
    {id, question_id, question, generated_answer,
     annotations: [{annotation_id, label(Ayah|Hadith), correction,
                    segments: [{type(Ayah|matn|isnad|claimed_source),
                                span_start, span_end, label(correct|incorrect|N/A)}]}]}

Each annotation has AT MOST ONE content segment (Ayah or matn). An annotation is a
correction target iff that content segment's label == "incorrect". The span text is
generated_answer[span_start:span_end]; ref = quran (Ayah) / hadith (matn). The
annotation's claimed_source segment text (if present) is carried for the Quran
reference-fallback tier at grounding time.

The join key is the stable `id` = "<split>:<response_id>:a<annotation_id>" (unique
per correction target, ORDER-INDEPENDENT). The batch pipeline preserves this `id` in
its per_sample output, so plug-back joins on it — NOT on the CSV-order-dependent uid.

Outputs (to outputs/correction/):
  * release_incorrect_spans.csv           — combined (all splits)
  * release_incorrect_spans_<split>.csv   — per-split, for staged submits
    columns: id,model,span,ref,split,response_id,annotation_id,claimed_source
    (id/model/span/ref are exactly what correction_oneshot_batch_openai expects;
     the extra provenance columns are ignored by the batch loader.)
  * release_span_provenance.json — id -> {split,response_id,annotation_id,ref,
     span,claimed_source}.

Usage:
    python extract_release_incorrect_spans.py
    python extract_release_incorrect_spans.py --release-dir ../data/islamiceval2026_RELEASE
"""
import os, csv, sys, json, argparse

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

SPLITS = ["train", "dev", "test"]
COLS = ["id", "model", "span", "ref", "split", "response_id", "annotation_id", "claimed_source"]
CONTENT_TYPES = {"Ayah": "quran", "matn": "hadith"}


def content_seg(ann):
    """The single Ayah/matn segment of an annotation, or None."""
    for s in ann.get("segments", []):
        if s.get("type") in CONTENT_TYPES:
            return s
    return None


def claimed_seg_text(ann, answer):
    for s in ann.get("segments", []):
        if s.get("type") == "claimed_source":
            a, b = s.get("span_start"), s.get("span_end")
            if isinstance(a, int) and isinstance(b, int):
                return answer[a:b]
    return ""


def extract(release_dir, out_csv, out_prov):
    rows = []          # dict rows for the CSV (in stable order = split, then file order)
    prov = {}          # id -> provenance dict
    n_by = {}
    for split in SPLITS:
        path = os.path.join(release_dir, f"{split}.jsonl")
        if not os.path.exists(path):
            print(f"  ! missing {path}, skipping")
            continue
        n = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                answer = r.get("generated_answer", "") or ""
                rid = r["id"]
                for ann in r.get("annotations", []):
                    cs = content_seg(ann)
                    if not cs or cs.get("label") != "incorrect":
                        continue
                    a, b = cs.get("span_start"), cs.get("span_end")
                    if not (isinstance(a, int) and isinstance(b, int)):
                        continue
                    span = answer[a:b]
                    ref = CONTENT_TYPES[cs["type"]]
                    aid = ann["annotation_id"]
                    claimed = claimed_seg_text(ann, answer)
                    uniq_id = f"{split}:{rid}:a{aid}"      # unique per correction target
                    rows.append({
                        "id": uniq_id, "model": split, "span": span, "ref": ref,
                        "split": split, "response_id": rid, "annotation_id": aid,
                        "claimed_source": claimed,
                    })
                    prov[uniq_id] = {
                        "split": split, "response_id": rid, "annotation_id": aid,
                        "ref": ref, "span": span, "claimed_source": claimed,
                    }
                    n += 1
        n_by[split] = n
        print(f"  {split}: {n} incorrect spans")

    def write_csv(path, subset):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLS, quoting=csv.QUOTE_ALL)
            w.writeheader()
            w.writerows(subset)

    write_csv(out_csv, rows)
    print(f"\n  wrote {out_csv}  ({len(rows)} rows)")
    base, ext = os.path.splitext(out_csv)
    for split in SPLITS:
        sub = [r for r in rows if r["split"] == split]
        if sub:
            p = f"{base}_{split}{ext}"
            write_csv(p, sub)
            print(f"  wrote {os.path.basename(p)}  ({len(sub)} rows)")

    if len(prov) != len(rows):
        print(f"  ! WARNING: provenance {len(prov)} != rows {len(rows)} (duplicate ids?)")
    json.dump(prov, open(out_prov, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"  wrote {out_prov}  ({len(prov)} ids)")

    # sanity preview
    print("\n  sample spans (verify Arabic offsets look correct):")
    for row in rows[:3] + rows[len(rows)//2: len(rows)//2 + 2]:
        print(f"    [{row['ref']:6}] {row['id']}  claim='{row['claimed_source']}'")
        print(f"           span: {row['span'][:80]}")
    ref_counts = {}
    for row in rows:
        ref_counts[row["ref"]] = ref_counts.get(row["ref"], 0) + 1
    print(f"\n  totals: {ref_counts} | per split {n_by}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release-dir", default=os.path.join(HERE, "..", "data", "islamiceval2026_RELEASE"))
    ap.add_argument("--out-csv", default=os.path.join(HERE, "..", "outputs", "correction", "release_incorrect_spans.csv"))
    ap.add_argument("--out-prov", default=os.path.join(HERE, "..", "outputs", "correction", "release_span_provenance.json"))
    args = ap.parse_args()
    extract(args.release_dir, args.out_csv, args.out_prov)


if __name__ == "__main__":
    main()
