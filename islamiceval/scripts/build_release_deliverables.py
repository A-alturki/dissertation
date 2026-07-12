#!/usr/bin/env python3
"""
build_release_deliverables.py
=============================
Assemble the FINAL IslamicEval2026 correction + relevance deliverables for the
RELEASE train/dev splits, under ONE folder.

Pipeline (LOCKED config, 2026-07-08 — see ground_corrections.py):
    raw one-shot GPT-5.4 corrections  (pipeline_oneshot_release_<split>.json)
      -> GROUND each INCORRECT span:  jaccard-snap anchor (SNAP_JAC_TH 0.40)
                                       -> dedup -> jaccard-expand (EXPAND_JAC_TH 0.35)
                                       -> if an anchor explodes to >= EXPAND_SKIP_GE(15)
                                          distinct variants, DO NOT expand (anchor only)
      -> correction list, SNAPPED ANCHOR FIRST, exact-dup matns collapsed
      -> nothing grounds  =>  NULL (snapped model outputs)  /  "خطأ" (participant files)

Deliverables written to <out-dir> (default data/islamiceval2026_RELEASE_correction/):
  <split>.jsonl                    RELEASE jsonl with `correction` filled
                                   (grounded -> list snapped-first; ungrounded -> "خطأ")
  <split>_correction.tsv           Response_ID, Annotation_ID, Segment_Type, Correction
                                   (list joined by ' ||| '; ungrounded -> "خطأ")
  <split>_relevance.tsv            question+answer rows for every CORRECT or CORRECTED
                                   span: the highest-probability (snapped) text + its
                                   type {quran|hadith} + an empty Relevance column
  raw_model_outputs/<split>_raw.jsonl        per incorrect span: verdict/rationale/candidates
  snapped_model_outputs/<split>_snapped.jsonl per incorrect span: corrections (list|null) + audit
  README.md, build_stats.json

Join key = the stable id "<split>:<response_id>:a<annotation_id>".

Usage:
    python build_release_deliverables.py                 # train + dev
    python build_release_deliverables.py --splits dev    # dev only
"""
import os, sys, json, csv, re, argparse
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import ground_corrections as G
from build_task4_relevance import relevance_rows as task4_relevance_rows, HEADER as REL_HEADER

GPT = "gpt-5.4"
KHATA = "خطأ"
CONTENT_TYPES = {"Ayah": "quran", "matn": "hadith"}
CORR_SEP = " ||| "
HADITH_BOOKS = ("البخاري", "مسلم", "داود", "الترمذي", "النسائي", "ماجه")

def _source_corpus(sources):
    """Corpus a grounded correction's first source belongs to ('quran' | 'hadith')."""
    s = (sources or [""])[0]
    return "hadith" if any(b in s for b in HADITH_BOOKS) else "quran"


def set_locked_config():
    """Pin the LOCKED grounding config explicitly (defensive — independent of defaults)."""
    G.SNAP_METRIC   = "jaccard"
    G.SNAP_JAC_TH   = 0.40
    G.SNAP_CHAR_FALLBACK_TH = 0.85   # jaccard fails -> char-snap at this lower threshold
    G.EXPAND_METRIC = "jaccard"
    G.EXPAND_JAC_TH = 0.35
    G.EXPAND_MAX    = 0            # no hard cap; superseded by the skip rule
    G.EXPAND_SKIP_GE = 15         # anchor exploding to >=15 variants -> no expansion
    G.HADITH_EXPAND = True
    G.ALEF_DEL      = True
    G.PARTIAL       = True
    return {"SNAP_METRIC": G.SNAP_METRIC, "SNAP_JAC_TH": G.SNAP_JAC_TH,
            "SNAP_CHAR_FALLBACK_TH": G.SNAP_CHAR_FALLBACK_TH,
            "EXPAND_METRIC": G.EXPAND_METRIC, "EXPAND_JAC_TH": G.EXPAND_JAC_TH,
            "EXPAND_MAX": G.EXPAND_MAX, "EXPAND_SKIP_GE": G.EXPAND_SKIP_GE,
            "HADITH_EXPAND": G.HADITH_EXPAND, "ALEF_DEL": G.ALEF_DEL, "PARTIAL": G.PARTIAL}


_WS = re.compile(r"\s+")
# tashkeel + tatweel + Quranic annotation marks (base letters kept)
_DIAC = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭـ]")
def clean_correction(t):
    """Tidy a grounded correction text: collapse all whitespace runs to a single space and
    strip. Corpus matns carry stray double/trailing spaces; normalizing here keeps the jsonl
    correction and the flattened TSV cell BYTE-IDENTICAL and is whitespace-insensitive so it
    never changes grounding (norm_dedup ignores spaces)."""
    return _WS.sub(" ", t or "").strip()

def strip_diacritics(t):
    """Remove tashkeel/tatweel (base letters kept). Applied to HADITH corrections only —
    Quran corrections keep full Uthmani diacritics (scripture-exact); hadith matn
    diacritization is non-authoritative, so we ship it undiacritized (Quran-strict /
    Hadith-lenient convention)."""
    return _WS.sub(" ", _DIAC.sub("", t or "")).strip()


def tsv_clean(x):
    """Make a text field safe for a TSV cell (no tab/CR/LF)."""
    return (x or "").replace("\t", " ").replace("\r", " ").replace("\n", " ").strip()


def _dedup_rows(rows):
    """[(text, sources, status)] -> [{text, sources, status}] collapsing exact-dup matns
    (norm_dedup, alif-kept) and unioning their sources, first-seen order preserved."""
    seen, out = {}, []
    for t, s, st in rows:
        k = G.norm_dedup(t)
        if not k:
            continue
        if k in seen:
            for src in s:
                if src not in seen[k]["sources"]:
                    seen[k]["sources"].append(src)
            continue
        d = {"text": t, "sources": list(s), "status": st}
        seen[k] = d
        out.append(d)
    return out


def ground_span(pred, ref, claimed_source, alef, snapc):
    """Ground one span's raw prediction -> (texts, detail), corrections SNAPPED-FIRST.
    snap -> dedup -> expand PER CORRECTION (resolve_correction is snapped-anchor-first and
    already applies the PER-ANCHOR skip: if a single correction expands to >= EXPAND_SKIP_GE
    corpus variants, only its snapped anchor is kept — no expansion). Multiple corrections
    for one span are concatenated (each may contribute its own expanded family) and exact-dup
    matns collapsed. texts=[] means nothing grounded."""
    G.ALEF_DEL = True
    G.PARTIAL = True
    if not pred or pred.get("verdict") != "ok":
        return [], []
    rows = []
    for c in (pred.get("candidates") or []):
        r = G.resolve_correction(c.get("text", ""), ref, c.get("source", ""), *alef, snapc)
        if r["status"] == "eliminated":
            continue
        for vt, vs in (r.get("variants") or [(r["gold_text"], r["sources"])]):
            vt = clean_correction(vt)
            if vt:
                rows.append((vt, vs, r["status"]))
    detail = _dedup_rows(rows)
    return [d["text"] for d in detail], detail


def load_raw(path):
    """pipeline json -> {id: gpt prediction dict|None}."""
    data = json.load(open(path, encoding="utf-8"))
    return {s["id"]: (s.get("oneshot") or {}).get(GPT, {}).get("prediction") for s in data["per_sample"]}


def content_seg(ann):
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


def write_tsv(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


def process_split(name, release_file, raw_paths, out_dir, alef, snapc, id_split=None):
    """Self-contained: incorrect correction targets are extracted from `release_file`
    itself (no provenance file); raw GPT predictions are MERGED from raw_paths and looked
    up by the stable id "<id_split>:<response_id>:a<annotation_id>". Writes every deliverable
    with the `name` prefix. id_split defaults to name (e.g. train_corrected uses id_split=train
    so ids match the existing raw preds)."""
    id_split = id_split or name
    raw = {}
    for p in raw_paths:
        raw.update(load_raw(p))
    responses = [json.loads(l) for l in open(release_file, encoding="utf-8") if l.strip()]

    # targets straight from the release file
    targets = []  # (sid, rid, aid, ref, span, claimed, seg_type)
    for r in responses:
        rid = r["id"]; ans = r.get("generated_answer", "") or ""
        for ann in r.get("annotations", []):
            cs = content_seg(ann)
            if not cs or cs.get("label") != "incorrect":
                continue
            a, b = cs.get("span_start"), cs.get("span_end")
            if not (isinstance(a, int) and isinstance(b, int)):
                continue
            targets.append((f"{id_split}:{rid}:a{ann['annotation_id']}", rid, ann["annotation_id"],
                            CONTENT_TYPES[cs["type"]], ans[a:b], claimed_seg_text(ann, ans), cs["type"]))
    missing = sum(1 for t in targets if t[0] not in raw)
    print(f"\n[{name}] raw preds merged={len(raw)}  incorrect targets={len(targets)}"
          + (f"  (! {missing} with NO raw pred -> خطأ)" if missing else ""))

    ground = {}          # sid -> {"texts":[...], "detail":[...]}
    cross_target = set()  # (rid,aid) whose grounded anchor is a DIFFERENT corpus than the span
    n_ok = n_null = n_declined = n_grounded_empty = 0
    for sid, rid, aid, ref, span, claimed, seg_type in targets:
        pred = raw.get(sid)
        texts, detail = ground_span(pred, ref, claimed, alef, snapc)
        ground[sid] = {"texts": texts, "detail": detail}
        if texts and detail and _source_corpus(detail[0].get("sources")) != ref:
            cross_target.add((rid, aid))
        if texts:
            n_ok += 1
        else:
            n_null += 1
            if not pred or pred.get("verdict") != "ok":
                n_declined += 1
            else:
                n_grounded_empty += 1
    print(f"       grounded={n_ok}  null={n_null} (declined {n_declined} + grounded-empty {n_grounded_empty})")

    # ---- raw_model_outputs / snapped_model_outputs (per incorrect span) ----------
    raw_out = os.path.join(out_dir, "raw_model_outputs", f"{name}_raw.jsonl")
    snap_out = os.path.join(out_dir, "snapped_model_outputs", f"{name}_snapped.jsonl")
    os.makedirs(os.path.dirname(raw_out), exist_ok=True)
    os.makedirs(os.path.dirname(snap_out), exist_ok=True)
    with open(raw_out, "w", encoding="utf-8") as fr, open(snap_out, "w", encoding="utf-8") as fs:
        for sid, rid, aid, ref, span, claimed, seg_type in targets:
            pred = raw.get(sid) or {}
            fr.write(json.dumps({
                "id": sid, "response_id": rid, "annotation_id": aid, "ref": ref, "span": span,
                "claimed_source": claimed, "verdict": pred.get("verdict"),
                "rationale": pred.get("rationale"), "candidates": pred.get("candidates") or [],
            }, ensure_ascii=False) + "\n")
            g = ground[sid]
            fs.write(json.dumps({
                "id": sid, "response_id": rid, "annotation_id": aid, "ref": ref, "span": span,
                "corrections": (g["texts"] or None),        # <-- NULL when nothing grounds
                "grounding": g["detail"],
            }, ensure_ascii=False) + "\n")

    # ---- <name>.jsonl (release + correction) + correction.tsv ------------------
    by_target = {(rid, aid): (ground[sid]["texts"] if ground[sid]["texts"] else KHATA)
                 for sid, rid, aid, *_ in targets}
    dst = os.path.join(out_dir, f"{name}.jsonl")
    corr_rows = []
    n_set = 0
    with open(dst, "w", encoding="utf-8") as fout:
        for r in responses:
            rid = r["id"]
            for ann in r.get("annotations", []):
                aid = ann["annotation_id"]; cs = content_seg(ann); key = (rid, aid)
                if key in by_target:                        # incorrect correction target
                    corr = by_target[key]
                    ann["correction"] = corr                # plug into the response (in memory)
                    n_set += 1
                    corr_rows.append([rid, aid, cs["type"] if cs else "",
                                      tsv_clean(KHATA if corr == KHATA else CORR_SEP.join(corr))])
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")

    write_tsv(os.path.join(out_dir, f"{name}_correction.tsv"),
              ["Response_ID", "Annotation_ID", "Segment_Type", "Correction"], corr_rows)

    # ---- relevance.tsv (Task 4): shared generator over the plugged responses -----
    # (correct spans only from fully-correct answers; corrected spans minus cross-corpus)
    snap_by_key = {(rid, aid): {"grounding": ground[sid]["detail"]} for sid, rid, aid, *_ in targets}
    rel_rows, n_correct, n_corrected = task4_relevance_rows(responses, snap_by_key)
    write_tsv(os.path.join(out_dir, f"{name}_relevance.tsv"), REL_HEADER, rel_rows)
    print(f"       wrote {name}.jsonl (set {n_set}) | correction.tsv ({len(corr_rows)}) | "
          f"relevance.tsv ({len(rel_rows)} = {n_correct} correct-from-clean-answers + {n_corrected} corrected"
          f"; {len(cross_target)} cross-corpus corrections excluded)")
    return {"split": name, "targets": len(targets), "grounded": n_ok, "null": n_null,
            "declined": n_declined, "grounded_empty": n_grounded_empty, "jsonl_set": n_set,
            "correction_rows": len(corr_rows), "relevance_rows": len(rel_rows),
            "relevance_correct": n_correct, "relevance_corrected": n_corrected}

    return {"split": split, "targets": len(ids), "grounded": n_ok, "null": n_null,
            "declined": n_declined, "grounded_empty": n_grounded_empty,
            "jsonl_set": n_set, "correction_rows": len(corr_rows),
            "relevance_rows": len(rel_rows), "relevance_correct": n_correct,
            "relevance_corrected": n_corrected}


README = """# IslamicEval 2026 — Subtask 3 (Correction) + Relevance handoff

Generated by `scripts/build_release_deliverables.py`. Splits: {splits}.
Test is intentionally excluded (not final).

## Grounding config (LOCKED)
snap -> dedup -> expand, hadith anchor by content-token Jaccard (frame-stripped):
- SNAP_METRIC=jaccard, SNAP_JAC_TH=0.40  (anchor = highest-Jaccard corpus matn)
- EXPAND_METRIC=jaccard, EXPAND_JAC_TH=0.35
- EXPAND_SKIP_GE=15: per correction — if a SINGLE snapped correction expands to >=15 corpus
  variants (a generic matn matching a large family), its expansion is skipped and only that
  snapped anchor is kept. (A span with several corrections may still list >15 total.)
- Quran corrections resolve to the FULL Uthmani ayah(s); Hadith to canonical matn(s).
- Corrections are sorted SNAPPED-ANCHOR-FIRST; exact-duplicate matns across books collapse.

## Files
- `<split>.jsonl` — the RELEASE jsonl with each incorrect annotation's `correction`
  filled: a LIST of canonical corrections (snapped anchor first) when grounded, else
  the string "خطأ" (no plausible grounded correction).
- `<split>_correction.tsv` — Response_ID, Annotation_ID, Segment_Type, Correction.
  Multiple corrections are joined by " ||| "; ungrounded -> "خطأ".
- `<split>_relevance.tsv` — one row per CORRECT or CORRECTED citation span, with the
  question, the generated answer, the span TYPE {quran|hadith}, the highest-probability
  (snapped) text, an Origin flag {correct|corrected}, and an EMPTY Relevance column to
  be filled by the relevance annotators.
- `raw_model_outputs/<split>_raw.jsonl` — the raw GPT-5.4 one-shot output per incorrect
  span (verdict / rationale / candidates), before grounding.
- `snapped_model_outputs/<split>_snapped.jsonl` — the grounded result per incorrect span:
  `corrections` is the grounded list, or **null** when nothing grounded (the participant
  files above map that null to "خطأ").
"""


def main():
    ap = argparse.ArgumentParser()
    corr = os.path.join(HERE, "..", "outputs", "correction")
    ap.add_argument("--splits", nargs="+", default=["train", "dev"], choices=["train", "dev", "test"])
    ap.add_argument("--raw-tmpl", default=os.path.join(corr, "pipeline_oneshot_release_{split}.json"))
    ap.add_argument("--release-dir", default=os.path.join(HERE, "..", "data", "islamiceval2026_RELEASE"))
    ap.add_argument("--out-dir", default=os.path.join(HERE, "..", "data", "islamiceval2026_RELEASE_correction"))
    # custom single-job mode (e.g. the merged train_corrected): overrides --splits when set
    ap.add_argument("--name", help="output prefix for a custom job (e.g. train_corrected)")
    ap.add_argument("--release-file", help="explicit release jsonl to plug (custom job)")
    ap.add_argument("--raw", nargs="+", help="one or more raw pipeline jsons, MERGED (custom job)")
    ap.add_argument("--id-split", help="stable-id namespace for lookup (default = --name)")
    args = ap.parse_args()

    cfg = set_locked_config()
    print("LOCKED grounding config:", json.dumps(cfg, ensure_ascii=False))
    print("loading corpora (alef + snap)...", flush=True)
    alef = G.load_corpus()
    snapc = G.load_snap_corpus()
    os.makedirs(args.out_dir, exist_ok=True)

    stats = {"config": cfg, "splits": {}}
    if args.name and args.release_file and args.raw:
        # custom job: plug the given release file with the merged raw preds
        stats["splits"][args.name] = process_split(
            args.name, args.release_file, args.raw, args.out_dir, alef, snapc,
            id_split=args.id_split or "train")
        done = [args.name]
    else:
        for split in args.splits:
            release_file = os.path.join(args.release_dir, f"{split}.jsonl")
            raw_path = args.raw_tmpl.format(split=split)
            if not os.path.exists(raw_path):
                print(f"  ! missing raw preds for {split}: {raw_path}"); continue
            stats["splits"][split] = process_split(split, release_file, [raw_path],
                                                   args.out_dir, alef, snapc)
        done = args.splits

    open(os.path.join(args.out_dir, "README.md"), "w", encoding="utf-8").write(
        README.replace("{splits}", ", ".join(done)))
    json.dump(stats, open(os.path.join(args.out_dir, "build_stats.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\nDONE -> {os.path.normpath(args.out_dir)}")
    print(json.dumps(stats["splits"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
