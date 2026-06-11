#!/usr/bin/env python3
"""
annotate_typology.py
====================
Stage-3 span annotation for IslamicEval — ONE-PASS extract + classify.

For every model answer, an annotator LLM (default gpt-5.5) is asked to return
EVERY claimed/intended citation span with, per span:
    span_start   0-based char index into the answer (inclusive)
    span_end     0-based char index into the answer (EXCLUSIVE, i.e. [start, end))
    type         Quran | Hadith_Matn | Hadith_Isnad | Reference
    label        Correct | Incorrect   (for Quran & Hadith_Matn only; NA otherwise)
    start_words  the first few words of the span, verbatim
    end_words    the last few words of the span, verbatim

start_words / end_words exist so a human reviewer can locate and verify a span
without the full text being duplicated; the script also assembles a convenience
field  span_text = "start_words .... end_words".

Output is STRUCTURED JSON (OpenAI strict json_schema), written incrementally and
resumable, one file per answer-model:
    outputs/annotations/<answer_model>__<annotator>_typology.json

NOTE: the SYSTEM_PROMPT below is a DRAFT — tune it freely. The schema and the
plumbing are the parts that are "ready"; the wording is meant to be iterated on.

Usage:
    # all model files in a directory (the 11 models), default annotator gpt-5.5
    python annotate_typology.py --input-dir ../outputs/answers/short_answers

    # a single file, quick test on 5 answers, 2 workers
    python annotate_typology.py --input ../outputs/answers/samples/allam-7b_sample.json \
                                --limit 5 --workers 2

    # pick the annotator / restrict which answer-model files to run
    python annotate_typology.py --input-dir ../outputs/answers/short_answers \
                                --annotator gpt-5.4 --models allam-7b qwen3-8b
"""
import os, re, sys, json, glob, time, argparse
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional
from tqdm.auto import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from annotate_spans import call_api, MODEL_REGISTRY, parse_json_response  # also runs load_dotenv()

DEFAULT_ANNOTATOR = "gpt-5.4-mini"

# Approx USD per 1M tokens (input, output) — PLACEHOLDERS; edit to current pricing.
PRICES = {
    "gpt-5.5": (5.0, 30.0),
    "gpt-5.4": (2.5, 15.0),
    "gpt-5":   (1.25, 10.0),
    "gpt-5.4-mini": (0.75, 4.5),
    "gemini-3-flash-preview": (1.5, 9.0),
    "gemini-3.1-pro-preview": (2.0, 12.0),
}

# ─────────────────────────────────────────────────────────────────────────────
# TASK CONFIG  —  SYSTEM_PROMPT is a DRAFT, tune freely. SCHEMA is enforced.
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are an expert annotator for an Arabic Islamic citation benchmark. You are given
the text of a single AI-generated Arabic answer. Identify EVERY span the writer
presents (claims or intends) as a Qur'anic verse, a Prophetic hadith, a hadith
chain of narration, or a source reference — EVEN IF it is fabricated, misquoted,
paraphrased, or misattributed. Do not skip doubtful citations and do not verify by
rewriting the text.

For each span return these fields:

1. type — exactly one of:
   - "Quran"        : text the writer presents as a Qur'anic ayah (the verse text).
   - "Hadith_Matn"  : text the writer presents as the body/matn of a hadith.
   - "Hadith_Isnad" : a chain of narrators (إسناد), e.g. "حدثنا ... عن ... عن ...".
   - "Reference"    : a source attribution, e.g. surah name + ayah number
                      (سورة البقرة: 6 / البقرة 2:185) or hadith source
                      (رواه البخاري / صحيح مسلم / الترمذي / متفق عليه).

2. label — correctness, exactly one of "Correct", "Incorrect", or "NA". Be STRICT.

   - type "Quran"  —  VERBATIM Uthmani match, INCLUDING the meaningful tashkeel (harakat:
     fatha, kasra, damma, shadda, sukun, tanwin that reflect a real vowel), EXCEPT for the
     purely notational variants listed below, which you MUST ignore (they change neither the
     word nor its pronunciation):
         1. tatweel / kashida (ـ)  — e.g. الرحمـٰن
         2. dagger / superscript alif (ٰ)  — هَٰذَا / ذَٰلِكَ / ٱلرَّحْمَٰن  ≡  هذا / ذلك / الرحمن
         3. alif wasla (ٱ) vs plain alif (ا)  — ٱللَّه / ٱهْدِنَا  ≡  الله / اهدنا
         4. madd sign (ٓ)  — جَآءَ / ٱلسَّمَآء  ≡  جاء / السماء
         5. sukun glyph + idgham notation  — regular sukun (ْ) vs the small-circle (ۡ), and
            assimilation shown as a shadda on the next letter or as no mark at all
         6. waqf / tajwid recitation marks placed between words (ۖ ۗ ۘ ۙ ۚ ۛ ۜ, iqlab ۢ)
         7. tanwin glyph form  — stacked/positional Uthmani tanwin vs sequential ً ٌ ٍ
       Correct   = matches a real ayah of the Uthmani Mushaf exactly once items 1–7 are
                   normalized away.
       Incorrect = ANY other difference: a changed / missing / added word, wrong word order,
                   a wrong or missing meaningful harakah, a final ى vs ي swap, a hamza-seat
                   (أ/إ/ا) or ة-vs-ه difference, or text not found in the Qur'an. If unsure
                   the diacritics match (beyond items 1–7), mark Incorrect.

   - type "Hadith_Matn"  —  SAME WORDING, diacritics ignored:
       Correct   = the matn has the same wording as a hadith in the six canonical books —
                   the same words in the same order. Diacritics (tashkeel) do NOT matter and
                   may differ freely.
       Incorrect = paraphrased, reworded, words added / dropped / changed / reordered,
                   fabricated, or not found in the canonical books.

   - type "Reference"  —  ACCURATE attribution of the citation it points to:
       Correct   = the named source is the TRUE source of the adjacent citation: the surah
                   name AND ayah number are where that verse actually occurs; or the named
                   collection really contains that hadith (e.g. if it says البخاري the hadith
                   is indeed in Sahih al-Bukhari; "متفق عليه" → in both Bukhari and Muslim).
       Incorrect = misattributed — wrong surah or wrong ayah number for the quoted verse, or
                   the hadith is not actually in the named collection.

   - "NA"          —  use ONLY for type "Hadith_Isnad" (an isnad carries no correctness verdict).

3. span_start, span_end — 0-based character indices into the answer text exactly as
   provided. Use half-open [span_start, span_end): answer[span_start:span_end] should
   be the span. Do not normalize, reorder, or rewrite the text when counting.

4. start_words — the FIRST few words (about 3–6) of the span, copied verbatim.
   end_words   — the LAST few words (about 3–6) of the span, copied verbatim.
   These must be exact substrings of the answer so a human can locate the span.

Boundary rules:
- Select the minimal contiguous citation text. Exclude surrounding quotes, brackets,
  ellipses, decorative marks, and verse numbers from Quran/Hadith_Matn spans (those
  belong to a separate "Reference" span).
- Extract every distinct occurrence by position; do not collapse repeats.
- If a verse is embedded inside a hadith narration, extract the verse as "Quran".

If the answer contains no citations, return {"spans": []}.

Return ONLY JSON of the form:
{"spans": [{"type": "...", "label": "...", "span_start": 0, "span_end": 0,
            "start_words": "...", "end_words": "..."}, ...]}
"""

SCHEMA = {
    "type": "object",
    "properties": {
        "spans": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type":        {"type": "string",
                                    "enum": ["Quran", "Hadith_Matn", "Hadith_Isnad", "Reference"]},
                    "label":       {"type": "string", "enum": ["Correct", "Incorrect", "NA"]},
                    "span_start":  {"type": "integer"},
                    "span_end":    {"type": "integer"},
                    "start_words": {"type": "string"},
                    "end_words":   {"type": "string"},
                },
                "required": ["type", "label", "span_start", "span_end",
                             "start_words", "end_words"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["spans"],
    "additionalProperties": False,
}


def build_user_prompt(answer_text: str) -> str:
    return f"Answer text to annotate:\n<<<\n{answer_text}\n>>>"


def assemble_span_text(span: Dict) -> str:
    """Build the human-revision snippet 'first words .... last words'."""
    head = str(span.get("start_words", "")).strip()
    tail = str(span.get("end_words", "")).strip()
    if head and tail and head != tail:
        return f"{head} .... {tail}"
    return head or tail


def _occurrences(text: str, sub: str) -> List[int]:
    """All start indices where `sub` occurs in `text`."""
    out, pos = [], 0
    if not sub:
        return out
    while True:
        i = text.find(sub, pos)
        if i == -1:
            break
        out.append(i)
        pos = i + 1
    return out


def snap_span_indices(answer: str, sp: Dict) -> None:
    """Re-derive span_start/span_end from the verbatim start_words/end_words.

    LLMs count Arabic character indices unreliably (diacritics/tokenization), but
    the words they copy are verbatim-correct. So we trust the words and SEARCH for
    them, using the model's reported indices only to disambiguate repeats.

    Mutates `sp`:
      - keeps the model's original indices as model_span_start / model_span_end
      - sets span_start/span_end to the located positions when found
      - words_verified : True if both anchors were located verbatim
      - index_source   : "snapped" (moved), "model_ok" (already matched), or "model" (not found)
    """
    head = str(sp.get("start_words", "")).strip()
    tail = str(sp.get("end_words", "")).strip()
    hs, he = sp.get("span_start"), sp.get("span_end")
    sp["model_span_start"], sp["model_span_end"] = hs, he

    starts = _occurrences(answer, head)
    if not head or not starts:
        sp["words_verified"], sp["index_source"] = False, "model"
        return

    s = min(starts, key=lambda i: abs(i - hs)) if isinstance(hs, int) else starts[0]

    if not tail or tail == head:
        e = s + len(head)
    else:
        ends = _occurrences(answer, tail)
        cand = [p for p in ends if p >= s] or ends
        if not cand:
            sp["words_verified"], sp["index_source"] = False, "model"
            return
        p = (min(cand, key=lambda i: abs((i + len(tail)) - he))
             if isinstance(he, int) else cand[0])
        e = p + len(tail)

    if e <= s:
        sp["words_verified"], sp["index_source"] = False, "model"
        return

    sp["span_start"], sp["span_end"] = s, e
    sp["words_verified"] = True
    sp["index_source"] = "model_ok" if (s == hs and e == he) else "snapped"


# ─────────────────────────────────────────────────────────────────────────────
# I/O — load answers, resume, incremental save
# ─────────────────────────────────────────────────────────────────────────────
def load_answers(path: str) -> List[Dict]:
    d = json.load(open(path, encoding="utf-8"))
    if isinstance(d, dict):
        d = d.get("per_sample", d) if "per_sample" in d else [d]
    return d


def token_totals(per_sample: List[Dict], annotator: str):
    """Sum input/output tokens across records and estimate USD cost."""
    tin = sum((s.get("usage") or {}).get("input_tokens") or 0 for s in per_sample)
    tout = sum((s.get("usage") or {}).get("output_tokens") or 0 for s in per_sample)
    pin, pout = PRICES.get(annotator, (None, None))
    cost = round(tin / 1e6 * pin + tout / 1e6 * pout, 4) if pin is not None else None
    return tin, tout, cost


def save_snapshot(out_path: str, annotator: str, answer_model: str,
                  per_sample: List[Dict], n_total: int) -> None:
    n_done = len(per_sample)
    tin, tout, cost = token_totals(per_sample, annotator)
    snap = {
        "task":            "typology_spans",
        "annotator":       annotator,
        "answer_model":    answer_model,
        "timestamp":       datetime.now(timezone.utc).isoformat(),
        "status":          "complete" if n_done >= n_total else "in_progress",
        "n_total":         n_total,
        "n_completed":     n_done,
        "n_failed":        sum(1 for s in per_sample if s.get("error")),
        "input_tokens":    tin,
        "output_tokens":   tout,
        "est_cost_usd":    cost,
        "per_sample":      per_sample,
    }
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    tmp = out_path + ".tmp"
    json.dump(snap, open(tmp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    os.replace(tmp, out_path)


def annotate_one(annotator: str, item: Dict) -> Dict:
    """Annotate a single answer. Returns a per_sample record (never raises)."""
    sid = item.get("id")
    answer = item.get("answer", "") or ""
    t0 = time.time()
    try:
        raw, usage = call_api(annotator, SYSTEM_PROMPT, build_user_prompt(answer), schema=SCHEMA)
        if raw is None:
            raise RuntimeError("API call failed (returned None)")
        parsed = parse_json_response(raw)
        spans = (parsed or {}).get("spans", []) if parsed else []
        for sp in spans:
            snap_span_indices(answer, sp)          # authoritative indices from verbatim words
            sp["span_text"] = assemble_span_text(sp)
        n_unverified = sum(1 for sp in spans if not sp.get("words_verified"))
        return {"sample_id": sid, "prompt": item.get("prompt", ""), "answer": answer,
                "answer_model": item.get("model", ""), "spans": spans,
                "n_unverified": n_unverified,
                "usage": usage, "latency_s": round(time.time() - t0, 2)}
    except Exception as e:
        return {"sample_id": sid, "prompt": item.get("prompt", ""), "answer": answer,
                "answer_model": item.get("model", ""), "spans": None,
                "usage": None, "error": f"{type(e).__name__}: {str(e)[:160]}",
                "latency_s": round(time.time() - t0, 2)}


def run_file(annotator: str, in_path: str, out_dir: str, workers: int,
             save_every: int, limit: Optional[int]) -> Dict:
    answers = load_answers(in_path)
    if limit:
        answers = answers[:limit]
    answer_model = answers[0].get("model") or os.path.splitext(os.path.basename(in_path))[0]
    out_path = os.path.join(out_dir, f"{answer_model}__{annotator}_typology.json")

    # resume: keep records that already succeeded
    done: Dict[str, Dict] = {}
    if os.path.exists(out_path):
        prev = json.load(open(out_path, encoding="utf-8"))
        if prev.get("status") == "complete":
            print(f"  [SKIP] {out_path} complete — delete to re-run.")
            return {"answer_model": answer_model, "out": out_path, "skipped": True}
        for r in prev.get("per_sample", []):
            if r.get("spans") is not None and not r.get("error"):
                done[r["sample_id"]] = r

    todo = [a for a in answers if a.get("id") not in done]
    n_total = len(answers)
    print(f"\n=== {answer_model}: {len(todo)}/{n_total} to annotate (annotator={annotator}) ===")

    results = dict(done)
    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(annotate_one, annotator, a): a for a in todo}
        bar = tqdm(as_completed(futs), total=len(futs), desc=answer_model)
        for fut in bar:
            rec = fut.result()
            results[rec["sample_id"]] = rec
            completed += 1
            n_sp = 0 if rec.get("error") else len(rec.get("spans") or [])
            bar.set_postfix_str(f"{rec['sample_id']} {'ERR' if rec.get('error') else f'{n_sp} spans'}")
            if completed % save_every == 0:
                ordered = [results[a["id"]] for a in answers if a["id"] in results]
                save_snapshot(out_path, annotator, answer_model, ordered, n_total)
            if rec.get("error"):
                bar.write(f"  [WARN] {rec['sample_id']}: {rec['error']}")

    ordered = [results[a["id"]] for a in answers if a["id"] in results]
    save_snapshot(out_path, annotator, answer_model, ordered, n_total)
    n_fail = sum(1 for r in ordered if r.get("error"))
    n_spans = sum(len(r.get("spans") or []) for r in ordered)
    n_snapped = sum(1 for r in ordered for sp in (r.get("spans") or [])
                    if sp.get("index_source") == "snapped")
    n_unverified = sum(1 for r in ordered for sp in (r.get("spans") or [])
                       if not sp.get("words_verified"))
    tin, tout, cost = token_totals(ordered, annotator)
    cost_str = f" | ~${cost:.4f}" if cost is not None else ""
    print(f"  -> {out_path}  ({len(ordered)} answers, {n_spans} spans, "
          f"{n_snapped} snapped, {n_unverified} unverified, {n_fail} failed | "
          f"in {tin:,} / out {tout:,} tok{cost_str})")
    return {"answer_model": answer_model, "out": out_path,
            "answers": len(ordered), "spans": n_spans, "snapped": n_snapped,
            "unverified": n_unverified, "failed": n_fail,
            "input_tokens": tin, "output_tokens": tout, "est_cost_usd": cost}


def main():
    ap = argparse.ArgumentParser(description="Stage-3 one-pass typology span annotation")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", help="a single answers JSON: [{id,prompt,answer,model}]")
    src.add_argument("--input-dir", help="directory of model answer JSON files (the 11 models)")
    ap.add_argument("--outdir", default="../outputs/annotations")
    ap.add_argument("--annotator", default=DEFAULT_ANNOTATOR)
    ap.add_argument("--models", nargs="+", default=None,
                    help="when using --input-dir, restrict to these answer-model file stems")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--limit", type=int, default=None, help="first N answers per file (testing)")
    args = ap.parse_args()

    if args.annotator not in MODEL_REGISTRY:
        sys.exit(f"Unknown annotator '{args.annotator}'. Known: {list(MODEL_REGISTRY)}")

    if args.input:
        files = [args.input]
    else:
        files = sorted(glob.glob(os.path.join(args.input_dir, "*.json")))
        if args.models:
            keep = set(args.models)
            files = [f for f in files
                     if os.path.splitext(os.path.basename(f))[0] in keep]
    if not files:
        sys.exit("No input files matched.")

    os.makedirs(args.outdir, exist_ok=True)
    print(f"annotator: {args.annotator} | files: {len(files)} | workers: {args.workers}")
    summary = [run_file(args.annotator, f, args.outdir, args.workers,
                        args.save_every, args.limit) for f in files]

    print("\n=== SUMMARY ===")
    grand = 0.0
    for s in summary:
        if s.get("skipped"):
            print(f"  {s['answer_model']:24} SKIPPED (complete)")
        else:
            c = s.get("est_cost_usd")
            grand += c or 0.0
            cost_str = f"${c:.4f}" if c is not None else "n/a"
            print(f"  {s['answer_model']:24} answers={s['answers']:>5} "
                  f"spans={s['spans']:>6} failed={s['failed']:>3} "
                  f"in={s['input_tokens']:>8,} out={s['output_tokens']:>8,} {cost_str:>10}")
    print(f"  {'TOTAL est. cost':24} ~${grand:.4f}   "
          f"(PRICES are placeholders — edit for real $)")


if __name__ == "__main__":
    main()
