#!/usr/bin/env python3
"""
correct_spans.py
================
Correction stage for IslamicEval. Reads the scraped citation-span CSV
(produced by extract_spans_for_correction.py — columns: id, model, span, ref,
correctness) and, for every INCORRECT Quran / Hadith span, asks a frontier model
to retrieve the top-K canonical references (ayahs / hadiths) closest to the
(wrong) span — i.e. the references the answering model most likely intended —
or to return "خطأ" if nothing canonical is plausibly close.

Reuses the shared provider plumbing in annotate_spans.py (call_api,
parse_json_response, MODEL_REGISTRY). Output is written incrementally and is
resumable, same snapshot shape as annotate_spans.

Usage:
    python correct_spans.py --model gemini-3-flash-preview --input ../outputs/correction/citation_spans.csv
    python correct_spans.py --model gpt-5.4 --input ../outputs/correction/citation_spans.csv --top-k 5

The prompt (SYSTEM_CORRECT / build_user_prompt) is a first cut — tune later.
"""
import os
import csv
import sys
import json
import argparse
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Dict, List, Optional

from annotate_spans import (
    MODEL_REGISTRY, call_api, parse_json_response,
)

# ─────────────────────────────────────────────────────────────────────────────
# TASK CONFIG — edit to tune the correction prompt / schema.
# ─────────────────────────────────────────────────────────────────────────────
TASK_NAME = "reference_correction"

SYSTEM_CORRECT = """\
You are a hafiz (حافظ) and Islamic scholar with complete memorisation of:
  • The Holy Quran in the Uthmani script (الرسم العثماني) with full tashkeel.
  • The six canonical Hadith collections (الكتب الستة): Sahih al-Bukhari,
    Sahih Muslim, Sunan Abu Dawud, Jami' al-Tirmidhi, Sunan al-Nasa'i,
    Sunan Ibn Majah.

You are given ONE Arabic text span that an AI model presented as a Quranic ayah
or a Prophetic hadith, but which is INCORRECT — it is misquoted, paraphrased,
garbled, or fabricated.

Your task: retrieve the TOP {k} ACTUAL references from the canonical sources that
are CLOSEST (in wording and meaning) to the given span — the references the model
most likely intended — ranked from closest to least close.

  • For a Quran span: return the exact Uthmani ayah text (with tashkeel) and its
    source as "سورة <name> <ayah-number>".
  • For a Hadith span: return the authentic matn and its source as
    "<collection> <number/reference>".

If the span is gibberish, or so fabricated that NO canonical ayah/hadith is
plausibly close, set "verdict" to "خطأ" and return an empty "candidates" list.

Return ONLY valid JSON in this exact shape:
{"verdict": "ok" | "خطأ", "candidates": [{"text": "...", "source": "..."}, ...]}
At most {k} candidates, ranked closest first. When verdict is "خطأ", candidates
must be []. No commentary outside the JSON.
"""

SCHEMA_CORRECT = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["ok", "خطأ"]},
        "candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text":   {"type": "string"},
                    "source": {"type": "string"},
                },
                "required": ["text", "source"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["verdict", "candidates"],
    "additionalProperties": False,
}

REF_LABEL = {"quran": "Quranic ayah (آية قرآنية)", "hadith": "Prophetic hadith (حديث نبوي)"}
REF_PLURAL = {"quran": "ayahs", "hadith": "hadiths"}


def system_prompt(k: int) -> str:
    return SYSTEM_CORRECT.replace("{k}", str(k))


def build_user_prompt(span: str, ref: str, k: int) -> str:
    return (
        f"The following span was presented as a {REF_LABEL.get(ref, ref)} but is "
        f"INCORRECT. Retrieve the top {k} closest canonical {REF_PLURAL.get(ref, ref)}, "
        f"or return \"خطأ\" if none is plausibly close.\n\n"
        f"Span:\n<<<\n{span}\n>>>"
    )


# ─────────────────────────────────────────────────────────────────────────────
# INPUT
# ─────────────────────────────────────────────────────────────────────────────
def uid_for(row_idx: int, r: Dict) -> str:
    """Stable per-span id for resume: row index (CSV order) + a short content hash."""
    h = hashlib.sha1(f"{r['id']}|{r['model']}|{r['ref']}|{r['span']}".encode("utf-8")).hexdigest()[:8]
    return f"{row_idx:05d}__{r['id']}__{r['model']}__{h}"


def load_spans(path: str, incorrect_only: bool = True) -> List[Dict]:
    with open(path, encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    out = []
    for i, r in enumerate(rows):
        if r["ref"] not in REF_LABEL:                 # quran / hadith only
            continue
        if incorrect_only and r.get("correctness") != "incorrect":
            continue
        out.append({
            "uid":   uid_for(i, r),
            "id":    r["id"],
            "model": r["model"],
            "span":  r["span"],
            "ref":   r["ref"],
            "correctness": r.get("correctness", ""),
        })
    return out


# ─────────────────────────────────────────────────────────────────────────────
# RESUME / SAVE  (same snapshot shape as annotate_spans)
# ─────────────────────────────────────────────────────────────────────────────
def load_resume_state(output_path: str):
    if not os.path.exists(output_path):
        return [], set()
    try:
        with open(output_path, encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception as e:
        print(f"  [WARN] could not read {output_path} for resume: {e}")
        return [], set()
    per_sample = data.get("per_sample", [])
    if data.get("status") == "complete":
        print(f"  [SKIP] {output_path} already complete -- delete it to re-run.")
        return per_sample, None
    if per_sample:
        print(f"  [RESUME] {len(per_sample)} already done -- skipping those.")
    return per_sample, {s["uid"] for s in per_sample}


def save_incremental(output_path: str, model_name: str, k: int,
                     per_sample: List[Dict], n_total: int) -> None:
    n_done = len(per_sample)
    snapshot = {
        "task":        TASK_NAME,
        "model":       model_name,
        "top_k":       k,
        "timestamp":   datetime.now(timezone.utc).isoformat(),
        "status":      "complete" if n_done >= n_total else "in_progress",
        "n_total":     n_total,
        "n_completed": n_done,
        "n_failed":    sum(1 for s in per_sample if s.get("error")),
        "per_sample":  per_sample,
    }
    d = os.path.dirname(output_path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(snapshot, fh, ensure_ascii=False, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────────────────────────
def correct_one(model_name: str, sysmsg: str, k: int, sp: Dict) -> Dict:
    """Worker: correct a single span. Returns the result record (thread-safe — no shared state)."""
    raw, usage = call_api(model_name, sysmsg,
                          build_user_prompt(sp["span"], sp["ref"], k),
                          schema=SCHEMA_CORRECT)
    rec = {
        "uid":          sp["uid"],
        "id":           sp["id"],
        "answer_model": sp["model"],
        "ref":          sp["ref"],
        "span":         sp["span"],
    }
    if raw is None:
        rec.update({"prediction": None, "raw_response": None, "usage": None,
                    "error": "API call failed"})
    else:
        rec.update({"prediction": parse_json_response(raw), "raw_response": raw, "usage": usage})
    return rec


def run(model_name: str, input_path: str, output_path: str, k: int,
        incorrect_only: bool = True, max_samples: Optional[int] = None,
        workers: int = 8) -> None:
    spans = load_spans(input_path, incorrect_only=incorrect_only)
    if max_samples:
        spans = spans[:max_samples]
    n_total = len(spans)

    per_sample, completed = load_resume_state(output_path)
    if completed is None:
        return
    todo = [s for s in spans if s["uid"] not in completed]
    print(f"\n=== {TASK_NAME} | {model_name} | top-{k} | {len(todo)} of {n_total} spans "
          f"| {workers} workers ===")

    sysmsg = system_prompt(k)
    lock = threading.Lock()
    done = 0
    # Concurrent calls — a flaky/slow call (14s back-off) no longer stalls the rest.
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(correct_one, model_name, sysmsg, k, sp): sp for sp in todo}
        for fut in as_completed(futures):
            rec = fut.result()
            with lock:
                per_sample.append(rec)
                done += 1
                if rec.get("error"):
                    tag = "FAILED"
                else:
                    p = rec.get("prediction") or {}
                    tag = f"{p.get('verdict','?')}, {len(p.get('candidates',[]) or [])} cand"
                print(f"  [{done}/{len(todo)}] {rec['id']}/{rec['answer_model']}/{rec['ref']} -> {tag}")
                # checkpoint every 10 (and implicitly at the end via final save below)
                if done % 10 == 0:
                    save_incremental(output_path, model_name, k, per_sample, n_total)

    save_incremental(output_path, model_name, k, per_sample, n_total)
    print(f"  -> wrote {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Retrieve correct references for incorrect citation spans")
    ap.add_argument("--model", required=True,
                    help="model, one of: " + ", ".join(MODEL_REGISTRY.keys()))
    ap.add_argument("--input", default="../outputs/correction/citation_spans.csv",
                    help="CSV from extract_spans_for_correction.py (id,model,span,ref,correctness)")
    ap.add_argument("--output", default=None,
                    help="output JSON (default: ../outputs/correction/corrections_<model>.json)")
    ap.add_argument("--top-k", type=int, default=5, help="number of candidate references to retrieve")
    ap.add_argument("--include-correct", action="store_true",
                    help="also process spans labelled correct (default: incorrect only)")
    ap.add_argument("--max-samples", type=int, default=None, help="limit to first N spans (testing)")
    ap.add_argument("--workers", type=int, default=8, help="concurrent API calls (default 8)")
    args = ap.parse_args()

    if args.model not in MODEL_REGISTRY:
        sys.exit(f"Unknown model '{args.model}'. Choices: {list(MODEL_REGISTRY.keys())}")
    _, _, key_env = MODEL_REGISTRY[args.model]
    if not os.environ.get(key_env):
        sys.exit(f"{key_env} is not set in the environment (.env) for model {args.model}")

    out = args.output or f"../outputs/correction/corrections_{args.model}.json"
    run(args.model, args.input, out, args.top_k,
        incorrect_only=not args.include_correct, max_samples=args.max_samples,
        workers=args.workers)


if __name__ == "__main__":
    main()
