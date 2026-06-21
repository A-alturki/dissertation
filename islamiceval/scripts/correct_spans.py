#!/usr/bin/env python3
"""
correct_spans.py
================
Correction stage for IslamicEval. For every INCORRECT Quran / Hadith span (from
extract_spans_for_correction.py → citation_spans.csv: id, model, span, ref,
correctness) it recovers the canonical text the answering model was trying to
quote, or declares the span baseless ("خطأ").

TWO METHODS (select with --method):

  generate   The model is given ONLY the span and must, from memory, output the
             top-K closest canonical candidates or "خطأ" (SYSTEM = GENERATION_PROMPT).

  judge      A deterministic corpus search (corpus_search.search) first finds the
             top-K canonical texts sharing the longest contiguous word-run with the
             span; the model then PICKS the genuine distortion among them or rules
             "خطأ" (SYSTEM = JUDGE_PROMPT). If the search finds nothing, we emit
             "خطأ" WITHOUT an LLM call (saves cost; search gate did the rejection).

Reuses the shared provider plumbing in annotate_spans.py (call_api,
parse_json_response, MODEL_REGISTRY). Output is written incrementally, is
resumable, and (unlike before) RETRIES records that previously errored.

Usage:
    python correct_spans.py --method generate --model gemini-3-flash-preview \
        --input ../outputs/correction/sample50_spans.csv
    python correct_spans.py --method judge --model gpt-5.4 \
        --input ../outputs/correction/sample50_spans.csv --top-k 5
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

from annotate_spans import MODEL_REGISTRY, call_api, parse_json_response

TASK_NAME = "reference_correction"

# (input, output) USD per 1M tokens — placeholders, mirror annotate_typology.PRICES.
PRICES = {
    "gpt-5.5": (5.0, 30.0),
    "gpt-5.4": (2.5, 15.0),
    "gpt-5":   (1.25, 10.0),
    "gpt-5.4-mini": (0.75, 4.5),
    "gemini-3-flash-preview": (1.5, 9.0),
    "gemini-3.5-flash": (1.5, 9.0),         # placeholder — update to real rate
    "gemini-3.1-pro-preview": (2.0, 12.0),
}


def token_totals(per_sample, model_name):
    """Sum input/output tokens (records with usage) and estimate USD cost."""
    tin = sum((s.get("usage") or {}).get("input_tokens", 0) or 0 for s in per_sample)
    tout = sum((s.get("usage") or {}).get("output_tokens", 0) or 0 for s in per_sample)
    pin, pout = PRICES.get(model_name, (None, None))
    cost = round(tin / 1e6 * pin + tout / 1e6 * pout, 4) if pin is not None else None
    return tin, tout, cost

# ─────────────────────────────────────────────────────────────────────────────
# METHOD 1 — GENERATION  (model retrieves from memory)
# ─────────────────────────────────────────────────────────────────────────────
GENERATION_PROMPT = """\
You are a hafiz (حافظ) and Islamic scholar with complete memorization of:
  • The Holy Quran in Uthmani script (الرسم العثماني) with full tashkeel.
  • The six canonical Hadith collections (الكتب الستة): Sahih al-Bukhari,
    Sahih Muslim, Sunan Abu Dawud, Jami' al-Tirmidhi, Sunan al-Nasa'i,
    Sunan Ibn Majah.

You are given ONE Arabic text span that an AI model presented as a Quranic ayah
or a Prophetic hadith, but which is INCORRECT — misquoted, misdiacritized,
paraphrased, garbled, or entirely fabricated. Recover the actual canonical text
it was attempting to quote, or declare it baseless (خطأ).

────────────────────────────────────────────────────────────────────────
THE CORE RULE — WHAT COUNTS AS A CORRECTION
────────────────────────────────────────────────────────────────────────
A candidate is a valid correction ONLY IF the span is a recoverable distortion
of that specific candidate's wording. After mentally stripping tashkeel and
normalizing letters (أ إ آ ا → ا ; ى → ي ; ة → ه ; ؤ ئ → hamza), a MEANINGFUL
CONTIGUOUS SEQUENCE OF WORDS from the span must appear in the candidate — a
recognizable phrase, not isolated words.

NOT a correction (these are "خطأ"):
  • Span shares only the TOPIC/MEANING with real texts (theft, marriage, charity)
    but no actual phrase matches.
  • Span shares only ISOLATED words scattered across different verses/hadiths —
    one word from text A, one from text B. Scattered single words ≠ correction.
  • Span is fluent and Islamic-sounding but reconstructs no real text.
Be STRICT. A thematically related text the span does not actually quote is a
SERIOUS ERROR. A weak guess is worse than "خطأ".

────────────────────────────────────────────────────────────────────────
OUTPUT — valid JSON only, FIELDS IN THIS EXACT ORDER:
{{
  "rationale": "<brief: what real text (if any) the span distorts, the longest
                contiguous phrase that matches, and why ok or خطأ>",
  "verdict": "ok" | "خطأ",
  "verdict_confidence": <0.0-1.0>,
  "candidates": [
    {{"text": "<exact canonical text>",
      "source": "سورة <name> <n>"  OR  "<collection> <n>",
      "matched_run": "<longest contiguous phrase shared with the span>",
      "confidence": <0.0-1.0>}}
  ]
}}

  • Quran "text": full Uthmani ayah(s) with tashkeel. Hadith "text": matn only,
    no isnad, no "رواه...".
  • verdict_confidence = confidence in the verdict you emit:
      if "خطأ"  → how sure the span is baseless (no canonical text fits the rule).
      if "ok"   → how sure the span is correctable to a real text at all.
  • Per-candidate "confidence" reflects contiguous-phrase overlap:
      0.8-1.0  long contiguous run; clearly this exact text, only diacritic/
               spelling/one-inflection differences.
      0.5-0.7  a real multi-word phrase matches but with paraphrase or a fused
               extra clause.
      0.2-0.4  only a short phrase matches; plausible but uncertain.
      below 0.2 → do NOT emit the candidate.
  • When verdict is "خطأ", candidates MUST be [].
  • Rank candidates by confidence, highest first.
  • State "matched_run" before "confidence"; the number must reflect that run.

────────────────────────────────────────────────────────────────────────
EXAMPLES
────────────────────────────────────────────────────────────────────────
# Quran — recoverable (pronoun changed, 3-4 word run matches)
Span: "رَبِّنَا هَبْ لَنَا مِنَ الصَّالِحِينَ"
{{"rationale": "Distorts الصافات 100; contiguous run 'هب . من الصالحين' matches, only the pronoun differs (لنا vs لي).",
  "verdict": "ok", "verdict_confidence": 0.95,
  "candidates": [
    {{"text": "رَبِّ هَبْ لِي مِنَ الصَّالِحِينَ", "source": "سورة الصافات 100",
      "matched_run": "هب من الصالحين", "confidence": 0.9}}
]}}

# Hadith — recoverable (verbatim spine + a fused extra clause)
Span: "أفضل الصدقة سقي الماء والحج المبرور"
{{"rationale": "Verbatim spine 'أفضل الصدقة سقي الماء' matches Ibn Majah 3684; trailing 'والحج المبرور' is a fused second topic, but the spine recovers one specific hadith.",
  "verdict": "ok", "verdict_confidence": 0.85,
  "candidates": [
    {{"text": "أَفْضَلُ الصَّدَقَةِ سَقْيُ الْمَاءِ", "source": "سنن ابن ماجه 3684",
      "matched_run": "أفضل الصدقة سقي الماء", "confidence": 0.8}}
]}}

# Quran — topic + scattered single words only → خطأ
Span: "وَالْحَجُّ أَشَدُّ الْفُرُوجِ"
{{"rationale": "'الحج' occurs in البقرة 197 and 'فروج' in المؤمنون 5, but no contiguous phrase from the span exists in any verse. Topic + isolated words.",
  "verdict": "خطأ", "verdict_confidence": 0.9, "candidates": []}}

# Hadith — topic only → خطأ
Span: "السرقة حرام، وإن كانت لطعام وحاجة"
{{"rationale": "Real hadiths on theft exist but none carries this phrasing; only scattered topic words (سرقة، حرام) overlap. No quotation recovered.",
  "verdict": "خطأ", "verdict_confidence": 0.88, "candidates": []}}

# Garbled / fabricated → خطأ
Span: "وَمَنْ لَا يُحِيطُ بِمَالِهِ إِلَّا بِإِذْنِهِ يُحِيطُ بِمَالِهِ وَسَعَ كُرْسِيُّ السَّلَامُ وَيَحْتَسِبُ بِاللَّهِ..."
{{"rationale": "Incoherent fragments stitched from multiple sources; reconstructs no single real ayah or hadith.",
  "verdict": "خطأ", "verdict_confidence": 0.95, "candidates": []}}
"""

SCHEMA_GENERATE = {
    "type": "object",
    "properties": {
        "rationale":          {"type": "string"},
        "verdict":            {"type": "string", "enum": ["ok", "خطأ"]},
        "verdict_confidence": {"type": "number"},
        "candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text":        {"type": "string"},
                    "source":      {"type": "string"},
                    "matched_run": {"type": "string"},
                    "confidence":  {"type": "number"},
                },
                "required": ["text", "source", "matched_run", "confidence"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["rationale", "verdict", "verdict_confidence", "candidates"],
    "additionalProperties": False,
}

# ─────────────────────────────────────────────────────────────────────────────
# METHOD 2 — JUDGE  (deterministic search proposes; model picks)
# chosen_index uses -1 (not null) for "خطأ": a single integer schema is the only
# nullable encoding that satisfies BOTH OpenAI-strict and Gemini response_schema.
# ─────────────────────────────────────────────────────────────────────────────
JUDGE_PROMPT = """\
You are a hafiz (حافظ) and Islamic scholar judging whether an INCORRECT Arabic
span can be corrected to a real canonical text. A deterministic search has
already found the {k} canonical texts (Quran or the six Hadith books) sharing the
LONGEST contiguous wording with the span. Choose which one — if any — the span is
genuinely a distortion of, or rule "خطأ". You do NOT retrieve; you ONLY pick among
the provided candidates or reject them all. Never invent texts or sources.

────────────────────────────────────────────────────────────────────────
DECISION RULE
────────────────────────────────────────────────────────────────────────
Select a candidate ONLY IF, after ignoring tashkeel and normalizing letters
(أ إ آ → ا ; ى → ي ; ة → ه), a MEANINGFUL CONTIGUOUS PHRASE of the span matches
it AND the span as a whole is plausibly that text — misquoted, misdiacritized, or
lightly paraphrased — not merely on the same topic.

Output "خطأ" when EVERY candidate fails, even though search returned them:
  • Longest shared run is just ONE or TWO common words (الله، الحج، حرام).
  • Candidates share only the TOPIC.
  • Shared words are scattered, not contiguous.
Search returning a candidate is NOT evidence it is correct — overlap can be
incidental. Reject freely. A wrong correction is worse than "خطأ".
If two+ candidates are genuine distortions, pick the longest/closest contiguous
match.

────────────────────────────────────────────────────────────────────────
INPUT
────────────────────────────────────────────────────────────────────────
SPAN: <the incorrect span>
CANDIDATES (index | canonical text | source | matched_run = longest contiguous
            word-run shared with the span, as found by search):
  [0] text=<...> | source=<...> | matched_run=<...>
  [1] ...

────────────────────────────────────────────────────────────────────────
OUTPUT — valid JSON only, FIELDS IN THIS EXACT ORDER:
{{
  "rationale": "<brief: which candidate (if any) the span distorts, how long the
                real contiguous match is, and why ok or خطأ>",
  "verdict": "ok" | "خطأ",
  "verdict_confidence": <0.0-1.0>,
  "chosen_index": <int index of the chosen candidate, or -1 if خطأ>,
  "text": "<chosen candidate text, or empty>",
  "source": "<chosen source, or empty>",
  "confidence": <0.0-1.0, confidence in the chosen candidate; 0.0 if خطأ>
}}
  • verdict_confidence: if "خطأ" → how sure NONE of the candidates fit;
                        if "ok"  → how sure the span is correctable at all.
  • confidence (chosen-candidate): grade by the contiguous run length —
      0.8-1.0 long run, clearly this exact text; 0.5-0.7 real phrase + paraphrase;
      0.2-0.4 short phrase only. Below 0.2 → rule خطأ instead.
  • When "خطأ": chosen_index -1, text and source empty, confidence 0.0.

────────────────────────────────────────────────────────────────────────
EXAMPLES
────────────────────────────────────────────────────────────────────────
SPAN: "رَبِّنَا هَبْ لَنَا مِنَ الصَّالِحِينَ"
CANDIDATES:
  [0] text="رَبِّ هَبْ لِي مِنَ الصَّالِحِينَ" | source="سورة الصافات 100" | matched_run="هب من الصالحين"
  [1] text="رَبِّ هَبْ لِي حُكْمًا وَأَلْحِقْنِي بِالصَّالِحِينَ" | source="سورة الشعراء 83" | matched_run="رب هب"
{{"rationale": "[0] shares a 3-word contiguous run ending the verse; [1] shares only 'رب هب'. The span is a pronoun-altered distortion of [0].",
  "verdict": "ok", "verdict_confidence": 0.95, "chosen_index": 0,
  "text": "رَبِّ هَبْ لِي مِنَ الصَّالِحِينَ", "source": "سورة الصافات 100", "confidence": 0.9}}

SPAN: "وَالْحَجُّ أَشَدُّ الْفُرُوجِ"
CANDIDATES:
  [0] text="الْحَجُّ أَشْهُرٌ مَّعْلُومَاتٌ ..." | source="سورة البقرة 197" | matched_run="الحج"
  [1] text="وَالَّذِينَ هُمْ لِفُرُوجِهِمْ حَافِظُونَ" | source="سورة المؤمنون 5" | matched_run="فروج"
{{"rationale": "Longest run anywhere is a single word; no contiguous phrase matches. Topic + isolated words.",
  "verdict": "خطأ", "verdict_confidence": 0.9, "chosen_index": -1, "text": "", "source": "", "confidence": 0.0}}

SPAN: "السرقة حرام، وإن كانت لطعام وحاجة"
CANDIDATES:
  [0] text="لَعَنَ اللَّهُ السَّارِقَ ..." | source="صحيح البخاري 6783" | matched_run="السارق"
  [1] text="كُلُّ الْمُسْلِمِ عَلَى الْمُسْلِمِ حَرَامٌ ..." | source="صحيح مسلم 2564" | matched_run="حرام"
{{"rationale": "Only isolated topic words (السارق، حرام) match; no contiguous phrase. Same subject, different wording.",
  "verdict": "خطأ", "verdict_confidence": 0.88, "chosen_index": -1, "text": "", "source": "", "confidence": 0.0}}
"""

SCHEMA_JUDGE = {
    "type": "object",
    "properties": {
        "rationale":          {"type": "string"},
        "verdict":            {"type": "string", "enum": ["ok", "خطأ"]},
        "verdict_confidence": {"type": "number"},
        "chosen_index":       {"type": "integer"},
        "text":               {"type": "string"},
        "source":             {"type": "string"},
        "confidence":         {"type": "number"},
    },
    "required": ["rationale", "verdict", "verdict_confidence",
                 "chosen_index", "text", "source", "confidence"],
    "additionalProperties": False,
}

REF_LABEL = {"quran": "Quranic ayah (آية قرآنية)", "hadith": "Prophetic hadith (حديث نبوي)"}
REF_PLURAL = {"quran": "ayahs", "hadith": "hadiths"}


def build_generate_user_prompt(span: str, ref: str, k: int) -> str:
    return (
        f"The following span was presented as a {REF_LABEL.get(ref, ref)} but is "
        f"INCORRECT. Retrieve the top {k} closest canonical {REF_PLURAL.get(ref, ref)}, "
        f"or return \"خطأ\" if none is plausibly close.\n\n"
        f"Span:\n<<<\n{span}\n>>>"
    )


def build_judge_user_prompt(span: str, cands: List[Dict]) -> str:
    lines = [f"SPAN: {span}", "CANDIDATES:"]
    for i, c in enumerate(cands):
        lines.append(f'  [{i}] text={c["text"]} | source={c["source"]} | '
                     f'matched_run={c["matched_run"]}')
    return "\n".join(lines)


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
# RESUME / SAVE  (now RETRIES previously-errored records — open item (a))
# ─────────────────────────────────────────────────────────────────────────────
def load_resume_state(output_path: str):
    """Return (per_sample_kept, completed_uids).
    Errored records are DROPPED from per_sample and excluded from completed so they
    are retried on resume. completed is None when the file is already complete."""
    if not os.path.exists(output_path):
        return [], set()
    try:
        with open(output_path, encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception as e:
        print(f"  [WARN] could not read {output_path} for resume: {e}")
        return [], set()
    per_sample = data.get("per_sample", [])
    if data.get("status") == "complete" and not any(s.get("error") for s in per_sample):
        print(f"  [SKIP] {output_path} already complete -- delete it to re-run.")
        return per_sample, None
    kept = [s for s in per_sample if not s.get("error")]
    n_retry = len(per_sample) - len(kept)
    if kept:
        print(f"  [RESUME] {len(kept)} done"
              + (f", retrying {n_retry} previously-errored" if n_retry else "") + ".")
    return kept, {s["uid"] for s in kept}


def save_incremental(output_path: str, model_name: str, method: str, k: int,
                     per_sample: List[Dict], n_total: int) -> None:
    n_done = len(per_sample)
    tin, tout, cost = token_totals(per_sample, model_name)
    snapshot = {
        "task":        TASK_NAME,
        "method":      method,
        "model":       model_name,
        "top_k":       k,
        "timestamp":   datetime.now(timezone.utc).isoformat(),
        "status":      "complete" if n_done >= n_total else "in_progress",
        "n_total":     n_total,
        "n_completed": n_done,
        "n_failed":    sum(1 for s in per_sample if s.get("error")),
        "n_llm_calls": sum(1 for s in per_sample if s.get("usage")),
        "input_tokens":  tin,
        "output_tokens": tout,
        "est_cost_usd":  cost,
        "per_sample":  per_sample,
    }
    d = os.path.dirname(output_path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(snapshot, fh, ensure_ascii=False, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# WORKERS
# ─────────────────────────────────────────────────────────────────────────────
def _base_rec(sp: Dict, method: str) -> Dict:
    return {"uid": sp["uid"], "id": sp["id"], "answer_model": sp["model"],
            "ref": sp["ref"], "span": sp["span"], "method": method}


def correct_one_generate(model_name: str, sysmsg: str, k: int, sp: Dict) -> Dict:
    raw, usage = call_api(model_name, sysmsg,
                          build_generate_user_prompt(sp["span"], sp["ref"], k),
                          schema=SCHEMA_GENERATE)
    rec = _base_rec(sp, "generate")
    if raw is None:
        rec.update({"prediction": None, "raw_response": None, "usage": None,
                    "error": "API call failed"})
    else:
        rec.update({"prediction": parse_json_response(raw), "raw_response": raw, "usage": usage})
    return rec


def correct_one_judge(model_name: str, sysmsg: str, k: int, sp: Dict) -> Dict:
    import corpus_search
    rec = _base_rec(sp, "judge")
    cands = corpus_search.search(sp["span"], sp["ref"], k=k)
    rec["search_candidates"] = cands
    if not cands:
        # search gate rejected everything → خطأ, no LLM call.
        rec.update({
            "prediction": {
                "rationale": "no canonical candidate met the search gate "
                             "(>=3 word overlap with a contiguous run)",
                "verdict": "خطأ", "verdict_confidence": 1.0,
                "chosen_index": -1, "text": "", "source": "", "confidence": 0.0,
            },
            "raw_response": None, "usage": None, "search_empty": True,
        })
        return rec
    raw, usage = call_api(model_name, sysmsg,
                          build_judge_user_prompt(sp["span"], cands),
                          schema=SCHEMA_JUDGE)
    if raw is None:
        rec.update({"prediction": None, "raw_response": None, "usage": None,
                    "error": "API call failed"})
    else:
        rec.update({"prediction": parse_json_response(raw), "raw_response": raw, "usage": usage})
    return rec


# ─────────────────────────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────────────────────────
def run(method: str, model_name: str, input_path: str, output_path: str, k: int,
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
    print(f"\n=== {TASK_NAME} | {method} | {model_name} | top-{k} | "
          f"{len(todo)} of {n_total} spans | {workers} workers ===")

    if method == "generate":
        sysmsg = GENERATION_PROMPT.format()           # collapse {{ }} → { }
        worker = correct_one_generate
    else:
        sysmsg = JUDGE_PROMPT.format(k=k)             # collapse {{ }} and inject k
        worker = correct_one_judge

    lock = threading.Lock()
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(worker, model_name, sysmsg, k, sp): sp for sp in todo}
        for fut in as_completed(futures):
            rec = fut.result()
            with lock:
                per_sample.append(rec)
                done += 1
                if rec.get("error"):
                    tag = "FAILED"
                elif rec.get("search_empty"):
                    tag = "خطأ (no search hit)"
                else:
                    p = rec.get("prediction") or {}
                    if method == "generate":
                        tag = f"{p.get('verdict','?')}, {len(p.get('candidates',[]) or [])} cand"
                    else:
                        tag = f"{p.get('verdict','?')}, idx={p.get('chosen_index','?')} ({len(rec['search_candidates'])} cand)"
                print(f"  [{done}/{len(todo)}] {rec['id']}/{rec['answer_model']}/{rec['ref']} -> {tag}")
                if done % 10 == 0:
                    save_incremental(output_path, model_name, method, k, per_sample, n_total)

    save_incremental(output_path, model_name, method, k, per_sample, n_total)
    tin, tout, cost = token_totals(per_sample, model_name)
    n_calls = sum(1 for s in per_sample if s.get("usage"))
    n_skip = sum(1 for s in per_sample if s.get("search_empty"))
    cost_str = f"~${cost:.4f}" if cost is not None else "n/a (no price)"
    print(f"  -> wrote {output_path}")
    print(f"  LLM calls: {n_calls}"
          + (f" (+{n_skip} auto-خطأ, no call)" if n_skip else "")
          + f" | tokens in {tin:,} / out {tout:,} | est cost {cost_str}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Recover correct references for incorrect citation spans")
    ap.add_argument("--method", choices=["generate", "judge"], default="generate",
                    help="generate = model retrieves from memory; "
                         "judge = corpus search proposes, model picks")
    ap.add_argument("--model", required=True,
                    help="model, one of: " + ", ".join(MODEL_REGISTRY.keys()))
    ap.add_argument("--input", default="../outputs/correction/citation_spans.csv",
                    help="CSV from extract_spans_for_correction.py (id,model,span,ref,correctness)")
    ap.add_argument("--output", default=None,
                    help="output JSON (default: ../outputs/correction/corrections_<method>_<model>.json)")
    ap.add_argument("--top-k", type=int, default=5, help="number of candidate references")
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

    out = args.output or f"../outputs/correction/corrections_{args.method}_{args.model}.json"
    run(args.method, args.model, args.input, out, args.top_k,
        incorrect_only=not args.include_correct, max_samples=args.max_samples,
        workers=args.workers)


if __name__ == "__main__":
    main()
