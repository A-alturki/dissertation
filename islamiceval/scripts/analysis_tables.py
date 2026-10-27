#!/usr/bin/env python3
"""
analysis_tables.py
==================
Reproduces the per-model tables of the dissertation's empirical-analysis
chapter (Chapter 5) from the definitive analysis files in
``data/__FINAL_ANALYSIS_SET__/``.

Tables produced (printed + written as CSV to outputs/analysis/tables/):

  explicit_faithfulness.csv   Table "Citation faithfulness under the explicit
                              prompt": answers, citations, cit./ans., coverage,
                              Quran %, Hadith %, combined %  (12 models)
  error_form.csv              Table "The form of unfaithfulness": inaccuracy,
                              errors n, baseless % (all / Quran / Hadith)
  implicit_faithfulness.csv   Table "Citation faithfulness under the implicit
                              prompt": same columns as explicit, 20 models
  attribution_isnad_relevance.csv
                              Per-model misattribution rate, isnad accuracy,
                              relevance rate (explicit set), plus pooled rows.

Definitions (identical to cost_of_asking.py)
  citation   = annotation segment with type in {Ayah, matn}
  correct    = segment label == "correct"
  coverage   = share of annotated answers with >= 1 citation
  baseless   = incorrect citation whose annotation-level ``correction`` is the
               literal string "خطأ" (correction stage found no canonical text);
               the complement (a list of canonical texts) is a "distortion".
  attribution: claimed_source segments labelled correct/incorrect (N/A = no
               source claimed -> excluded), counted ONLY on citations whose
               text (Ayah/matn) is correct; misattribution = incorrect share.
  isnad      : isnad segments labelled correct/incorrect (N/A excluded) on
               Hadith citations whose matn is correct; "supplied" = share of
               correct matns carrying a judged isnad; mismatch = incorrect share.
  relevance  : label 1/0 rows in train_task_4.tsv + dev_task_4.tsv, joined to
               the model through private_mapping.csv (response_id -> model).

Inputs
  data/__FINAL_ANALYSIS_SET__/fanar_all_corrected.jsonl     explicit, 12 models
  data/__FINAL_ANALYSIS_SET__/vanilla12_corrected.jsonl     implicit, 12 models
  data/__FINAL_ANALYSIS_SET__/vanilla8_corrected.jsonl      implicit, +8 models
  data/__FINAL_ANALYSIS_SET__/{train,dev}_task_4.tsv        relevance gold
  data/islamiceval2026_RELEASE/private_mapping.csv          response_id -> model

Usage
  python scripts/analysis_tables.py            # prints all tables, writes CSVs
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

REPO = Path(__file__).resolve().parent.parent
FINAL = REPO / "data" / "__FINAL_ANALYSIS_SET__"
MAPPING = REPO / "data" / "islamiceval2026_RELEASE" / "private_mapping.csv"
OUT = REPO / "outputs" / "analysis" / "tables"

CIT_TYPES = {"Ayah", "matn"}
KHATA = "خطأ"

FRONTIER = {"gpt-5.4", "gemini-3.1-pro"}
DISPLAY = {
    "gemini-3.1-pro": "Gemini-3.1-Pro", "gpt-5.4": "GPT-5.4",
    "allam-7b": "ALLaM-7B", "fanar-1-9b": "Fanar-1-9B", "jais-13b": "Jais-13B",
    "acegpt-8b": "AceGPT-v2-8B", "command-r-7b": "Command-R-7B",
    "gemma-3-12b": "Gemma-3-12B", "qwen3-8b": "Qwen3-8B",
    "llama-3.1-8b": "Llama-3.1-8B", "phi-3.5": "Phi-3.5-mini",
    "mistral-7b": "Mistral-7B-v0.3",
}


def norm_model(m: str) -> str:
    return "gemini-3.1-pro" if m == "gemini-pro-3.1" else m


def pct(k, n):
    return 100.0 * k / n if n else float("nan")


# ----------------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------------
def load_mapping():
    with open(MAPPING, encoding="utf-8") as f:
        return {r["new_id"]: norm_model(r["source_model"]) for r in csv.DictReader(f)}


def iter_answers(path: Path, mapping=None):
    """Yield (model, question_id, response_id, annotations)."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            model = norm_model(r["model"]) if "model" in r else mapping.get(r["id"])
            if model is None:
                continue
            yield model, r["question_id"], r["id"], r["annotations"]


# ----------------------------------------------------------------------------
# per-model accumulators
# ----------------------------------------------------------------------------
def new_stats():
    return {
        "answers": 0, "answers_with_cite": 0,
        "cites": 0, "cites_ok": 0,
        "quran": 0, "quran_ok": 0, "hadith": 0, "hadith_ok": 0,
        "quran_bad": 0, "quran_baseless": 0, "hadith_bad": 0, "hadith_baseless": 0,
        "correct_spans": 0, "matn_correct": 0,
        "attr": 0, "attr_bad": 0, "isnad": 0, "isnad_ok": 0,
        "rel": 0, "rel_ok": 0,
    }


def accumulate(path: Path, mapping=None):
    stats = defaultdict(new_stats)
    for model, _qid, _rid, anns in iter_answers(path, mapping):
        s = stats[model]
        s["answers"] += 1
        has_cite = False
        for a in anns:
            corr = a.get("correction")
            segs = a.get("segments", [])
            text_segs = [x for x in segs if x.get("type") in CIT_TYPES]
            text_ok = bool(text_segs) and text_segs[0].get("label") == "correct"
            if text_ok:
                s["correct_spans"] += 1
                if text_segs[0]["type"] == "matn":
                    s["matn_correct"] += 1
            for seg in segs:
                t, lab = seg.get("type"), seg.get("label")
                if t in CIT_TYPES:
                    has_cite = True
                    s["cites"] += 1
                    ok = lab == "correct"
                    s["cites_ok"] += ok
                    key = "quran" if t == "Ayah" else "hadith"
                    s[key] += 1
                    s[key + "_ok"] += ok
                    if lab == "incorrect":
                        s[key + "_bad"] += 1
                        s[key + "_baseless"] += (corr == KHATA)
                # attribution / isnad are judged only on citations whose TEXT
                # is correct (the chapter's definition); N/A = none claimed.
                elif text_ok and t == "claimed_source" and lab in ("correct", "incorrect"):
                    s["attr"] += 1
                    s["attr_bad"] += lab == "incorrect"
                elif text_ok and t == "isnad" and lab in ("correct", "incorrect"):
                    s["isnad"] += 1
                    s["isnad_ok"] += lab == "correct"
        s["answers_with_cite"] += has_cite
    return stats


def add_relevance(stats, mapping):
    for name in ("train_task_4.tsv", "dev_task_4.tsv"):
        with open(FINAL / name, encoding="utf-8") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                lab = r["relevance_label"].strip()
                if lab not in ("0", "1"):
                    continue
                model = mapping.get(r["response_id"])
                if model is None:
                    continue
                stats[model]["rel"] += 1
                stats[model]["rel_ok"] += lab == "1"


def pooled(stats, models):
    p = new_stats()
    for m in models:
        for k, v in stats[m].items():
            p[k] += v
    return p


# ----------------------------------------------------------------------------
# table writers
# ----------------------------------------------------------------------------
def faithfulness_rows(stats, models):
    rows = []
    for m in models:
        s = stats[m]
        rows.append({
            "model": DISPLAY.get(m, m),
            "answers": s["answers"],
            "citations": s["cites"],
            "cit_per_ans": round(s["cites"] / s["answers"], 2) if s["answers"] else "",
            "coverage_pct": round(pct(s["answers_with_cite"], s["answers"]), 1),
            "quran_pct": round(pct(s["quran_ok"], s["quran"]), 1),
            "hadith_pct": round(pct(s["hadith_ok"], s["hadith"]), 1),
            "combined_pct": round(pct(s["cites_ok"], s["cites"]), 1),
        })
    return rows


def error_form_rows(stats, models):
    rows = []
    for m in models:
        s = stats[m]
        bad = s["quran_bad"] + s["hadith_bad"]
        base = s["quran_baseless"] + s["hadith_baseless"]
        rows.append({
            "model": DISPLAY.get(m, m),
            "inaccuracy_pct": round(pct(s["cites"] - s["cites_ok"], s["cites"]), 1),
            "errors_n": bad,
            "baseless_pct": round(pct(base, bad), 1),
            "quran_baseless_pct": round(pct(s["quran_baseless"], s["quran_bad"]), 1),
            "hadith_baseless_pct": round(pct(s["hadith_baseless"], s["hadith_bad"]), 1),
        })
    return rows


def attr_rows(stats, models):
    rows = []
    for m in models:
        s = stats[m]
        rows.append({
            "model": DISPLAY.get(m, m),
            "correct_spans": s["correct_spans"],
            "attribution_n": s["attr"],
            "misattribution_pct": round(pct(s["attr_bad"], s["attr"]), 1),
            "matn_correct_n": s["matn_correct"],
            "isnad_n": s["isnad"],
            "isnad_supplied_pct": round(pct(s["isnad"], s["matn_correct"]), 1),
            "isnad_mismatch_pct": round(pct(s["isnad"] - s["isnad_ok"], s["isnad"]), 1),
            "relevance_n": s["rel"],
            "relevant_pct": round(pct(s["rel_ok"], s["rel"]), 1),
        })
    return rows


def show(title, rows):
    print(f"\n== {title}")
    if not rows:
        return
    cols = list(rows[0].keys())
    w = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in cols}
    print("  ".join(c.ljust(w[c]) for c in cols))
    for r in rows:
        print("  ".join(str(r[c]).ljust(w[c]) for c in cols))


def write_csv(name, rows):
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / name, "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)


def main():
    mapping = load_mapping()

    # ---- explicit (12 models) -------------------------------------------
    ex = accumulate(FINAL / "fanar_all_corrected.jsonl", mapping)
    add_relevance(ex, mapping)
    ex_models = sorted(ex, key=lambda m: -pct(ex[m]["cites_ok"], ex[m]["cites"]))
    open_models = [m for m in ex_models if m not in FRONTIER]
    ex["__OPEN__"] = pooled(ex, open_models)
    ex["__FRONTIER__"] = pooled(ex, [m for m in ex_models if m in FRONTIER])
    DISPLAY["__OPEN__"], DISPLAY["__FRONTIER__"] = "Open-weight pooled", "Frontier pooled"

    rows = faithfulness_rows(ex, ex_models + ["__OPEN__", "__FRONTIER__"])
    show("Explicit prompt: citation faithfulness (tab:explicit-faithfulness)", rows)
    write_csv("explicit_faithfulness.csv", rows)

    rows = error_form_rows(ex, ex_models + ["__OPEN__", "__FRONTIER__"])
    show("Explicit prompt: form of unfaithfulness (tab:error-form)", rows)
    write_csv("error_form.csv", rows)

    rows = attr_rows(ex, ex_models + ["__OPEN__", "__FRONTIER__"])
    show("Explicit prompt: misattribution / isnad / relevance per model", rows)
    write_csv("attribution_isnad_relevance.csv", rows)

    # ---- implicit (20 models) -------------------------------------------
    im = accumulate(FINAL / "vanilla12_corrected.jsonl")
    for m, s in accumulate(FINAL / "vanilla8_corrected.jsonl").items():
        if m in im:
            raise ValueError(f"model {m} present in both vanilla files")
        im[m] = s
    im_models = sorted(im, key=lambda m: -pct(im[m]["cites_ok"], im[m]["cites"]))
    rows = faithfulness_rows(im, im_models)
    show("Implicit prompt: citation faithfulness, 20 models (tab:implicit-faithfulness)", rows)
    write_csv("implicit_faithfulness.csv", rows)

    rows = error_form_rows(im, im_models)
    show("Implicit prompt: form of unfaithfulness, 20 models", rows)
    write_csv("implicit_error_form.csv", rows)

    print(f"\nCSV tables written to {OUT}")


if __name__ == "__main__":
    main()
