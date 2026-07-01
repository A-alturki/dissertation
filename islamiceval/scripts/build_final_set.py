#!/usr/bin/env python3
"""
build_final_set.py — assemble the FINAL cleaned question set + final per-model answers.

Final prompts = level-1 word-cleaned survivors (questions_level1_wordcleaned.json)
                MINUS the vulgarity-flash exclusions the human CONFIRMED remove
                (vulgarity_review_38.xlsx, decision column; rows marked 'keep' stay in).

Final answers = each model's CLEANED answers (*_clean.json) intersected with the final
                prompt set, for all models EXCEPT deepseek-r1-llama-8b and qwen3.5-9b
                (excluded for low quality). -> outputs/answers/final/<model>_final.json

Outputs:
  outputs/classification/questions_final_cleaned.json       final prompts [{qid,prompt}]
  outputs/classification/questions_removed_log_final.json   every removed prompt + stage/reason
  outputs/classification/final_set_stats.json               counts
  outputs/answers/final/<model>_final.json                  final per-model answers
"""
import os, json, glob
import pandas as pd
from collections import Counter

CLS = "../outputs/classification"
ANS = "../outputs/answers/explicit_large_token_size"
OUT = "../outputs/answers/final"
EXCLUDE_MODELS = {"deepseek-r1-llama-8b", "qwen3.5-9b"}


def main():
    survivors = json.load(open(f"{CLS}/questions_level1_wordcleaned.json", encoding="utf-8"))
    surv_prompt = {r["qid"]: r["prompt"] for r in survivors}

    # human review of the 38 vulgarity exclusions
    rv = pd.read_excel(f"{CLS}/vulgarity_review_38.xlsx", sheet_name="review")
    rv["decision"] = rv["decision"].astype(str).str.strip().str.lower()
    vulg_remove = {r.qid: r.flash for r in rv.itertuples() if r.decision == "remove"}
    vulg_keep = [r.qid for r in rv.itertuples() if r.decision == "keep"]
    assert set(vulg_remove) | set(vulg_keep) >= set(rv["qid"]), "some review rows have no keep/remove"

    final_qids = [q for q in surv_prompt if q not in vulg_remove]      # keeps stay, removes drop
    final = [{"qid": q, "prompt": surv_prompt[q]} for q in final_qids]
    final_set = set(final_qids)
    json.dump(final, open(f"{CLS}/questions_final_cleaned.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    # rebuild the complete removal log: word stages (severity+vulgar+sectarian) + confirmed vulgarity
    word_removed = json.load(open(f"{CLS}/questions_removed_log.json", encoding="utf-8"))
    removed = list(word_removed)
    rv_by_qid = {r.qid: r for r in rv.itertuples()}
    for q, flash in vulg_remove.items():
        code = flash.split("|")[1]
        removed.append({"qid": q, "stage": "vulgarity", "reason": "vulg_" + code,
                        "matched": flash, "prompt": rv_by_qid[q].prompt})
    json.dump(removed, open(f"{CLS}/questions_removed_log_final.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    # ── final per-model answers = cleaned answers ∩ final prompts ──
    os.makedirs(OUT, exist_ok=True)
    model_rows = []
    grand = 0
    for path in sorted(glob.glob(f"{ANS}/*_alt2025-explicit_clean.json")):
        model = os.path.basename(path).split("_alt2025")[0]
        if model in EXCLUDE_MODELS:
            continue
        clean = json.load(open(path, encoding="utf-8"))
        final_ans = [r for r in clean if r.get("id") in final_set]
        json.dump(final_ans, open(f"{OUT}/{model}_final.json", "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        model_rows.append((model, len(clean), len(final_ans)))
        grand += len(final_ans)

    # ── stats ──
    by_stage = Counter(r["stage"] for r in removed)
    stats = {
        "start_prompts": 8323,
        "final_prompts": len(final),
        "removed_total": 8323 - len(final),
        "removed_by_stage": dict(by_stage),
        "vulgarity_reviewed": {"keep": len(vulg_keep), "remove": len(vulg_remove)},
        "excluded_models": sorted(EXCLUDE_MODELS),
        "final_models": len(model_rows),
        "final_answers_total": grand,
        "per_model": {m: {"clean": c, "final": f} for m, c, f in model_rows},
    }
    json.dump(stats, open(f"{CLS}/final_set_stats.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    # ── report ──
    print("================ FINAL SET ================")
    print(f"  start prompts                 8323")
    print(f"  removed by stage: " + ", ".join(f"{k}={v}" for k, v in by_stage.items()))
    print(f"  vulgarity human review: keep {len(vulg_keep)}, remove {len(vulg_remove)}")
    print(f"  FINAL PROMPTS                 {len(final)}  ({100*len(final)/8323:.1f}%)")
    print(f"\n  final per-model answers (cleaned ∩ final prompts), excl {sorted(EXCLUDE_MODELS)}:")
    print(f"  {'model':24}{'clean':>8}{'final':>8}{'dropped(prompt removed)':>26}")
    for m, c, f in model_rows:
        print(f"  {m:24}{c:8}{f:8}{c-f:26}")
    print(f"  {'-'*66}")
    print(f"  {'TOTAL ('+str(len(model_rows))+' models)':24}{sum(c for _,c,_ in model_rows):8}{grand:8}")
    print(f"\n-> {CLS}/questions_final_cleaned.json ({len(final)})")
    print(f"-> {CLS}/questions_removed_log_final.json ({len(removed)})")
    print(f"-> {OUT}/<model>_final.json  ({len(model_rows)} models)")
    print(f"-> {CLS}/final_set_stats.json")


if __name__ == "__main__":
    main()
