#!/usr/bin/env python3
"""Independent verification of outputs/answers/final/ — the final deliverable.
Re-derives expected contents from source and asserts every invariant. Prints PASS/FAIL."""
import os, json, glob, re, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CLS = "../outputs/classification"
ANS = "../outputs/answers/explicit_large_token_size"
FIN = "../outputs/answers/final"
EXPECTED_MODELS = {"acegpt-8b", "allam-7b", "command-r-7b", "fanar-1-9b", "gemma-3-12b",
                   "jais-13b", "llama-3.1-8b", "mistral-7b", "phi-3.5", "qwen3-8b"}
EXCLUDED = {"deepseek-r1-llama-8b", "qwen3.5-9b"}
MOJIBAKE = re.compile(r"[ØÙÚÛ][\x80-\xFF-¿]|Ã[-¿]|Ø§Ù|Ù†|Ø¨")

results = []
def check(name, ok, detail=""):
    results.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))

final_prompts = json.load(open(f"{CLS}/questions_final_cleaned.json", encoding="utf-8"))
final_qids = {r["qid"] for r in final_prompts}
canon_prompt = {r["qid"]: r["prompt"] for r in final_prompts}
removed = json.load(open(f"{CLS}/questions_removed_log_final.json", encoding="utf-8"))
removed_qids = {r["qid"] for r in removed}
stats = json.load(open(f"{CLS}/final_set_stats.json", encoding="utf-8"))

print("=== STRUCTURE ===")
files = {os.path.basename(p).replace("_final.json", ""): p for p in glob.glob(f"{FIN}/*_final.json")}
check("exactly the 10 expected models present", set(files) == EXPECTED_MODELS,
      f"got {sorted(set(files)^EXPECTED_MODELS)} diff" if set(files) != EXPECTED_MODELS else "")
check("no excluded models in final/", not (set(files) & EXCLUDED))

print("\n=== PROMPT SET INTEGRITY ===")
check("final + removed == 8323 (no loss/dup)", len(final_qids) + len(removed_qids) == 8323,
      f"{len(final_qids)}+{len(removed_qids)}={len(final_qids)+len(removed_qids)}")
check("final and removed disjoint", not (final_qids & removed_qids),
      f"overlap={len(final_qids & removed_qids)}")
# the 22 confirmed vulgarity removes must be gone; the 16 keeps must be present
import pandas as pd
rv = pd.read_excel(f"{CLS}/vulgarity_review_38.xlsx", sheet_name="review")
rv["decision"] = rv["decision"].astype(str).str.strip().str.lower()
keeps = set(rv[rv.decision == "keep"]["qid"]); rems = set(rv[rv.decision == "remove"]["qid"])
check("all 16 review-keeps in final prompts", keeps <= final_qids, f"missing {keeps-final_qids}")
check("all 22 review-removes absent from final prompts", not (rems & final_qids), f"leaked {rems & final_qids}")

print("\n=== PER-MODEL ANSWER FILES ===")
all_ans_qids = set()
total = 0
for m in sorted(EXPECTED_MODELS):
    fin = json.load(open(files[m], encoding="utf-8"))
    clean = json.load(open(f"{ANS}/{m}_alt2025-explicit_clean.json", encoding="utf-8"))
    clean_by_id = {r["id"]: r for r in clean}
    ids = [r.get("id") for r in fin]
    idset = set(ids)
    all_ans_qids |= idset
    total += len(fin)
    # invariants
    schema_ok = all(set(("id", "prompt", "answer", "model")) <= set(r) for r in fin)
    nonempty = all((r.get("answer") or "").strip() and (r.get("prompt") or "").strip() for r in fin)
    nodup = len(ids) == len(idset)
    subset = idset <= final_qids                       # no answer to a removed/non-final prompt
    # independent reconstruction: final == clean ∩ final_qids, same answer text
    expect = {i for i in clean_by_id if i in final_qids}
    recon_ids = (idset == expect)
    recon_text = all(fin_r["answer"] == clean_by_id[fin_r["id"]]["answer"]
                     and fin_r["prompt"] == clean_by_id[fin_r["id"]]["prompt"]
                     for fin_r in fin if fin_r["id"] in clean_by_id)
    prompt_match = all(r["prompt"] == canon_prompt.get(r["id"]) for r in fin)
    moji = sum(1 for r in fin if MOJIBAKE.search(r.get("prompt", "") + r.get("answer", "")))
    stat_ok = stats["per_model"].get(m, {}).get("final") == len(fin)
    ok = all([schema_ok, nonempty, nodup, subset, recon_ids, recon_text, prompt_match, moji == 0, stat_ok])
    flags = []
    if not schema_ok: flags.append("schema")
    if not nonempty: flags.append("empty-field")
    if not nodup: flags.append("dup-id")
    if not subset: flags.append(f"non-final-id×{len(idset-final_qids)}")
    if not recon_ids: flags.append(f"recon-mismatch(±{len(idset^expect)})")
    if not recon_text: flags.append("answer/prompt-text-differs-from-clean")
    if not prompt_match: flags.append("prompt≠canonical")
    if moji: flags.append(f"mojibake×{moji}")
    if not stat_ok: flags.append("count≠stats")
    check(f"{m:14} n={len(fin):5}", ok, ", ".join(flags))

print("\n=== CROSS-FILE ===")
check("every answered qid is a final prompt", all_ans_qids <= final_qids,
      f"stray {len(all_ans_qids-final_qids)}")
check("no removed qid answered in any model", not (all_ans_qids & removed_qids),
      f"leaked {len(all_ans_qids & removed_qids)}")
check("grand total matches stats", total == stats["final_answers_total"],
      f"{total} vs {stats['final_answers_total']}")

print(f"\n{'='*50}\n{'ALL CHECKS PASSED ✅' if all(results) else 'SOME CHECKS FAILED ❌'}  ({sum(results)}/{len(results)})")
