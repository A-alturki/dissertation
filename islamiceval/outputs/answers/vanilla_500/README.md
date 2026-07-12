# vanilla_500 — vanilla-prompt generations on the 500 annotated questions

Answers from the **10 original IslamicEval models** to the **500 annotated questions**
(`data/islamiceval2026_RELEASE/batch1_qids_500.txt`, verified against
`private_mapping.csv`) under the **vanilla** system prompt:

> أجب عن السؤال التالي.

Purpose: a minimal-instruction baseline to compare against the original
`alt2025-explicit` (and `default`) prompt runs on the same annotated subset.

## How it was produced

- Engine: `scripts/inference_hugging_face.py` (HF Transformers, `--no-quantize`).
- Driver: `scripts/run_vanilla_box.sh` — all 10 models via HF, split 5+5 across 2 GPUs,
  running in parallel; each model downloads → generates → its HF cache is deleted → next
  (box has ~153 GB disk).
- Prompt: `--prompt vanilla`. Thinking **allowed** (`--allow-thinking`): Qwen3 reasons and
  its `<think>…</think>` is split into a separate `thinking` field (`answer` keeps the final text).
- Sampling: each model's **own** generation_config defaults (temperature/top_p/top_k).
- Token budget: 1500 (qwen3-8b: 3000 for thinking headroom).
- Input: `data/islamiceval2026_RELEASE/batch1_500_questions.json` (id = qid, prompt = question text
  taken from the RELEASE jsonl; all 500 present and text-consistent).

## Files

- `<model>_vanilla.json` — list of `{id, prompt, answer, model[, thinking]}` (500 rows/model).
- `logs/<model>.log` — per-model run log.

Models: allam-7b, command-r-7b, fanar-1-9b, gemma-3-12b, jais-13b, llama-3.1-8b,
mistral-7b, phi-3.5, qwen3-8b, acegpt-8b.
