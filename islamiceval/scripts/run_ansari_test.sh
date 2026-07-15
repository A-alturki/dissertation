#!/usr/bin/env bash
# ==============================================================================
# run_ansari_test.sh — Stage-2 generation for the OFFICIAL Ansari TEST-500
# (ansari_test_500.json), all 10 original models, on GPU 3 (single GPU, SEQUENTIAL).
#
# GPUs 0+1 are busy with the big-model run, so this pins CUDA_VISIBLE_DEVICES=3 and
# runs one model at a time. Same engine routing as before:
#   * 8 non-Gemma models -> vLLM (single GPU, tp=1). vLLM auto-sets the Turing-safe
#     config: VLLM_USE_FLASHINFER_SAMPLER=0 + TRITON_ATTN. fp16 (Turing has no fast bf16).
#   * 2 Gemma-family (fanar-1-9b, gemma-3-12b) -> HF (vLLM can't run Gemma attention
#     on Turing/sm_75). model-aware dtype (Gemma->bf16) via --no-quantize.
#
# Protocol = shared-task train/dev parity (the script defaults):
#   --prompt alt2025-explicit, thinking OFF (no --allow-thinking), --max-tokens 1500,
#   seed 42, each model's own default sampling.
#
# Per model: skip if already done -> generate -> delete HF cache (free disk). Resumable.
# Prereqs: `huggingface-cli login` (gemma-3-12b, llama-3.1-8b gated); vLLM + HF deps.
# Usage:   bash run_ansari_test.sh
# ==============================================================================
set -u
cd "$(dirname "$0")"                       # -> scripts/

# Activate the project venv (vLLM 0.22 + transformers 5.10 live here, NOT system python3).
VENV="$(cd .. && pwd)/venv"
if [ -f "$VENV/bin/activate" ]; then
  # shellcheck disable=SC1090
  source "$VENV/bin/activate"
else
  echo "[WARN] venv not found at $VENV — using system python (vLLM may be missing)."
fi
PY="${PYTHON:-python3}"
export CUDA_VISIBLE_DEVICES=3

INPUT="../data/islamiceval2026_RELEASE/ansari_test_500.json"
OUTDIR="../outputs/answers/ansari_test_500"
LOGDIR="$OUTDIR/logs"
PROMPT="alt2025-explicit"
MAXTOK=1500
mkdir -p "$OUTDIR" "$LOGDIR"

# vLLM models first (fast, results early), the 2 slow HF-Gemma models last.
ORDER=(allam-7b command-r-7b jais-13b llama-3.1-8b mistral-7b phi-3.5 qwen3-8b acegpt-8b fanar-1-9b gemma-3-12b)
declare -A ENGINE=(
  [allam-7b]=vllm [command-r-7b]=vllm [jais-13b]=vllm [llama-3.1-8b]=vllm [mistral-7b]=vllm
  [phi-3.5]=vllm  [qwen3-8b]=vllm     [acegpt-8b]=vllm
  [fanar-1-9b]=hf [gemma-3-12b]=hf
)
declare -A REPO=(
  [allam-7b]="ALLaM-AI/ALLaM-7B-Instruct-preview"
  [command-r-7b]="CohereForAI/c4ai-command-r7b-12-2024"
  [jais-13b]="inceptionai/jais-13b-chat"
  [llama-3.1-8b]="meta-llama/Llama-3.1-8B-Instruct"
  [mistral-7b]="mistralai/Mistral-7B-Instruct-v0.3"
  [phi-3.5]="microsoft/Phi-3.5-mini-instruct"
  [qwen3-8b]="Qwen/Qwen3-8B"
  [acegpt-8b]="FreedomIntelligence/AceGPT-v2-8B-Chat"
  [fanar-1-9b]="QCRI/Fanar-1-9B-Instruct"
  [gemma-3-12b]="google/gemma-3-12b-it"
)
HF_HUB="${HF_HOME:-$HOME/.cache/huggingface}/hub"

# already fully generated? (500 answers, none are ERROR) -> skip.
is_done() {
  "$PY" - "$OUTDIR/$1_${PROMPT}.json" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1], encoding="utf-8"))
    ok = [r for r in d if not str(r.get("answer", "")).startswith("ERROR")]
    sys.exit(0 if len(ok) >= 500 else 1)
except Exception:
    sys.exit(1)
PY
}

echo "Input : $INPUT"
echo "Output: $OUTDIR   (GPU 3, sequential)"
echo "Prompt: $PROMPT  (thinking OFF, max_tokens=$MAXTOK, seed 42)"

for model in "${ORDER[@]}"; do
  eng="${ENGINE[$model]}"; repo="${REPO[$model]}"; log="$LOGDIR/$model.log"
  echo "============================================================"
  echo "[$model] engine=$eng repo=$repo"
  if is_done "$model"; then echo "  already complete — skipping."; continue; fi

  if [ "$eng" = "vllm" ]; then
    "$PY" inference_vllm.py --model "$model" --input "$INPUT" --output-dir "$OUTDIR" \
        --prompt "$PROMPT" --dtype float16 --tensor-parallel 1 \
        --max-tokens "$MAXTOK" --seed 42 >> "$log" 2>&1
  else   # hf, Gemma-family, model-aware dtype (bf16)
    "$PY" inference_hugging_face.py --model "$model" --input "$INPUT" --output-dir "$OUTDIR" \
        --prompt "$PROMPT" --no-quantize --max-tokens "$MAXTOK" --batch-size 16 >> "$log" 2>&1
  fi
  rc=$?
  echo "  rc=$rc; removing cache."
  rm -rf "$HF_HUB/models--${repo//\//--}"
done

echo "============================================================"
echo "Done. Answers in $OUTDIR/<model>_${PROMPT}.json ; logs in $LOGDIR/"
