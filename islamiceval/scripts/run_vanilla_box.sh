#!/usr/bin/env bash
# ==============================================================================
# run_vanilla_box.sh — Stage-2 vanilla-prompt generation for the 500-question
# annotated batch (batch1_qids_500.txt), all 10 original models, on the GPU box.
#
# Strategy: run everything through HuggingFace (inference_hugging_face.py) — at
# only 500 questions HF generate() is fast enough, no vLLM/Turing backend fuss.
# The 10 models are split 5+5 across 2 GPUs and the two halves run IN PARALLEL.
# Within each half models run one at a time: download -> generate -> delete its
# HF cache -> next model, so peak disk is ~one model per GPU (box has ~153 GB).
#
# Per run: vanilla system prompt ("أجب عن السؤال التالي."), thinking ALLOWED
# (Qwen3 reasons; its <think> is split into a 'thinking' field), and each model's
# OWN default sampling (temperature/top_p/top_k from its generation_config).
#
# Prereqs on the box:
#   - `huggingface-cli login` (gemma-3-12b and llama-3.1-8b are GATED).
#   - deps: torch, transformers(>=5), accelerate, bitsandbytes(not needed here), tqdm.
#   - >=2 visible GPUs (RTX 8000 48GB fits every model in fp16/bf16 unquantized).
#
# Usage:   bash run_vanilla_box.sh
# Resume:  just re-run — finished models (500 non-error answers) are skipped, and
#          each model's own output file resumes mid-way if interrupted.
# ==============================================================================
set -u
cd "$(dirname "$0")"                       # -> scripts/
PY="${PYTHON:-python3}"

INPUT="../data/islamiceval2026_RELEASE/batch1_500_questions.json"
OUTDIR="../outputs/answers/vanilla_500"
LOGDIR="$OUTDIR/logs"
PROMPT="vanilla"
BATCH=16
mkdir -p "$OUTDIR" "$LOGDIR"

# model key -> HF repo id (keep in sync with inference_hugging_face.py MODELS)
declare -A REPO=(
  [allam-7b]="ALLaM-AI/ALLaM-7B-Instruct-preview"
  [command-r-7b]="CohereForAI/c4ai-command-r7b-12-2024"
  [fanar-1-9b]="QCRI/Fanar-1-9B-Instruct"
  [gemma-3-12b]="google/gemma-3-12b-it"
  [jais-13b]="inceptionai/jais-13b-chat"
  [llama-3.1-8b]="meta-llama/Llama-3.1-8B-Instruct"
  [mistral-7b]="mistralai/Mistral-7B-Instruct-v0.3"
  [phi-3.5]="microsoft/Phi-3.5-mini-instruct"
  [qwen3-8b]="Qwen/Qwen3-8B"
  [acegpt-8b]="FreedomIntelligence/AceGPT-v2-8B-Chat"
)

# output-token budget: the thinking model needs room to reason + answer. jais-13b has
# only a 2048-token context window, so it's capped to fit prompt+generation.
budget() {
  case "$1" in
    qwen3-8b) echo 4500;;   # thinking: room to reason + answer
    jais-13b) echo 1700;;   # 2048 ctx — keep prompt+generation inside the window
    *)        echo 3000;;
  esac
}

HF_HUB="${HF_HOME:-$HOME/.cache/huggingface}/hub"

# already fully generated? (500 answers, none are ERROR) -> skip re-download.
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

run_one() {
  local gpu="$1" model="$2" repo="${REPO[$2]}"
  if is_done "$model"; then
    echo "[GPU$gpu] $model already complete — skipping."
    return 0
  fi
  echo "[GPU$gpu] === $model ($repo) — max_tokens=$(budget "$model") ==="
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" inference_hugging_face.py \
      --model "$model" --input "$INPUT" --output-dir "$OUTDIR" \
      --prompt "$PROMPT" --allow-thinking --no-quantize \
      --max-tokens "$(budget "$model")" --batch-size "$BATCH" \
      >> "$LOGDIR/$model.log" 2>&1
  local rc=$?
  # free disk regardless of success/failure so the next model has room
  local cache="$HF_HUB/models--${repo//\//--}"
  echo "[GPU$gpu] $model finished (rc=$rc); removing cache $cache"
  rm -rf "$cache"
  return $rc
}

run_group() {
  local gpu="$1"; shift
  for m in "$@"; do run_one "$gpu" "$m"; done
  echo "[GPU$gpu] ===== group done ====="
}

# 5+5 split balanced for wall-clock: the two slow gemma-family HF models
# (gemma-3-12b, fanar-1-9b) go on SEPARATE GPUs; Qwen3 (thinking, slow) is paired
# opposite gemma-3-12b.
GROUP_A=(gemma-3-12b allam-7b mistral-7b phi-3.5 acegpt-8b)   # -> GPU 0
GROUP_B=(fanar-1-9b  jais-13b llama-3.1-8b command-r-7b qwen3-8b)  # -> GPU 1

echo "Input : $INPUT"
echo "Output: $OUTDIR"
echo "GPU0  : ${GROUP_A[*]}"
echo "GPU1  : ${GROUP_B[*]}"

run_group 0 "${GROUP_A[@]}" &  PID_A=$!
run_group 1 "${GROUP_B[@]}" &  PID_B=$!
wait "$PID_A"; wait "$PID_B"

echo "All models done. Answers in $OUTDIR/<model>_${PROMPT}.json ; logs in $LOGDIR/"
