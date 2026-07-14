#!/usr/bin/env bash
# ==============================================================================
# run_vanilla_large.sh — vanilla-prompt generation for the 8 LARGER models on the
# 500-question batch. Same protocol as the original run (vanilla prompt, seed 42,
# thinking allowed, output -> outputs/answers/vanilla_500/, sidecar metadata), but
# routed to the RIGHT engine per model:
#
#   * 6 non-Gemma models -> vLLM (tensor-parallel over GPU 0+1 = REAL 2-GPU compute
#     + continuous batching): fast (~<1h for a 27B, vs ~22h on HF pipeline-parallel).
#   * 2 Gemma-family (fanar-2-27b = Gemma-3, gemma-4-31b) -> HF fp16: vLLM can't run
#     Gemma attention on Turing/sm_75; fp16 (user-approved) is ~2x bf16. These stay slow.
#
# Sequential, one model at a time (each uses both GPUs). Per model: nvidia-smi room
# check -> print first rendered prompt + 3-prompt SMOKE (skip+log on load-fail/degenerate)
# -> full 500-run (+meta) -> delete HF cache. Failures never block the queue. Resumable.
#
# GPUs 0+1 only (GPU2 busy, GPU3 unreliable). Prereqs: HF login / HF_TOKEN in ../.env
# (gemma-4-31b + jais-30b are gated). Usage: tmux new -s biginfer ; bash run_vanilla_large.sh
# ==============================================================================
set -u
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
INPUT="../data/islamiceval2026_RELEASE/batch1_500_questions.json"
OUTDIR="../outputs/answers/vanilla_500"
LOGDIR="$OUTDIR/logs"
PROMPT="vanilla"
MIN_FREE_MB=40000
export CUDA_VISIBLE_DEVICES=0,1
mkdir -p "$OUTDIR" "$LOGDIR"

if [ -f ../.env ]; then
  t=$(grep -E '^\s*HF_TOKEN\s*=' ../.env | head -1 | sed -E 's/^[^=]*=\s*"?([^"]+)"?.*/\1/')
  [ -n "${t:-}" ] && export HF_TOKEN="$t" HUGGING_FACE_HUB_TOKEN="$t"
fi

# order: fast vLLM models first (results early), slow Gemma HF-fp16 pair last.
ORDER=(ministral-3-14b acegpt-32b qwen3.5-27b falcon-h1-34b karnak-40b jais-30b fanar-2-27b gemma-4-31b)
declare -A ENGINE=(
  [ministral-3-14b]=vllm [acegpt-32b]=vllm [qwen3.5-27b]=vllm [falcon-h1-34b]=vllm
  [karnak-40b]=vllm [jais-30b]=vllm [fanar-2-27b]=hf [gemma-4-31b]=hf
)
declare -A REPO=(
  [ministral-3-14b]="mistralai/Ministral-3-14B-Instruct-2512-BF16"
  [acegpt-32b]="FreedomIntelligence/AceGPT-v2-32B-Chat"
  [qwen3.5-27b]="Qwen/Qwen3.5-27B"
  [falcon-h1-34b]="tiiuae/Falcon-H1-34B-Instruct"
  [karnak-40b]="Applied-Innovation-Center/Karnak-40B-v1.0"
  [jais-30b]="inceptionai/jais-family-30b-16k-chat"
  [fanar-2-27b]="QCRI/Fanar-2-27B-Instruct"
  [gemma-4-31b]="google/gemma-4-31B-it"
)
budget() {  # thinking-capable vLLM models get room; slow HF-Gemma pair is capped
  case "$1" in
    fanar-2-27b|gemma-4-31b) echo 2048;;
    qwen3.5-27b|karnak-40b)  echo 4500;;
    *)                       echo 3000;;
  esac
}
HF_HUB="${HF_HOME:-$HOME/.cache/huggingface}/hub"

is_done() {
  "$PY" - "$OUTDIR/$1_${PROMPT}.json" <<'PY'
import json,sys
try:
    d=json.load(open(sys.argv[1],encoding="utf-8"))
    ok=[r for r in d if not str(r.get("answer","")).startswith("ERROR")]
    sys.exit(0 if len(ok)>=500 else 1)
except Exception: sys.exit(1)
PY
}
gpus_have_room() {
  local f0 f1
  f0=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0 2>/dev/null | tr -d ' ')
  f1=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 1 2>/dev/null | tr -d ' ')
  echo "    GPU0 free=${f0:-?}MiB  GPU1 free=${f1:-?}MiB  (need >=${MIN_FREE_MB} each)"
  [ -n "${f0:-}" ] && [ -n "${f1:-}" ] && [ "$f0" -ge "$MIN_FREE_MB" ] && [ "$f1" -ge "$MIN_FREE_MB" ]
}

# build the per-engine command (as an array) into REPLY_CMD
build_cmd() {  # $1=model $2=engine $3=maxtok ; extra args ($4..) appended
  local m="$1" eng="$2" mt="$3"; shift 3
  if [ "$eng" = "vllm" ]; then
    # fp16 (not auto/bf16): Turing sm_75 has no fast bf16; these 6 are non-Gemma (fp16-safe).
    REPLY_CMD=("$PY" inference_vllm.py --model "$m" --input "$INPUT" --output-dir "$OUTDIR"
               --prompt "$PROMPT" --allow-thinking --seed 42 --dtype float16
               --tensor-parallel 2 --max-tokens "$mt" "$@")
  else   # hf, fp16, both GPUs via device_map=auto
    REPLY_CMD=("$PY" inference_hugging_face.py --model "$m" --input "$INPUT" --output-dir "$OUTDIR"
               --prompt "$PROMPT" --allow-thinking --no-quantize --dtype fp16 --attn sdpa
               --seed 42 --batch-size 8 --max-tokens "$mt" "$@")
  fi
}

for model in "${ORDER[@]}"; do
  eng="${ENGINE[$model]}"; repo="${REPO[$model]}"; log="$LOGDIR/$model.log"; mt="$(budget "$model")"
  echo "============================================================"
  echo "[$model] engine=$eng repo=$repo max_tokens=$mt"
  if is_done "$model"; then echo "  already complete — skipping."; continue; fi
  if ! gpus_have_room; then echo "  [SKIP] GPUs 0+1 not free enough — skipping."; continue; fi

  echo "  smoke (3 prompts)…"
  build_cmd "$model" "$eng" "$mt" --print-first-prompt --smoke 3
  "${REPLY_CMD[@]}" > "$log" 2>&1
  rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "  [SKIP] smoke failed (rc=$rc: engine can't load, or degenerate) — see $log"
    grep -E "SMOKE|Error|Traceback|Unrecognized|Unsupported|ValueError|not support" "$log" | tail -6
    rm -rf "$HF_HUB/models--${repo//\//--}"
    continue
  fi
  echo "  smoke PASS — full 500-run…"

  build_cmd "$model" "$eng" "$mt" --emit-meta
  "${REPLY_CMD[@]}" >> "$log" 2>&1
  rc=$?
  echo "  full run rc=$rc; deleting cache."
  rm -rf "$HF_HUB/models--${repo//\//--}"
  tail -2 "$log"
done
echo "============================================================"
echo "Done. Answers + *_${PROMPT}.meta.json + config_table.tsv in $OUTDIR ; logs in $LOGDIR/"
echo "vLLM models should be fast; fanar-2-27b + gemma-4-31b (HF fp16) are the slow pair."
