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
# jais-30b -> HF: vLLM 0.22 can't build it ("'JAISConfig' object has no attribute
# 'tie_word_embeddings'"); the HF path has a working manual ALiBi template for it.
# qwen3.5-27b / falcon-h1-34b / karnak-40b -> HF (was vllm): on Turing/sm_75 (RTX 8000,
# compute 7.5) vLLM HARD-rejects bf16 ("needs >=8.0"), and these three emit degenerate
# output in fp16 -> vLLM is a dead end for them. HF does bf16 (software-emulated, correct),
# same path that made the Gemma pair work. Slow (pipeline-parallel); karnak-40b MoE may OOM.
declare -A ENGINE=(
  [ministral-3-14b]=vllm [acegpt-32b]=vllm [qwen3.5-27b]=hf [falcon-h1-34b]=hf
  [karnak-40b]=hf [jais-30b]=hf [fanar-2-27b]=hf [gemma-4-31b]=hf
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

# ---- per-model crash fixes (added 2026-07-15 after the first big run) ----
# vLLM memory/cache knobs: raise gpu-mem-util for large models that hit
# 'No available memory for the cache blocks' / OOM; lower max-num-seqs for
# Mamba/hybrid models that hit 'max_num_seqs exceeds Mamba cache blocks'.
declare -A GPU_UTIL=( [qwen3.5-27b]=0.85 [falcon-h1-34b]=0.90 [karnak-40b]=0.92 )
declare -A MAX_SEQS=( [qwen3.5-27b]=128  [falcon-h1-34b]=64   [karnak-40b]=32   )
# vLLM dtype override (vllm-branch only). NB bf16 is IMPOSSIBLE on Turing/sm_75 in vLLM
# (compute 7.5 < 8.0) -> the three ex-vllm models that needed bf16 were moved to HF above;
# nothing left here needs an override (the remaining vLLM pair is fp16-safe).
declare -A VLLM_DTYPE=()
# HF dtype = 'native' (repo torch_dtype = bf16). Needed for: Gemma-family (fp16 overflows
# the logit soft-cap -> inf/nan) AND the three ex-vllm hybrids (fp16 = degenerate).
declare -A HF_DTYPE=( [fanar-2-27b]=native [gemma-4-31b]=native
  [qwen3.5-27b]=native [falcon-h1-34b]=native [karnak-40b]=native )
# HF batch size: shrink for the big bf16 models so device_map=auto doesn't OOM on 2x48GB
# (40B MoE weights alone ~= 80GB). Default 8 for everything else.
declare -A HF_BSZ=( [qwen3.5-27b]=4 [falcon-h1-34b]=2 [karnak-40b]=1 )
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
    local extra=() vdt="${VLLM_DTYPE[$m]:-float16}"
    [ -n "${GPU_UTIL[$m]:-}" ] && extra+=(--gpu-memory-utilization "${GPU_UTIL[$m]}")
    [ -n "${MAX_SEQS[$m]:-}" ] && extra+=(--max-num-seqs "${MAX_SEQS[$m]}")
    REPLY_CMD=("$PY" inference_vllm.py --model "$m" --input "$INPUT" --output-dir "$OUTDIR"
               --prompt "$PROMPT" --allow-thinking --seed 42 --dtype "$vdt"
               --tensor-parallel 2 --max-tokens "$mt" "${extra[@]}" "$@")
  else   # hf, both GPUs via device_map=auto; Gemma-family + ex-vllm hybrids -> native(bf16)
    local dt="${HF_DTYPE[$m]:-fp16}" bsz="${HF_BSZ[$m]:-8}"
    REPLY_CMD=("$PY" inference_hugging_face.py --model "$m" --input "$INPUT" --output-dir "$OUTDIR"
               --prompt "$PROMPT" --allow-thinking --no-quantize --dtype "$dt" --attn sdpa
               --seed 42 --batch-size "$bsz" --max-tokens "$mt" "$@")
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
