#!/usr/bin/env bash
# ==============================================================================
# run_vanilla_large_box.sh — vanilla-prompt generation for the 8 LARGER models
# on the 500-question batch, through the SAME pipeline (inference_hugging_face.py)
# as the original 10. Sequential, one model fully loaded/unloaded at a time; each
# model shards across GPUs 0+1 via device_map="auto".
#
# Per model:  nvidia-smi room check -> print first rendered prompt -> 3-prompt fp16
#             SMOKE (skip+log if degenerate) -> full 500-run (+sidecar metadata)
#             -> delete HF cache. Failures are logged and skipped, never block the queue.
#
# Constraints baked in (Turing sm_75): HF Transformers, attn=sdpa, dtype=native
# (bf16-native models stay bf16 — user-approved), CUDA_VISIBLE_DEVICES=0,1 only
# (GPU2 busy, GPU3 unreliable), seed 42, creator-default generation configs,
# thinking kept on where native. Output -> the SAME outputs/answers/vanilla_500/.
#
# Prereqs: huggingface-cli login (gemma-4, jais-30b are gated) or HF_TOKEN in ../.env.
# Usage:   tmux new -s biginfer ; bash run_vanilla_large_box.sh
# Resume:  re-run — finished models (500 non-error answers) are skipped.
# ==============================================================================
set -u
cd "$(dirname "$0")"                        # -> scripts/
PY="${PYTHON:-python3}"

INPUT="../data/islamiceval2026_RELEASE/batch1_500_questions.json"
OUTDIR="../outputs/answers/vanilla_500"
LOGDIR="$OUTDIR/logs"
PROMPT="vanilla"
BATCH=8                                     # big models -> smaller batch (OOM-retry still halves)
MIN_FREE_MB=40000                           # require ~40GB free on EACH of GPU 0 and 1
export CUDA_VISIBLE_DEVICES=0,1
mkdir -p "$OUTDIR" "$LOGDIR"

# pull HF_TOKEN from ../.env if present (gated: gemma-4-31b, jais-30b)
if [ -f ../.env ]; then
  t=$(grep -E '^\s*HF_TOKEN\s*=' ../.env | head -1 | sed -E 's/^[^=]*=\s*"?([^"]+)"?.*/\1/')
  [ -n "${t:-}" ] && export HF_TOKEN="$t" HUGGING_FACE_HUB_TOKEN="$t"
fi

# run order: smallest -> largest (failures surface cheap). model -> HF repo.
MODELS=(ministral-3-14b fanar-2-27b gemma-4-31b qwen3.5-27b karnak-40b falcon-h1-34b acegpt-32b jais-30b)
declare -A REPO=(
  [ministral-3-14b]="mistralai/Ministral-3-14B-Instruct-2512-BF16"
  [fanar-2-27b]="QCRI/Fanar-2-27B-Instruct"
  [gemma-4-31b]="google/gemma-4-31B-it"
  [qwen3.5-27b]="Qwen/Qwen3.5-27B"
  [karnak-40b]="Applied-Innovation-Center/Karnak-40B-v1.0"
  [falcon-h1-34b]="tiiuae/Falcon-H1-34B-Instruct"
  [acegpt-32b]="FreedomIntelligence/AceGPT-v2-32B-Chat"
  [jais-30b]="inceptionai/jais-family-30b-16k-chat"
)
# output-token budget: thinking-capable models get more room.
budget() { case "$1" in qwen3.5-27b|karnak-40b|gemma-4-31b) echo 4500;; *) echo 3000;; esac; }
HF_HUB="${HF_HOME:-$HOME/.cache/huggingface}/hub"

is_done() {  # 500 non-error answers already?
  "$PY" - "$OUTDIR/$1_${PROMPT}.json" <<'PY'
import json,sys
try:
    d=json.load(open(sys.argv[1],encoding="utf-8"))
    ok=[r for r in d if not str(r.get("answer","")).startswith("ERROR")]
    sys.exit(0 if len(ok)>=500 else 1)
except Exception: sys.exit(1)
PY
}

gpus_have_room() {  # GPU 0 and 1 each >= MIN_FREE_MB free?
  local f0 f1
  f0=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0 2>/dev/null | tr -d ' ')
  f1=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 1 2>/dev/null | tr -d ' ')
  echo "    GPU0 free=${f0:-?}MiB  GPU1 free=${f1:-?}MiB  (need >=${MIN_FREE_MB} each)"
  [ -n "${f0:-}" ] && [ -n "${f1:-}" ] && [ "$f0" -ge "$MIN_FREE_MB" ] && [ "$f1" -ge "$MIN_FREE_MB" ]
}

COMMON=(--input "$INPUT" --output-dir "$OUTDIR" --prompt "$PROMPT" --allow-thinking
        --no-quantize --dtype native --attn sdpa --seed 42)

for model in "${MODELS[@]}"; do
  repo="${REPO[$model]}"; log="$LOGDIR/$model.log"; mt="$(budget "$model")"
  echo "============================================================"
  echo "[$model] repo=$repo  max_tokens=$mt"
  if is_done "$model"; then echo "  already complete — skipping."; continue; fi
  if ! gpus_have_room; then echo "  [SKIP] GPUs 0+1 don't have room — skipping (freed later? re-run)."; continue; fi

  # 1) first rendered prompt (leak check) + 3-prompt smoke, logged
  echo "  smoke (3 prompts, fp16 degeneracy check)…"
  "$PY" inference_hugging_face.py --model "$model" "${COMMON[@]}" \
        --max-tokens "$mt" --print-first-prompt --smoke 3 > "$log" 2>&1
  rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "  [SKIP] smoke failed (rc=$rc: load error or degenerate fp16) — see $log"
    grep -E "SMOKE|Error|Traceback|attn|Cannot|Unrecognized" "$log" | tail -5
    rm -rf "$HF_HUB/models--${repo//\//--}"        # free disk; move on
    continue
  fi
  echo "  smoke PASS — full 500-run…"

  # 2) full run with metadata sidecar
  "$PY" inference_hugging_face.py --model "$model" "${COMMON[@]}" \
        --max-tokens "$mt" --batch-size "$BATCH" --emit-meta >> "$log" 2>&1
  rc=$?
  echo "  full run rc=$rc; deleting cache."
  rm -rf "$HF_HUB/models--${repo//\//--}"
  tail -2 "$log"
done
echo "============================================================"
echo "Done. Answers + <model>_${PROMPT}.meta.json + config_table.tsv in $OUTDIR ; logs in $LOGDIR/"
