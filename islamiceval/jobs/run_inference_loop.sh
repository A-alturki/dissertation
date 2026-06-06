#!/usr/bin/env bash
# Stage 2 — sequential vLLM inference over the islamic-eval model roster.
#
# For the BAREBONES Linux GPU box (NOT SLURM). Runs each model one-at-a-time,
# auto-picking whichever GPU currently has the most free memory, and writes one
# answers file per model named  <model>_<inputstem>.json.
#
# Usage (from anywhere; paths are resolved relative to this script):
#   bash jobs/run_inference_loop.sh
#   bash jobs/run_inference_loop.sh ../data/classified/dummy_rag_questions.json
#   bash jobs/run_inference_loop.sh ../data/classified/prompts_10.json "qwen3-8b mistral-7b"
#
# Args (all optional):
#   $1  INPUT  — prompts JSON                 (default: prompts_10.json)
#   $2  MODELS — space-separated model keys   (default: the 10 islamic-eval models)
#
# Notes:
#   * Requires `hf auth login` for gated models (gemma-3-12b, llama-3.1-8b).
#   * vLLM grabs ~90% of the chosen GPU (hardcoded in inference_vllm.py). If the
#     freest GPU is still partly occupied you may OOM — pick a quieter time or
#     lower gpu_memory_utilization in the script.

set -u

# resolve repo paths from this script's location (jobs/ -> repo root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_DIR="$SCRIPT_DIR/../scripts"
OUTDIR="$SCRIPT_DIR/../outputs/answers"

INPUT="${1:-$SCRIPT_DIR/../data/classified/prompts_10.json}"
MODELS="${2:-allam-7b jais-13b acegpt-8b silma-9b fanar-1-9b qwen3-8b gemma-3-12b mistral-7b deepseek-r1-llama-8b llama-3.1-8b}"

STEM="$(basename "$INPUT" .json)"
mkdir -p "$OUTDIR"
cd "$SCRIPTS_DIR"

# index of the GPU with the most free memory right now
pick_gpu() {
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | sort -t, -k2 -nr | head -n1 | awk -F, '{gsub(/ /,"",$1); print $1}'
}

failed=""
total=0; ok=0
for m in $MODELS; do
  total=$((total + 1))
  GPU="$(pick_gpu)"
  echo ""
  echo "============================================================"
  echo "[$total] $m  ->  GPU $GPU   ($(date +%H:%M:%S))"
  echo "============================================================"
  if CUDA_VISIBLE_DEVICES="$GPU" python inference_vllm.py --model "$m" --input "$INPUT"; then
    mv -f "$OUTDIR/$m.json" "$OUTDIR/${m}_${STEM}.json"
    echo "  saved -> outputs/answers/${m}_${STEM}.json"
    ok=$((ok + 1))
  else
    echo "  [WARN] $m failed — continuing with the rest."
    failed="$failed $m"
  fi
done

echo ""
echo "Done. $ok/$total succeeded.${failed:+  Failed:$failed}"
