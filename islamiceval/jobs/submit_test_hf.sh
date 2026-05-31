#!/bin/bash
# Submit one HF inference test job per model family to verify the script works.
# Uses prompts_10.json (10 prompts) for a fast smoke test.
#
# Usage:
#   bash submit_test_hf.sh              # submit all test models
#   bash submit_test_hf.sh allam-7b     # submit one specific model

TEST_MODELS=(
    "gemma-3-4b"      # multimodal — tests apply_chat_template
    "qwen3-8b"        # standard ChatML
    "llama-3.1-8b"    # Llama template — gated, needs HF token
    "jais-13b"        # no system role fallback — gated, needs HF token
    "mistral-7b"      # Mistral template
    "allam-7b"        # Arabic-centric
    "deepseek-r1-llama-8b"   # test thinking mode disable
)

INPUT="/disk/scratch/s2870640/islamiceval/data/prompts_10.json"
SCRIPT="$(dirname "$0")/run_inference_hf.sh"

if [ -n "$1" ]; then
    echo "Submitting single model: $1"
    sbatch "$SCRIPT" "$1" "$INPUT"
else
    for MODEL in "${TEST_MODELS[@]}"; do
        JOB=$(sbatch --parsable "$SCRIPT" "$MODEL" "$INPUT")
        echo "Submitted $MODEL -> job $JOB"
    done
    echo ""
    echo "Monitor with:  squeue -u $USER"
    echo "Logs in:       ~/dissertation/islamiceval/logs/"
fi
