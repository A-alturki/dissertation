#!/bin/bash
# Submit one vLLM test job per model family to verify the inference script works.
# Uses prompts_10.json (10 prompts) for a fast smoke test.
#
# Usage:
#   bash submit_test.sh              # submit all test models
#   bash submit_test.sh allam-7b     # submit one specific model

TEST_MODELS=(
    "gemma-3-4b"      # multimodal model — tests limit_mm_per_prompt fix
    "qwen3-4b"        # standard ChatML template
    "llama-3.2-3b"    # Llama template
    "jais-13b"        # no system role — tests NO_SYSTEM_ROLE fallback
    "mistral-7b"      # Mistral template
    "allam-7b"        # Arabic-centric
)

INPUT="/disk/scratch/s2870640/islamiceval/data/prompts_10.json"
SCRIPT="$(dirname "$0")/run_inference_vllm.sh"

if [ -n "$1" ]; then
    # Single model passed as argument
    echo "Submitting single model: $1"
    sbatch "$SCRIPT" "$1" "$INPUT"
else
    # Submit all test models
    for MODEL in "${TEST_MODELS[@]}"; do
        JOB=$(sbatch --parsable "$SCRIPT" "$MODEL" "$INPUT")
        echo "Submitted $MODEL -> job $JOB"
    done
    echo ""
    echo "Monitor with:  squeue -u $USER"
    echo "Logs in:       ~/dissertation/islamiceval/logs/"
fi
