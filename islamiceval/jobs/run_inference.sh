#!/bin/bash
# SLURM job script for Stage 2: LLM Answer Generation
# University of Edinburgh cluster (Teaching / ICF-Research partitions)
#
# Usage:
#   sbatch run_inference.sh allam-7b
#   sbatch run_inference.sh qwen3-8b ../data/classified/rag_questions.json
#   sbatch --gres=gpu:4 run_inference.sh llama-3.3-70b ../data/classified/rag_questions.json 4
#
# For A6000 nodes (48GB): sbatch -p ICF-Research --nodelist=landonia11 run_inference.sh qwen3-32b

#SBATCH --job-name=islamiceval
#SBATCH --partition=ICF-Research
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=../logs/%j_%x.out
#SBATCH --error=../logs/%j_%x.err

MODEL=${1:-allam-7b}
INPUT=${2:-../data/classified/rag_questions.json}
TENSOR_PARALLEL=${3:-1}
OUTPUT_DIR="../outputs/answers/"

echo "========================================"
echo "Job ID    : $SLURM_JOB_ID"
echo "Node      : $SLURM_NODELIST"
echo "Model     : $MODEL"
echo "Input     : $INPUT"
echo "TP size   : $TENSOR_PARALLEL"
echo "GPU       : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
echo "Started   : $(date)"
echo "========================================"

source ~/.bashrc
conda activate islamiceval

cd "$(dirname "$0")"

python ../scripts/inference_vllm.py \
    --model          "$MODEL" \
    --input          "$INPUT" \
    --output-dir     "$OUTPUT_DIR" \
    --max-tokens     512 \
    --tensor-parallel "$TENSOR_PARALLEL"

EXIT_CODE=$?
echo "========================================"
echo "Finished : $(date)"
echo "Exit code: $EXIT_CODE"
echo "========================================"
exit $EXIT_CODE
