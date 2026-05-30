#!/bin/bash
# SLURM job script for Stage 2: LLM Answer Generation
# University of Edinburgh cluster (Teaching / ICF-Research partitions)
#
# Usage:
#   sbatch run_inference_vllm.sh allam-7b
#   sbatch run_inference_vllm.sh qwen3-8b ../data/classified/rag_questions.json
#   sbatch --gres=gpu:4 run_inference_vllm.sh llama-3.3-70b ../data/classified/rag_questions.json 4
#
# For A6000 nodes (48GB): sbatch -p Teaching --nodelist=landonia11 run_inference_vllm.sh qwen3-32b

#SBATCH --job-name=tueki-IE
#SBATCH --partition=Teaching
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/home/s2870640/dissertation/logs/%j_%x.out
#SBATCH --error=/home/s2870640/dissertation/logs/%j_%x.err

. /home/htang2/toolchain-20251006/toolchain.rc
source ~/venv/bin/activate

# location of data to copy from to scratch
data_path="$HOME/dissertation/islamiceval/data/classified/"
scratch_path="/disk/scratch/s2870640/islamiceval"
mkdir -p $scratch_path/data
mkdir -p $scratch_path/outputs

# copy the files to scratch from data directory
cp $data_path/* $scratch_path/data/

# configure the model, input, and tensor parallelism (if needed)
MODEL=${1:-gemma-3-4b}
# change the path depending on the input file we want to use
INPUT=${2:-$scratch_path/data/prompts_10.json}
TENSOR_PARALLEL=${3:-1}

inference_file="$HOME/dissertation/islamiceval/scripts/inference_vllm.py"

OUTPUT_DIR="$scratch_path/outputs/${MODEL}_tp${TENSOR_PARALLEL}_$(date +%Y%m%d_%H%M%S)"

echo "========================================"
echo "Job ID    : $SLURM_JOB_ID"
echo "Node      : $SLURM_NODELIST"
echo "Model     : $MODEL"
echo "Input     : $INPUT"
echo "TP size   : $TENSOR_PARALLEL"
echo "GPU       : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
echo "Started   : $(date)"
echo "========================================"

# run the inference script
python "$inference_file" \
    --model           "$MODEL" \
    --input           "$INPUT" \
    --output-dir      "$OUTPUT_DIR" \
    --max-tokens      512 \
    --tensor-parallel "$TENSOR_PARALLEL"

EXIT_CODE=$?

mkdir -p "$HOME/dissertation/islamiceval/outputs/"
cp -r "$OUTPUT_DIR"/* "$HOME/dissertation/islamiceval/outputs/"

echo "========================================"
echo "Finished : $(date)"
echo "Exit code: $EXIT_CODE"
echo "========================================"
exit $EXIT_CODE
