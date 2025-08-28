#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1 
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --array=0-4
#SBATCH --mem=32G


TASK_GROUPS=("lambada" "gsm8k" "boolq,truthfulqa,mmlu" "winogrande,arc_challenge,arc_easy" "hellaswag") 

TASKS=${TASK_GROUPS[$SLURM_ARRAY_TASK_ID]}

echo "Running tasks: $TASKS"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate myenv310
cd ~/utils/eval/lm-evaluation-harness

pip install -e .

# Run evaluation
lm_eval \
    --model hf \
    --model_args "$1" \
    --tasks $TASKS \
    --device cuda:0 \
    --batch_size 4 \
    --output_path "$2"