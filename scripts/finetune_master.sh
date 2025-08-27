#!/bin/bash
#SBATCH --job-name=finetune_3
#SBATCH --output=logs/ft_master_%j.out
#SBATCH --error=logs/ft_master_%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:1  
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00

# Define an array of small values in scientific notation
vals=(1e-5) #5e-5 1e-4 1e-6 5e-6 1e-5

# Loop over each value
for v in "${vals[@]}"; do
    echo "Running with learning rate: $v"
    sbatch --output="logs/tulu_ft_b25.out" --error="logs/tulu_ft_b25.err" /home/catheri4/utils/finetune/scripts/finetune.sh \
    "allenai/OLMo-2-0425-1B" \
    "/data/user_data/catheri4/datasets/tulu_quartiles/bottom_25.json" \
    "/data/user_data/catheri4/models/tulu_b25" \
    "$v" 
done