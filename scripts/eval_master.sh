#!/bin/bash
#SBATCH --job-name=eval_master
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:1  
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00

#all models we want to evaluate
vals=(1e-5) #1e-4 5e-5 1e-5 5e-6 1e-6

# Loop over each value
for v in "${vals[@]}"; do
    echo "Running with learning rate: $v"
    sbatch /home/catheri4/utils/eval/eval.sh \
    "pretrained=/data/user_data/catheri4/models/olmo_1b_b25_${v}" \
    "/home/catheri4/outputs/b25_${v}"  
done
