#!/bin/bash
#SBATCH --job-name=tulu_split_perplexity
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1 
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --mem=32G



source ~/miniconda3/etc/profile.d/conda.sh
conda activate myenv310
cd /home/catheri4/experiments/tulu_partition/tulu_split_perplexity

# Run split
python tulu_perplexity_split.py \
 --dataset_path /data/user_data/catheri4/datasets/tulu_40000_items \
 --output_dir /data/user_data/catheri4/datasets/tulu_quartiles

