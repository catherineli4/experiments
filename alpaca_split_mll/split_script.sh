#!/bin/bash
#SBATCH --job-name=alpaca_split_perplexity
#SBATCH --output=/home/catheri4/experiments/alpaca_partition/alpaca_split_mll/logs/%j.out
#SBATCH --error=/home/catheri4/experiments/alpaca_partition/alpaca_split_mll/logs/%j.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1 
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --mem=32G



source ~/miniconda3/etc/profile.d/conda.sh
conda activate myenv310

# Run split
python /home/catheri4/experiments/alpaca_partition/alpaca_split_mll/alpaca_split_mll.py \
 --dataset_path /data/user_data/catheri4/datasets/alpaca/alpaca_data_cleaned_train.json \
 --output_dir /data/user_data/catheri4/datasets/alpaca/mll_quartiles_train 


