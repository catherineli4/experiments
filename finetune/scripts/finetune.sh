#!/bin/bash
#SBATCH --job-name=finetune_%j
#SBATCH --partition=general
#SBATCH --gres=gpu:1  
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00


source ~/miniconda3/etc/profile.d/conda.sh
conda activate myenv310
# Run evaluation
python /home/catheri4/utils/finetune/finetune_tulu.py --model_name "$1" --dataset_path "$2" --output_dir "$3" --learning_rate "$4"
