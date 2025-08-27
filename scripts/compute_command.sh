#!/bin/bash
#SBATCH --job-name=compute_command
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1 
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --mem=32G

ls -lh /data/user_data/catheri4

rm -rf /data/user_data/catheri4/models

ls -lh /data/user_data/catheri4

