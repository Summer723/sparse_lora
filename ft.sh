#!/bin/bash
#SBATCH --output=%N-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:v100l:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=10:00:00

source /home/summer23/scratch/lora_dev/bin/activate
python3 /home/summer23/scratch/sparse_lora/main.py --lora_algo adalora --epoch 100 --lora_r_lst 10 20 30 