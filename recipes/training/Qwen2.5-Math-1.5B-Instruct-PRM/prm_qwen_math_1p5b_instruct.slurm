#!/bin/bash
#SBATCH --job-name=PRM-qwen-1.5
#SBATCH --partition=hopper-prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --output=./logs/%x-%j.out
#SBATCH --err=./logs/%x-%j.err
#SBATCH --time=02-00:00:00

set -ex

module load cuda/12.1

conda activate sal

# Call the training script with srun
srun --nodes=1 --ntasks=1 --export=ALL,ACCELERATE_LOG_LEVEL=info ./train.sh