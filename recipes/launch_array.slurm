#!/bin/bash
#SBATCH --job-name=test_time_compute    # Job name
#SBATCH --array=1-20%8                  # Array range (this creates 20 tasks with IDs from 1 to 20, with max 8 tasks concurrently)
#SBATCH --gres=gpu:1                    # Number of GPUs (per node)
#SBATCH --output=logs/%x-%j_%A_%a.out   # Standard output (%A is replaced by job ID, %a by task ID)
#SBATCH --error=logs/%x-%j_%A_%a.err    # Standard error
#SBATCH --time=4:00:00                  # Time limit hrs:min:sec

'''Usage:
# Best-of-N on the MATH-500 dataset

sbatch recipes/launch_array.slurm recipes/Llama-3.2-1B-Instruct/best_of_n.yaml \
    --hub_dataset_id=<YOUR_ORG>/Llama-3.2-1B-Instruct-bon-completions
'''

source ~/.bashrc
set -x -e
conda activate sal

# Define the array of input files (assuming 10 input files)
STEP=50
ENDPOINT=$((SLURM_ARRAY_TASK_COUNT * STEP - STEP))
STARTS=($(seq 0 $STEP $ENDPOINT))  # Generate sequence from 0 to ENDPOINT with a step size of 100
# Use the SLURM_ARRAY_TASK_ID to pick the correct input file
INPUT_FILE=${INPUT_FILES[$SLURM_ARRAY_TASK_ID-1]}

DATASET_START=${STARTS[$SLURM_ARRAY_TASK_ID-1]}
DATASET_END=$((${STARTS[$SLURM_ARRAY_TASK_ID-1]}+$STEP))

python scripts/test_time_compute.py "$@" \
    --dataset_start=$DATASET_START \
    --dataset_end=$DATASET_END \
    --push_to_hub