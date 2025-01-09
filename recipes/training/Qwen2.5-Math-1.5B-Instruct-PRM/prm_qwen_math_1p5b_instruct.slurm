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

source .venv/bin/activate

srun --nodes=1 --ntasks=1 --export=ALL,ACCELERATE_LOG_LEVEL=info accelerate launch --config_file examples/accelerate_configs/deepspeed_zero3.yaml examples/scripts/prm.py \
    --run_name="prm/Qwen2.5-Math-1.5B-Instruct-PRM-0.2" \
    --model_name_or_path="Qwen/Qwen2.5-Math-1.5B-Instruct" \
    --dataset_name="plaguss/prm800k-trl-dedup" \
    --output_dir="prm/Qwen2.5-Math-1.5B-Instruct-PRM-0.2" \
    --report_to="wandb" \
    --learning_rate=1.0e-06 \
    --per_device_train_batch_size=16 \
    --per_device_eval_batch_size=8 \
    --do_eval \
    --eval_strategy="steps" \
    --eval_steps=50 \
    --gradient_accumulation_steps=4 \
    --logging_steps=25 \
    --num_train_epochs=1 \
    --max_steps=-1 \
    --warmup_steps=50 \
    --push_to_hub \
    --gradient_checkpointing \
    --max_length=2048 \
    --step_separator="\n\n" \
    --bf16 \
    --dataset_num_proc=8
