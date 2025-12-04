#!/bin/bash
#
#SBATCH --partition=m1266
#SBATCH --account=m1266
#SBATCH --job-name=flux-finetune
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --time=12:00:00
#SBATCH --nodes=1                # Single node
#SBATCH --gpus=a100:4
#SBATCH --cpus-per-task=8       # CPUs for the job
#SBATCH --ntasks=4            # Number of tasks (one per GPU)

export MODEL_NAME="black-forest-labs/FLUX.1-dev"
export INSTANCE_DIR="does_not_exist"
export OUTPUT_DIR="lsd-finetune"
export HF_HOME="/pscratch/sd/g/gabeguo/cache/huggingface"

accelerate launch train_dreambooth_lora_flux.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --mixed_precision="bf16" \
  --instance_prompt="" \
  --resolution=256 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=10000 \
  --seed="0" \
  --cache_dir="/pscratch/sd/g/gabeguo/cache/huggingface" \
  --streaming \
  --rank=64 \
  --gradient_checkpointing