#!/bin/bash
#
#SBATCH -A m1266
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 12:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=none
#SBATCH --mem=224GB

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