#!/bin/bash
#SBATCH --job-name=batch_submission
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_albergo_lab
#SBATCH --nodes=1               # Number of nodes (edit this: 1 for single-node, 2+ for multi-node)
#SBATCH --ntasks-per-node=4     # Tasks per node (typically matches GPUs per node)
#SBATCH --cpus-per-task=16      # CPU cores per task
#SBATCH --mem=512GB             # Memory per node
#SBATCH --gres=gpu:4            # GPUs per node
#SBATCH --time=48:00:00         # Time limit (adjust as needed)
#SBATCH --output=/n/netscratch/albergo_lab/Everyone/gabeguo/projects/flux-lora/results/lsd/12-05-25/slurm-%j_rank%t.out
#SBATCH --error=/n/netscratch/albergo_lab/Everyone/gabeguo/projects/flux-lora/results/lsd/12-05-25/slurm-%j_rank%t.err
#SBATCH --constraint=h100

export MODEL_NAME="black-forest-labs/FLUX.1-dev"
export INSTANCE_DIR="does_not_exist"
export OUTPUT_DIR="/n/netscratch/albergo_lab/Everyone/gabeguo/projects/flux-lora/results/lsd/12-05-25"
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
  --learning_rate=1e-5 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=10000 \
  --seed="0" \
  --cache_dir="/pscratch/sd/g/gabeguo/cache/huggingface" \
  --streaming \
  --rank=128 \
  --gradient_checkpointing