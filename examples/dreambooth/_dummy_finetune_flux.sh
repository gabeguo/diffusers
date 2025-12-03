export MODEL_NAME="black-forest-labs/FLUX.1-dev"
export INSTANCE_DIR="does_not_exist"
export OUTPUT_DIR="sanity-check-dual-embedding"
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
  --max_train_steps=500 \
  --seed="0" \
  --cache_dir="/pscratch/sd/g/gabeguo/cache/huggingface" \
  --streaming