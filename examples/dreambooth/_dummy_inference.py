import torch
from utils.two_timestep_inference import FluxPipelineTwoTimestep as FluxPipeline
from utils.dual_time_embedder import add_dual_time_embedder

# Load the base FLUX.1-dev model
model_id = "black-forest-labs/FLUX.1-dev"
pipeline = FluxPipeline.from_pretrained(
    model_id, 
    torch_dtype=torch.bfloat16,
    cache_dir="/pscratch/sd/g/gabeguo/cache/huggingface"
).to("cuda")

# Load your trained LoRA weights
lora_path = "/pscratch/sd/g/gabeguo/diffusers/examples/dreambooth/sanity-check-dual-embedding"

pipeline.transformer = add_dual_time_embedder(pipeline.transformer)

pipeline.load_lora_weights(lora_path)

# Generate image with your custom prompt
prompt = "A unicorn eating a sandwich."
image = pipeline(
    prompt=prompt,
    num_inference_steps=128,
    guidance_scale=3.5,
    width=256,
    height=256,
    generator=torch.Generator("cuda").manual_seed(42)
).images[0]

# Save the result
image.save("unicorn.png")
print("Image saved as unicorn.png")