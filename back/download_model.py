from diffusers import StableDiffusionPipeline
import torch

# Specify the model ID and local save path
model_id = "stabilityai/stable-diffusion-2-1-base"
save_path = "./stable-diffusion-2-1-base"

# Download and save the model
pipeline = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32).to("cuda")
pipeline.save_pretrained(save_path)

print(f"Model downloaded and saved locally at {save_path}")