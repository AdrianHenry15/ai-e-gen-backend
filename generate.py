import torch
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler

# Load Stable Diffusion
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.to("cuda", torch.float16)  # Force torch.float16

# Optimize GPU memory usage (Since xFormers isn't supported)
pipe.unet.to(memory_format=torch.channels_last)
pipe.vae.to(memory_format=torch.channels_last)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# Use a different scheduler for better quality
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

# High-quality image generation settings
prompt = "A futuristic cyberpunk city at night, neon lights, highly detailed"
image = pipe(prompt, height=1024, width=1024, num_inference_steps=100, guidance_scale=12.5).images[0]

# Save image
image.save("high_quality_image.png")
print("Image saved as 'high_quality_image.png'")
