import torch
from diffusers import FluxPipeline

base_model_id = "Freepik/flux.1-lite-8B-alpha"
torch_dtype = torch.bfloat16
device = "cuda"

# Load the pipe
model_id = "Freepik/flux.1-lite-8B-alpha"
pipe = FluxPipeline.from_pretrained(
    model_id, torch_dtype=torch_dtype
).to(device)

# Inference
prompts =  ["A photograph of a majestic Bengal tiger in a lush jungle, with soft sunlight filtering through the canopy, detailed fur, and sharp focus on its eyes.",
           "A bustling city street in the heart of a modern metropolis, filled with people walking on sidewalks, cars and buses in traffic, neon signs and billboards glowing, skyscrapers towering above, reflections on wet asphalt, dynamic lighting and cinematic atmosphere, photographed at street level during rush hour."
]

guidance_scale = 3.5  # Keep guidance_scale at 3.5
n_steps = 50
seed = 11

with torch.inference_mode():
    for idx, prompt in enumerate(prompts):
        image = pipe(
            prompt=prompt,
            generator=torch.Generator(device="cpu").manual_seed(seed),
            num_inference_steps=n_steps,
            guidance_scale=guidance_scale,
            height=1024,
            width=1024,
        ).images[0]
        image.save("/home/hd/hd_hd/hd_om233/SVD/flux/new_images/FluxLite/image_"+str(idx)+".png")
