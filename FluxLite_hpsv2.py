# Minimum Inference Code for FLUX

import argparse
import os

import hpsv2
import torch
from diffusers import FluxPipeline
from library import device_utils
from library.device_utils import get_preferred_device
from pytorch_lightning import seed_everything
from torchvision.utils import save_image

if __name__ == "__main__":
    target_height = 1024  # 1024
    target_width = 1024  # 1024

    # steps = 50  # 28  # 50
    # guidance_scale = 5
    # seed = 1  # None  # 1

    device = get_preferred_device()

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="/home/hd/hd_hd/hd_om233/flux/image/FastFlux/43_it")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=50, help="Number of steps. Default is 4 for schnell, 50 for dev")
    parser.add_argument("--guidance", type=float, default=3.5)

    

    args = parser.parse_args()
  
    base_model_id = "Freepik/flux.1-lite-8B-alpha"
    torch_dtype = torch.bfloat16
    # Load the pipe
    model_id = "Freepik/flux.1-lite-8B-alpha"
    pipe = FluxPipeline.from_pretrained(
        model_id, torch_dtype=torch_dtype
    ).to(device)

    all_prompts =  hpsv2.benchmark_prompts('all')  

    seed = args.seed
    steps = args.steps
    guidance_scale = args.guidance

    for style, prompts in all_prompts.items():
        seed_everything(args.seed)
        
        with torch.inference_mode():
            for idx, prompt in enumerate(prompts):
                # Generate images
                sample = pipe(
                    prompt=prompt,
                    generator=torch.Generator(device="cpu").manual_seed(seed),
                    num_inference_steps=args.steps,
                    guidance_scale=guidance_scale,
                    height=1024,
                    width=1024,
                ).images[0]
                if not os.path.exists(os.path.join(args.output_dir, style)):
                    os.makedirs(os.path.join(args.output_dir, style))
                sample.save(os.path.join(args.output_dir, style, f"{idx:05d}.jpg"))
               




