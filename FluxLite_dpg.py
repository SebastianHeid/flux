# Minimum Inference Code for FLUX

import argparse
import os
import json
import random
import glob
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
    parser.add_argument("--output_dir", type=str, default="/gpfs/bwfor/work/ws/hd_om233-flux/flux/DPG/Flux_Lite")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=50, help="Number of steps. Default is 4 for schnell, 50 for dev")
    parser.add_argument("--guidance", type=float, default=3.5)
    parser.add_argument("--json_file", type=str, default="/home/hd/hd_hd/hd_om233/ModelEvaluationBenchmarks/DPG_Bench/prompts.json")

    

    args = parser.parse_args()
  
    base_model_id = "Freepik/flux.1-lite-8B-alpha"
    torch_dtype = torch.bfloat16
    # Load the pipe
    model_id = "Freepik/flux.1-lite-8B-alpha"
    pipe = FluxPipeline.from_pretrained(
        model_id, torch_dtype=torch_dtype
    ).to(device)

    with open(args.json_file, "r") as file:
        prompts = json.load(file)  

    
    steps = args.steps
    guidance_scale = args.guidance
  
    
    prompt_items_list = list(prompts.items())


    with torch.inference_mode():
        for keys, value in prompt_items_list:
            seed_everything(args.seed)

            base_name = os.path.splitext(keys)[0]
            output_filename = f"{base_name}.png"
            if not os.path.exists(os.path.join(args.output_dir)):
                os.makedirs(os.path.join(args.output_dir))
            save_path = os.path.join(args.output_dir, output_filename)
            if os.path.exists(save_path):
                print(f"Skipping {output_filename}, already exists in {args.output_dir}")
                continue
        
            all_samples = []
    
        
            with torch.no_grad():
                for n in range(4):
                
                    # Seed variieren, damit die 4 Bilder unterschiedlich sind
                    current_seed = args.seed + n if args.seed is not None else None
                    sample = pipe(
                        prompt=value,
                        generator=torch.Generator(device="cpu").manual_seed(current_seed),
                        num_inference_steps=args.steps,
                        guidance_scale=guidance_scale,
                        height=1024,
                        width=1024,
                        output_type="pt",
                    ).images[0]
                    # Zur Liste hinzufügen (statt sofort speichern)
                    all_samples.append(sample)
            # 2. Grid erstellen und speichern
            if len(all_samples) > 0:
                # Liste von Tensoren zu einem Batch zusammenfügen: [4, C, H, W]
                grid_tensor = torch.stack(all_samples, dim=0)
                
            # nrow=2 erzeugt bei 4 Bildern ein 2x2 Grid
            save_image(
                grid_tensor, 
                save_path, 
                nrow=2
            )
            
 
                    




