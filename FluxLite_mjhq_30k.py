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
    parser.add_argument("--output_dir", type=str, default="/gpfs/bwfor/work/ws/hd_om233-flux/flux/MJHQ_30k/Flux_Lite")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=50, help="Number of steps. Default is 4 for schnell, 50 for dev")
    parser.add_argument("--guidance", type=float, default=3.5)
    parser.add_argument("--json_file", type=str, default="/gpfs/lsdf02/sd23g007/datasets/MJHQ-30k/meta_data.json")

    

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
    seed = random.randint(0, 2**32 - 1)
    
    prompt_items_list = list(prompts.items())
    print(len(prompt_items_list))
    random.shuffle(prompt_items_list)
    seed_everything(seed)
    prompt="A luxurious spacious modern coffee shop full of south koreans inside. The interior is circular with celling glass windows. there is some red roses for decoration "
    sample = pipe(
                prompt=prompt,
                generator=torch.Generator(device="cpu").manual_seed(seed),
                num_inference_steps=args.steps,
                guidance_scale=guidance_scale,
                height=1024,
                width=1024,
            ).images[0]
    sample.save(os.path.join(args.output_dir, "d6b659fd5b3c6e81496321fab6206d75ffaaa3bf_2966723581.png"))
    # with torch.inference_mode():
    #     for keys, value in prompt_items_list:
    #         prompt_name_w = keys.split(".")[0]
            
    #         search_pattern = os.path.join(args.output_dir, f"{prompt_name_w}_*.png")
    #         # 2. Suche nach Dateien, die passen
    #         existing_files = glob.glob(search_pattern)
        
    #         if len(existing_files) > 0:
    #             print("CONTINUE")
    #             continue 
    #             # Generate images
    #         sample = pipe(
    #             prompt=value["prompt"],
    #             generator=torch.Generator(device="cpu").manual_seed(seed),
    #             num_inference_steps=args.steps,
    #             guidance_scale=guidance_scale,
    #             height=1024,
    #             width=1024,
    #         ).images[0]
    #         if not os.path.exists(os.path.join(args.output_dir)):
    #             os.makedirs(os.path.join(args.output_dir))
    #         sample.save(os.path.join(args.output_dir, prompt_name_w + "_" + str(seed) + ".png"))
                    




