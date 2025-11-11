# Minimum Inference Code for FLUX

import argparse
import json
import os

import torch
from diffusers import FluxPipeline
from library.device_utils import get_preferred_device
from pytorch_lightning import seed_everything
from tqdm import tqdm, trange

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
   
    
    # ------------------- geneval parameters -----------------
    parser.add_argument(
        "--metadata_file",
        type=str,
        help="JSONL file containing lines of metadata for each prompt"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=4,
        help="number of samples",
    )
    args = parser.parse_args()
  

    with open(args.metadata_file) as fp:
        metadata = [json.loads(line) for line in fp]

    seed = args.seed
    steps = args.steps
    guidance_scale = args.guidance

    
    

    base_model_id = "Freepik/flux.1-lite-8B-alpha"
    torch_dtype = torch.bfloat16
    # Load the pipe
    model_id = "Freepik/flux.1-lite-8B-alpha"
    pipe = FluxPipeline.from_pretrained(
        model_id, torch_dtype=torch_dtype
    ).to(device)

    for index, metadata in enumerate(metadata):
        seed_everything(args.seed)
        outpath = os.path.join(args.output_dir, f"{index:0>5}")
        os.makedirs(outpath, exist_ok=True)
        prompt = metadata['prompt']
        
        sample_path = os.path.join(outpath, "samples")
        os.makedirs(sample_path, exist_ok=True)
        with open(os.path.join(outpath, "metadata.jsonl"), "w") as fp:
            json.dump(metadata, fp)
        
        sample_count = 0
        
        with torch.no_grad():
            all_samples = list()
            for n in trange(args.n_samples, desc="Sampling"):
                sample = pipe(
                    prompt=prompt,
                    generator=torch.Generator(device="cpu").manual_seed(args.seed+n),
                    num_inference_steps=args.steps,
                    guidance_scale=guidance_scale,
                    height=1024,
                    width=1024,
                ).images[0]
                sample.save(os.path.join(sample_path, f"{sample_count:05}.png"))
                sample_count += 1
               
