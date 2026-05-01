# Minimum Inference Code for FLUX

import argparse
import datetime
import json
import math
import os
import random
from typing import Callable, List, Optional, Dict, Any, Tuple
import sys
import accelerate
import einops
import numpy as np
import torch
from library import device_utils
from library.device_utils import get_preferred_device, init_ipex
from myCode.modify_model_iterative import modify_model_it, modify_model
from networks import oft_flux
from PIL import Image
from safetensors.torch import load_file
from tqdm import tqdm
from transformers import CLIPTextModel, T5EncoderModel

init_ipex()

sys.path.append("/home/hd/hd_hd/hd_om233/partially_removal/MasterThesis_Evaluation")
from evaluation_CLIP_2 import compute_clip
from evaluation_cmmd import compute_cmmd

from library.utils import setup_logging, str_to_dtype

setup_logging()
import logging

logger = logging.getLogger(__name__)

import networks.lora_flux as lora_flux
from library import flux_models, flux_utils, sd3_utils, strategy_flux



def get_sorted_block_names(single_dict, double_dict):
    combined_list = []

    # 1. Single Blocks verarbeiten
    for block_nr, score in single_dict.items():
        combined_list.append({
            "name": f"single_block_{block_nr}",
            "score": float(score)
        })

    # 2. Double Blocks verarbeiten
    for block_nr, score in double_dict.items():
        combined_list.append({
            "name": f"double_block_{block_nr}",
            "score": float(score)
        })

    # 3. Sortieren nach dem Score (kleinster CMMD zuerst)
    # Falls du absteigend sortieren willst (größter zuerst), setze reverse=True
    sorted_data = sorted(combined_list, key=lambda x: x["score"])

    # 4. Nur die Namen (Strings) in ein Array extrahieren
    result_array = [item["name"] for item in sorted_data]
    
    return result_array

if __name__ == "__main__":
    target_height = 1024  # 1024
    target_width = 1024  # 1024

    # steps = 50  # 28  # 50
    # guidance_scale = 5
    # seed = 1  # None  # 1

    device = get_preferred_device()

    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default=" A close-up portrait of an elderly man with a weathered face, showing every wrinkle and detail, against a simple, dark background, shot with a shallow depth of field.")
    parser.add_argument("--output_dir", type=str, default="/gpfs/bwfor/work/ws/hd_om233-flux/flux/image/block_analysis/")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="base dtype")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=None, help="Number of steps. Default is 4 for schnell, 50 for dev")
    parser.add_argument("--guidance", type=float, default=3.5)
    parser.add_argument("--negative_prompt", type=str, default=None)
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--max_count", type=int, default=-1)
    parser.add_argument("--offload", action="store_true", help="Offload to CPU")
    parser.add_argument(
        "--lora_weights",
        type=str,
        nargs="*",
        default=[],
        help="LoRA weights, only supports networks.lora_flux and lora_oft, each argument is a `path;multiplier` (semi-colon separated)",
    )
    parser.add_argument("--merge_lora_weights", action="store_true", help="Merge LoRA weights to model")
    parser.add_argument("--width", type=int, default=target_width)
    parser.add_argument("--height", type=int, default=target_height)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--image_name", type=str, default="img.png")
    parser.add_argument("--compress_single_blocks", nargs='+', type=int, default=[])
    parser.add_argument("--compress_double_blocks", nargs='+', type=int, default=[])
    parser.add_argument("--compression_ratio_single_blocks", nargs='+', type=float, default=[])
    parser.add_argument("--compression_ratio_double_blocks", nargs='+', type=float, default=[])
    
    parser.add_argument("--double_block_list", nargs='+', type=int, default=[])
    parser.add_argument("--single_block_list", nargs='+', type=int, default=[])
    parser.add_argument("--ref_path", type=str, default="/gpfs/bwfor/work/ws/hd_om233-flux/flux/image/original_100")
    parser.add_argument("--logdir", type=str, default="/home/hd/hd_hd/hd_om233/SVD/flux/block_analysis/result.txt")
    parser.add_argument("--prompt_path", type=str, default="/home/hd/hd_hd/hd_om233/partially_removal/100_prompts_laion.json")
    parser.add_argument("--max_blocks_to_remove", type=int, default=15)
    args = parser.parse_args()
    
    with open(args.prompt_path, "r") as file:
        data = json.load(file)
        
    prompts = []
    for d in data.values():
        prompts.append(d)
    
        
    single_blocks_to_evaluate = args.single_block_list.copy()
    double_blocks_to_evaluate = args.double_block_list.copy()
    cmmd_single = {}
    cmmd_double = {}


    MAX_BLOCKS_TO_REMOVE = args.max_blocks_to_remove
    iteration_idx = 0

    print(f"\n--- STARTE OPTIMIERUNGS-ITERATION {iteration_idx} von {MAX_BLOCKS_TO_REMOVE} ---")

 
    # --- 1. Evaluierung der potenziellen Single Blocks ---
    print(args.single_block_list)
    PARAMS_SINGLE = 140
    RATIO = 0.6
    
    for s_block in args.single_block_list:
        # Berechne Metriken
        print("Single Block: ", s_block)
    
        # ANNAHME: compute_cmmd und compute_clip sind implementiert und geben den Metrikwert zurück.
        output_dir_block = os.path.join(args.output_dir, f"temp_iter1_single_{s_block}")
        if s_block in args.compress_single_blocks:
            idx = args.compress_single_blocks.index(s_block)
            removed_params = PARAMS_SINGLE * (1-args.compression_ratio_single_blocks[idx])*(RATIO)
        else: 
            removed_params = PARAMS_SINGLE * RATIO
        print(output_dir_block)
        cmmd_single[s_block] = compute_cmmd(args.ref_path, output_dir_block, max_count=args.max_count) / removed_params
        
          

        
        # Löschen der temporären Bilder nach Metrikberechnung (optional, aber empfohlen)
        # import shutil; shutil.rmtree(output_dir_block) 

        # --- 2. Evaluierung der potenziellen Double Blocks ---
    PARAMS_DOUBLE = 340
    RATIO = 0.6
    for d_block in args.double_block_list:
        print("Double Block: ", d_block)
        if d_block in args.compress_double_blocks:
            idx = args.compress_double_blocks.index(d_block)
            removed_params = PARAMS_DOUBLE * (1-args.compression_ratio_double_blocks[idx])*(RATIO)
        else: 
            removed_params = PARAMS_DOUBLE * RATIO
        output_dir_block = os.path.join(args.output_dir, f"temp_iter1_double_{d_block}")
        cmmd_double[d_block] = compute_cmmd(args.ref_path,output_dir_block, max_count=args.max_count) / removed_params

    sorted_dict = get_sorted_block_names(cmmd_single, cmmd_double)
    # --- 3. Besten Block bestimmen und Listen aktualisieren ---
    if args and hasattr(args, 'logdir'):
        try:
            with open(args.logdir, "a") as f: 
                f.write(f"\n=======================================================\n")
                f.write(f"ZWISCHENSTÄNDE FÜR ITERATION {iteration_idx} VOR BLOCKAUSWAHL\n")
                f.write(f"=======================================================\n")
                
                # Protokollierung der Single Block Metriken
                if cmmd_single:
                    f.write("\n### Single Block Scores ###\n")
                    # Die Keys in cmmd_single/clip_single sind die Blöcke, die *evaluiert* wurden.
                    # Wir verwenden die Evaluierungsliste für eine klare Darstellung.
                    for block in single_blocks_to_evaluate:
                        cmmd_val = cmmd_single.get(block, 'FEHLER')
                        f.write(f"  Single Block {block}: CMMD = {cmmd_val}\n")
                    f.write(f"Single Blocks: {cmmd_single} \n")
                # Protokollierung der Double Block Metriken
                if cmmd_double:
                    f.write("\n### Double Block Scores ###\n")
                    for block in double_blocks_to_evaluate:
                        cmmd_val = cmmd_double.get(block, 'FEHLER')
                        f.write(f"  Double Block {block}: CMMD = {cmmd_val}\n")
                    f.write(f"Double Blocks: {cmmd_double} \n")
                
                f.write(f"\n-------------------------------------------------------\n")
                f.write(f"Sorted Blocks: {sorted_dict} \n")
        except Exception as e:
            logger.error(f"Fehler beim Speichern der Zwischenstände in {args.logdir}: {e}")
        # Rufe die Logik zur Bestimmung des besten Blocks auf
       

    print(f"\n--- OPTIMIERUNG ABGESCHLOSSEN ---")
    print(f"Gesamt entfernte Blöcke: {iteration_idx}")
