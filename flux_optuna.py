# Minimum Inference Code for FLUX

import argparse
import datetime
import json
import math
import os
import random
from typing import Callable, List, Optional
import optuna
import accelerate
import einops
import numpy as np
import torch
from library import device_utils
from library.device_utils import get_preferred_device, init_ipex
from myCode.modify_model import modify_model
from networks import oft_flux
from PIL import Image
from safetensors.torch import load_file
from tqdm import tqdm
from transformers import CLIPTextModel, T5EncoderModel
from Optuna_helper.objectives import objective_fixed_no_double_blocks_fixed_ratio
init_ipex()
from functools import partial

from library.utils import setup_logging, str_to_dtype

setup_logging()
import logging

logger = logging.getLogger(__name__)

import networks.lora_flux as lora_flux
from library import flux_models, flux_utils, sd3_utils, strategy_flux
from library.utils import setup_logging, str_to_dtype


def main(clip_l,
        t5xxl,
        ae,
        args,
        device,
        tokenize_strategy,
        encoding_strategy,
        accelerator):
    
    context = {
        "clip_l": clip_l,
        "t5xxl": t5xxl,
        "ae": ae,
        "args": args,
        "device": device,
        "logger": logger,
        "tokenize_strategy": tokenize_strategy,
        "encoding_strategy": encoding_strategy,
        "accelerator": accelerator,
    }
    
    
    objective_with_context = partial(objective_fixed_no_double_blocks_fixed_ratio, context=context)
    
    study = optuna.create_study(direction="minimize")
    if args.use_prior:
        greedy_blocks_to_prune = args.greedy_double_blocks_to_prune
        greedy_n_blocks = args.greedy_n_blocks

        # 2. Übersetzen Sie das Greedy-Ergebnis in das Format,
        #    das Ihre 'objective'-Funktion erwartet.
        #    Wir simulieren die "block_score"-Parameter:
        #    Die 11 Greedy-Blöcke erhalten einen hohen Score (1.0), der Rest einen niedrigen (0.0).
        
        params_from_greedy = { }
        
        # Initialisiere alle Scores mit 0.0
        for i in range(19):
            params_from_greedy[f"block_score_{i}"] = 0.0
            
        # Setze die Scores der 11 "guten" Blöcke auf 1.0
        for block_idx in greedy_blocks_to_prune:
            params_from_greedy[f"block_score_{block_idx}"] = 1.0
            
        
        # 3. Fügen Sie diesen "Prior" (Ihren Greedy-Lauf) der Warteschlange hinzu.
        #    Optuna wird DIESEN Trial als Allererstes ausführen (Trial 0).
        try:
            study.enqueue_trial(params_from_greedy)
            print("Erfolgreich 'Warm-Start'-Trial (Greedy-Lösung) zur Warteschlange hinzugefügt.")
            print(f"Parameter: n_blocks={greedy_n_blocks}, blocks={greedy_blocks_to_prune}")
        except ValueError as e:
            print(f"Warnung: Konnte 'Warm-Start'-Trial nicht hinzufügen (möglicherweise existiert er bereits): {e}")
    
    study.optimize(objective_with_context,  n_trials=args.n_trials,) 

    print("Best trial:", study.best_trial)
    
    best_keep_ratios = {k.replace("_keep_ratio", ""): v for k, v in study.best_params.items()}
    
    
if __name__ == "__main__":
    target_height = 1024  # 1024
    target_width = 1024  # 1024



    device = get_preferred_device()

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, default="/gpfs/bwfor/work/ws/hd_om233-flux/model_flux/flux/flux1-dev.safetensors")
    parser.add_argument("--ckpt_path_org", type=str, default="/gpfs/bwfor/work/ws/hd_om233-flux/model_flux/flux/flux1-dev.safetensors")
    parser.add_argument("--clip_l", type=str, default="/gpfs/bwfor/work/ws/hd_om233-flux/model_flux/clip/model.safetensors")
    parser.add_argument("--t5xxl", type=str, default="/gpfs/bwfor/work/ws/hd_om233-flux/model_flux/t5xxl/model.safetensors")
    parser.add_argument("--ae", type=str, default="/gpfs/bwfor/work/ws/hd_om233-flux/model_flux/ae/ae.safetensors")
    parser.add_argument("--apply_t5_attn_mask", action="store_true")
    parser.add_argument("--save_path", type=str, default="/gpfs/bwfor/work/ws/hd_om233-flux/flux/Optuna/test/")
    parser.add_argument("--ref_path", type=str, default="/gpfs/bwfor/work/ws/hd_om233-flux/flux/image/block_investigation_compress_blocks/original")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="base dtype")
    parser.add_argument("--clip_l_dtype", type=str, default=None, help="dtype for clip_l")
    parser.add_argument("--ae_dtype", type=str, default=None, help="dtype for ae")
    parser.add_argument("--t5xxl_dtype", type=str, default=None, help="dtype for t5xxl")
    parser.add_argument("--flux_dtype", type=str, default=None, help="dtype for flux")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=None, help="Number of steps. Default is 4 for schnell, 50 for dev")
    parser.add_argument("--guidance", type=float, default=3.5)
    parser.add_argument("--negative_prompt", type=str, default=None)
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--offload", action="store_true", help="Offload to CPU")
    parser.add_argument("--width", type=int, default=target_width)
    parser.add_argument("--height", type=int, default=target_height)
    parser.add_argument("--interactive", action="store_true")
    
    parser.add_argument("--single_blocks", nargs='+', type=int, default=[])
    parser.add_argument("--double_blocks", nargs='+', type=int, default=[])
    parser.add_argument("--single_blocks_compress", nargs='+', type=int, default=[])
    parser.add_argument("--double_blocks_compress", nargs='+', type=int, default=[])
    parser.add_argument("--image_name", type=str, default="img.png")
    parser.add_argument("--single_flag_attn", action="store_true", help="Flag")
    parser.add_argument("--single_flag_mlp", action="store_true", help="Flag")
    parser.add_argument("--single_flag_mlp2", action="store_true", help="Flag")
    parser.add_argument("--single_flag_mod", action="store_true", help="Flag")
    parser.add_argument("--single_rank_mod", type=int, default=256, help="rank")
    parser.add_argument("--single_rank_mlp2", type=int, default=512, help="rank")
    parser.add_argument("--single_rank_attn", type=int, default=512, help="rank")
    parser.add_argument("--single_rank_mlp", type=int, default=512, help="rank")
    
    parser.add_argument("--double_flag_img_attn", action="store_true", help="Flag")
    parser.add_argument("--double_flag_txt_attn", action="store_true", help="Flag")
    parser.add_argument("--double_flag_img_mlp", action="store_true", help="Flag")
    parser.add_argument("--double_flag_txt_mlp", action="store_true", help="Flag")
    parser.add_argument("--double_flag_img_mod", action="store_true", help="Flag")
    parser.add_argument("--double_flag_txt_mod", action="store_true", help="Flag")
    
    parser.add_argument("--double_rank_img_mod", type=int, default=256, help="rank")
    parser.add_argument("--double_rank_img_mlp", type=int, default=512, help="rank")
    parser.add_argument("--double_rank_img_attn",type=int, default=512, help="rank")
    parser.add_argument("--double_rank_txt_mod", type=int, default=256, help="rank")
    parser.add_argument("--double_rank_txt_mlp", type=int, default=512, help="rank")
    parser.add_argument("--double_rank_txt_attn", type=int, default=512, help="rank")
    
    
    parser.add_argument("--use_prior", action="store_true", help="Flag")
    parser.add_argument("--greedy_double_blocks_to_prune", nargs='+', type=int, default=[ 13, 14, 10, 12, 11, 16, 9, 15, 3, 5, 17, 6])
    parser.add_argument("--greedy_single_blocks_to_prune", nargs='+', type=int, default=[])
    parser.add_argument("--greedy_keep_ratio", type=float, default=1.0, help="rank")
    parser.add_argument("--greedy_n_blocks", type=int, default=12, help="rank")
    parser.add_argument("--n_blocks", type=int, default=12, help="rank")
    parser.add_argument("--n_trials", type=int, default=1000, help="rank")
    parser.add_argument("--global_keep_ratio", type=float, default=1.0, help="rank")

    args = parser.parse_args()
  

    seed = args.seed
    steps = args.steps
    guidance_scale = args.guidance

    def is_fp8(dt):
        return dt in [torch.float8_e4m3fn, torch.float8_e4m3fnuz, torch.float8_e5m2, torch.float8_e5m2fnuz]

    dtype = str_to_dtype(args.dtype)
    clip_l_dtype = str_to_dtype(args.clip_l_dtype, dtype)
    t5xxl_dtype = str_to_dtype(args.t5xxl_dtype, dtype)
    ae_dtype = str_to_dtype(args.ae_dtype, dtype)
    flux_dtype = str_to_dtype(args.flux_dtype, dtype)

    logger.info(f"Dtypes for clip_l, t5xxl, ae, flux: {clip_l_dtype}, {t5xxl_dtype}, {ae_dtype}, {flux_dtype}")

    loading_device = "cpu" if args.offload else device

    use_fp8 = [is_fp8(d) for d in [dtype, clip_l_dtype, t5xxl_dtype, ae_dtype, flux_dtype]]
    if any(use_fp8):
        accelerator = accelerate.Accelerator(mixed_precision="bf16")
    else:
        accelerator = None

    
    
    

    logger.info(f"Loading t5xxl from {args.t5xxl}...")
    t5xxl = flux_utils.load_t5xxl(args.t5xxl, t5xxl_dtype, loading_device)
    #t5xxl = T5EncoderModel.from_pretrained("google/t5-v1_1-xxl")
    t5xxl.eval()
    
    # load clip_l
    logger.info(f"Loading clip_l from {args.clip_l}...")
    clip_l = flux_utils.load_clip_l(args.clip_l, clip_l_dtype, loading_device)
    clip_l.eval()

    

    t5xxl_max_length = 512
    tokenize_strategy = strategy_flux.FluxTokenizeStrategy(t5xxl_max_length)
    encoding_strategy = strategy_flux.FluxTextEncodingStrategy()

    # AE
    ae = flux_utils.load_ae(args.ae, ae_dtype, loading_device)
    ae.eval()
    

    main(clip_l,
            t5xxl,
            ae,
            args,
            device,
            tokenize_strategy,
            encoding_strategy, 
            accelerator,)
