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


def best_network_cmmd_clip(
    idx: int, 
    cmmd_single: Dict[int, float] = {}, 
    clip_single: Dict[int, float] = {}, 
    cmmd_double: Dict[int, float] = {}, 
    clip_double: Dict[int, float] = {}, 
    single_blocks: List[int] = [], 
    double_blocks: List[int] = [], 
    args: Any = None
) -> Tuple[Optional[int], str]:

    # Initialisierung der Flags (basierend auf vorhandenen Daten)
    # Ich setze die Flags hier auf True, um die Berechnung zu ermöglichen, 
    # da sie im Originalcode verwendet wurden.
    cmmd_flag = bool(cmmd_single or cmmd_double)
    clip_flag = bool(clip_single or clip_double)
    
    # Hilfsfunktion zum Zuweisen von Rängen (Positionen)
    def calculate_ranks(metric_dict: Dict[int, float], reverse_sort: bool = False) -> Dict[int, int]:
        if not metric_dict:
            return {}
        # Sortiere: CMMD: kleiner ist besser (kein reverse), CLIP: größer ist besser (reverse=True)
        # Die Rangposition beginnt bei 0
        sorted_items = sorted(metric_dict.items(), key=lambda item: item[1], reverse=reverse_sort)
        return {key: rank for rank, (key, _) in enumerate(sorted_items)}

    # --- 1. Double Blocks (CMMD und CLIP) ---
    double_total_position: Dict[int, float] = {}
    double_best_block: Optional[int] = None

    if cmmd_double and clip_double:
        # 1.1 Ränge berechnen
        double_position_cmmd = calculate_ranks(cmmd_double, reverse_sort=False) # CMMD: Kleiner ist besser
        double_position_clips = calculate_ranks(clip_double, reverse_sort=True) # CLIP: Größer ist besser
        
        # 1.2 Gesamtposition berechnen
        for key in cmmd_double.keys(): # Verwende cmmd_double als Referenz für die Schlüssel
            total_rank = 0
            if cmmd_flag and key in double_position_cmmd:
                total_rank += double_position_cmmd[key]
            if clip_flag and key in double_position_clips:
                total_rank += double_position_clips[key]
            
            double_total_position[key] = total_rank
        
        # 1.3 Besten Block finden (niedrigster Gesamtrang)
        if double_total_position:
            double_best_block = min(double_total_position, key=double_total_position.get)


    # --- 2. Single Blocks (CMMD und CLIP) ---
    single_total_position: Dict[int, float] = {}
    single_best_block: Optional[int] = None
    
    if cmmd_single and clip_single:
        # 2.1 Ränge berechnen
        single_position_cmmd = calculate_ranks(cmmd_single, reverse_sort=False) # CMMD: Kleiner ist besser
        single_position_clips = calculate_ranks(clip_single, reverse_sort=True) # CLIP: Größer ist besser
        
        # 2.2 Gesamtposition berechnen
        for key in cmmd_single.keys(): # Verwende cmmd_single als Referenz für die Schlüssel
            total_rank = 0
            if cmmd_flag and key in single_position_cmmd:
                total_rank += single_position_cmmd[key]
            if clip_flag and key in single_position_clips:
                total_rank += single_position_clips[key]
                
            single_total_position[key] = total_rank
                
        # 2.3 Besten Block finden (niedrigster Gesamtrang)
        if single_total_position:
            single_best_block = min(single_total_position, key=single_total_position.get)


    # --- 3. Besten Block auswählen (Vergleich Single vs. Double) ---
    best_block: Optional[int] = None
    block_type: str = "none"

    has_single = bool(single_best_block is not None and single_total_position)
    has_double = bool(double_best_block is not None and double_total_position)
    
    if has_single and has_double:
        min_single_rank = min(single_total_position.values())
        min_double_rank = min(double_total_position.values())
        
        # Wähle den Block mit dem niedrigsten Gesamtrang (d.h. der am wenigsten wichtig ist)
        if min_single_rank <= min_double_rank:
            best_block = single_best_block
            block_type = "single"
        else:
            best_block = double_best_block
            block_type = "double"
            
    elif has_single:
        best_block = single_best_block
        block_type = "single"
        
    elif has_double:
        best_block = double_best_block
        block_type = "double"
        
    else:
        # Kein Block gefunden
        best_block = None
        block_type = "none"
        

    # --- 4. Protokollierung (Logging) ---
    if args and hasattr(args, 'logdir'):
        try:
            with open(args.logdir, "a") as f: 
                f.write(f"--- Iteration {idx} ---\n")
                f.write(f"Best Block zum Entfernen ({block_type}): {best_block}\n\n")
                
                # Single Block Details
                if cmmd_single:
                    f.write("Single Block Metriken:\n")
                    # Verwende die Liste der Blöcke, die evaluiert wurden (für das Logging)
                    for block in single_blocks:             
                        clip_val = clip_single.get(block, 'N/A')
                        cmmd_val = cmmd_single.get(block, 'N/A')
                        total_pos = single_total_position.get(block, 'N/A')
                        f.write(f"  Block {block}: CMMD={cmmd_val}, CLIP={clip_val}, Rank Summe={total_pos}\n")
                    f.write("\n")
                    
                # Double Block Details
                if cmmd_double:
                    f.write("Double Block Metriken:\n")
                    for block in double_blocks:
                        clip_val = clip_double.get(block, 'N/A')
                        cmmd_val = cmmd_double.get(block, 'N/A')
                        total_pos = double_total_position.get(block, 'N/A')
                        f.write(f"  Block {block}: CMMD={cmmd_val}, CLIP={clip_val}, Rank Summe={total_pos}\n")
                    f.write("\n")
                
        except Exception as e:
            print(f"Fehler beim Schreiben des Logs: {e}")
    
    return best_block, block_type



def time_shift(mu: float, sigma: float, t: torch.Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_lin_function(x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15) -> Callable[[float], float]:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def get_schedule(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
) -> list[float]:
    # extra step for zero
    timesteps = torch.linspace(1, 0, num_steps + 1)

    # shifting the schedule to favor high timesteps for higher signal images
    if shift:
        # eastimate mu based on linear estimation between two points
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)

    return timesteps.tolist()


def denoise(
    model: flux_models.Flux,
    img: torch.Tensor,
    img_ids: torch.Tensor,
    txt: torch.Tensor,
    txt_ids: torch.Tensor,
    vec: torch.Tensor,
    timesteps: list[float],
    guidance: float = 4.0,
    t5_attn_mask: Optional[torch.Tensor] = None,
    neg_txt: Optional[torch.Tensor] = None,
    neg_vec: Optional[torch.Tensor] = None,
    neg_t5_attn_mask: Optional[torch.Tensor] = None,
    cfg_scale: Optional[float] = None,
):
    # this is ignored for schnell
    logger.info(f"guidance: {guidance}, cfg_scale: {cfg_scale}")
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)

    # prepare classifier free guidance
    if neg_txt is not None and neg_vec is not None:
        b_img_ids = torch.cat([img_ids, img_ids], dim=0)
        b_txt_ids = torch.cat([txt_ids, txt_ids], dim=0)
        b_txt = torch.cat([neg_txt, txt], dim=0)
        b_vec = torch.cat([neg_vec, vec], dim=0)
        if t5_attn_mask is not None and neg_t5_attn_mask is not None:
            b_t5_attn_mask = torch.cat([neg_t5_attn_mask, t5_attn_mask], dim=0)
        else:
            b_t5_attn_mask = None
    else:
        b_img_ids = img_ids
        b_txt_ids = txt_ids
        b_txt = txt
        b_vec = vec
        b_t5_attn_mask = t5_attn_mask

    for t_curr, t_prev in zip(tqdm(timesteps[:-1]), timesteps[1:]):
        t_vec = torch.full((b_img_ids.shape[0],), t_curr, dtype=img.dtype, device=img.device)

        # classifier free guidance
        if neg_txt is not None and neg_vec is not None:
            b_img = torch.cat([img, img], dim=0)
        else:
            b_img = img

        pred = model(
            img=b_img,
            img_ids=b_img_ids,
            txt=b_txt,
            txt_ids=b_txt_ids,
            y=b_vec,
            timesteps=t_vec,
            guidance=guidance_vec,
            txt_attention_mask=b_t5_attn_mask,
        )

        # classifier free guidance
        if neg_txt is not None and neg_vec is not None:
            pred_uncond, pred = torch.chunk(pred, 2, dim=0)
            pred = pred_uncond + cfg_scale * (pred - pred_uncond)

        img = img + (t_prev - t_curr) * pred

    return img


def do_sample(
    accelerator: Optional[accelerate.Accelerator],
    model: flux_models.Flux,
    img: torch.Tensor,
    img_ids: torch.Tensor,
    l_pooled: torch.Tensor,
    t5_out: torch.Tensor,
    txt_ids: torch.Tensor,
    num_steps: int,
    guidance: float,
    t5_attn_mask: Optional[torch.Tensor],
    is_schnell: bool,
    device: torch.device,
    flux_dtype: torch.dtype,
    neg_l_pooled: Optional[torch.Tensor] = None,
    neg_t5_out: Optional[torch.Tensor] = None,
    neg_t5_attn_mask: Optional[torch.Tensor] = None,
    cfg_scale: Optional[float] = None,
):
    logger.info(f"num_steps: {num_steps}")
    timesteps = get_schedule(num_steps, img.shape[1], shift=not is_schnell)

    # denoise initial noise
    if accelerator:
        with accelerator.autocast(), torch.no_grad():
            x = denoise(
                model,
                img,
                img_ids,
                t5_out,
                txt_ids,
                l_pooled,
                timesteps,
                guidance,
                t5_attn_mask,
                neg_t5_out,
                neg_l_pooled,
                neg_t5_attn_mask,
                cfg_scale,
            )
    else:
        with torch.autocast(device_type=device.type, dtype=flux_dtype), torch.no_grad():
            x = denoise(
                model,
                img,
                img_ids,
                t5_out,
                txt_ids,
                l_pooled,
                timesteps,
                guidance,
                t5_attn_mask,
                neg_t5_out,
                neg_l_pooled,
                neg_t5_attn_mask,
                cfg_scale,
            )

    return x


def generate_image(
    model,
    clip_l: CLIPTextModel,
    t5xxl,
    ae,
    prompt: str,
    seed: Optional[int],
    image_width: int,
    image_height: int,
    steps: Optional[int],
    guidance: float,
    negative_prompt: Optional[str],
    cfg_scale: float,
    idx: int,
    prompt_name: str = "prompt",
    output_dir_block: str = ""
):
    seed = seed if seed is not None else random.randint(0, 2**32 - 1)
    logger.info(f"Seed: {seed}")

    # make first noise with packed shape
    # original: b,16,2*h//16,2*w//16, packed: b,h//16*w//16,16*2*2
    packed_latent_height, packed_latent_width = math.ceil(image_height / 16), math.ceil(image_width / 16)
    noise_dtype = torch.float32 if is_fp8(dtype) else dtype
    noise = torch.randn(
        1,
        packed_latent_height * packed_latent_width,
        16 * 2 * 2,
        device=device,
        dtype=noise_dtype,
        generator=torch.Generator(device=device).manual_seed(seed),
    )

    # prepare img and img ids

    # this is needed only for img2img
    # img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    # if img.shape[0] == 1 and bs > 1:
    #     img = repeat(img, "1 ... -> bs ...", bs=bs)

    # txt2img only needs img_ids
    img_ids = flux_utils.prepare_img_ids(1, packed_latent_height, packed_latent_width)

    # prepare fp8 models
    if is_fp8(clip_l_dtype) and (not hasattr(clip_l, "fp8_prepared") or not clip_l.fp8_prepared):
        logger.info(f"prepare CLIP-L for fp8: set to {clip_l_dtype}, set embeddings to {torch.bfloat16}")
        clip_l.to(clip_l_dtype)  # fp8
        clip_l.text_model.embeddings.to(dtype=torch.bfloat16)
        clip_l.fp8_prepared = True

    if is_fp8(t5xxl_dtype) and (not hasattr(t5xxl, "fp8_prepared") or not t5xxl.fp8_prepared):
        logger.info(f"prepare T5xxl for fp8: set to {t5xxl_dtype}")

        def prepare_fp8(text_encoder, target_dtype):
            def forward_hook(module):
                def forward(hidden_states):
                    hidden_gelu = module.act(module.wi_0(hidden_states))
                    hidden_linear = module.wi_1(hidden_states)
                    hidden_states = hidden_gelu * hidden_linear
                    hidden_states = module.dropout(hidden_states)

                    hidden_states = module.wo(hidden_states)
                    return hidden_states

                return forward

            for module in text_encoder.modules():
                if module.__class__.__name__ in ["T5LayerNorm", "Embedding"]:
                    # print("set", module.__class__.__name__, "to", target_dtype)
                    module.to(target_dtype)
                if module.__class__.__name__ in ["T5DenseGatedActDense"]:
                    # print("set", module.__class__.__name__, "hooks")
                    module.forward = forward_hook(module)

        t5xxl.to(t5xxl_dtype)
        prepare_fp8(t5xxl.encoder, torch.bfloat16)
        t5xxl.fp8_prepared = True

    # prepare embeddings
    logger.info("Encoding prompts...")
    clip_l = clip_l.to(device)
    t5xxl = t5xxl.to(device)

    def encode(prpt: str):
        tokens_and_masks = tokenize_strategy.tokenize(prpt)
        with torch.no_grad():
            if is_fp8(clip_l_dtype):
                with accelerator.autocast():
                    l_pooled, _, _, _ = encoding_strategy.encode_tokens(tokenize_strategy, [clip_l, None], tokens_and_masks)
            else:
                with torch.autocast(device_type=device.type, dtype=clip_l_dtype):
                    l_pooled, _, _, _ = encoding_strategy.encode_tokens(tokenize_strategy, [clip_l, None], tokens_and_masks)

            if is_fp8(t5xxl_dtype):
                with accelerator.autocast():
                    _, t5_out, txt_ids, t5_attn_mask = encoding_strategy.encode_tokens(
                        tokenize_strategy, [clip_l, t5xxl], tokens_and_masks, args.apply_t5_attn_mask
                    )
            else:
                with torch.autocast(device_type=device.type, dtype=t5xxl_dtype):
                    _, t5_out, txt_ids, t5_attn_mask = encoding_strategy.encode_tokens(
                        tokenize_strategy, [None, t5xxl], tokens_and_masks, args.apply_t5_attn_mask
                    )
        return l_pooled, t5_out, txt_ids, t5_attn_mask

    print("prompt: ", prompt)
    l_pooled, t5_out, txt_ids, t5_attn_mask = encode(prompt)
    if negative_prompt:
        neg_l_pooled, neg_t5_out, _, neg_t5_attn_mask = encode(negative_prompt)
    else:
        neg_l_pooled, neg_t5_out, neg_t5_attn_mask = None, None, None

    # NaN check
    if torch.isnan(l_pooled).any():
        raise ValueError("NaN in l_pooled")
    if torch.isnan(t5_out).any():
        raise ValueError("NaN in t5_out")

    if args.offload:
        clip_l = clip_l.cpu()
        t5xxl = t5xxl.cpu()
    # del clip_l, t5xxl
    device_utils.clean_memory()

    # generate image
    logger.info("Generating image...")
    #model = model.to(device)
    # Check if model has meta tensors
    if any(param.is_meta for param in model.parameters()):
        model = model.to_empty(device=device)
    else:
        model = model.to(device)
    
    if steps is None:
        steps = 4 if is_schnell else 50

    img_ids = img_ids.to(device)
    t5_attn_mask = t5_attn_mask.to(device) if args.apply_t5_attn_mask else None

    x = do_sample(
        accelerator,
        model,
        noise,
        img_ids,
        l_pooled,
        t5_out,
        txt_ids,
        steps,
        guidance,
        t5_attn_mask,
        is_schnell,
        device,
        flux_dtype,
        neg_l_pooled,
        neg_t5_out,
        neg_t5_attn_mask,
        cfg_scale,
    )
    if args.offload:
        model = model.cpu()
    # del model
    device_utils.clean_memory()

    # unpack
    x = x.float()
    x = einops.rearrange(x, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=packed_latent_height, w=packed_latent_width, ph=2, pw=2)

    # decode
    logger.info("Decoding image...")
    ae = ae.to(device)
    with torch.no_grad():
        if is_fp8(ae_dtype):
            with accelerator.autocast():
                x = ae.decode(x)
        else:
            with torch.autocast(device_type=device.type, dtype=ae_dtype):
                x = ae.decode(x)
    if args.offload:
        ae = ae.cpu()

    x = x.clamp(-1, 1)
    x = x.permute(0, 2, 3, 1)
    img = Image.fromarray((127.5 * (x + 1.0)).float().cpu().numpy().astype(np.uint8)[0])

    # save image
    output_dir = output_dir_block
    #output_path = os.path.join(output_dir, f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    output_path = os.path.join(output_dir,  prompt_name + "_" + str(idx) + ".png" )
    #output_path = os.path.join(output_dir, prompt_name + ".png" )
    img.save(output_path)

    logger.info(f"Saved image to {output_path}")


if __name__ == "__main__":
    target_height = 1024  # 1024
    target_width = 1024  # 1024

    # steps = 50  # 28  # 50
    # guidance_scale = 5
    # seed = 1  # None  # 1

    device = get_preferred_device()

    parser = argparse.ArgumentParser()
    #parser.add_argument("--ckpt_path", type=str, default="/export/scratch/sheid/flux/transformer/transformer.safetensors")
    # parser.add_argument("--clip_l", type=str, default="/export/scratch/sheid/flux/text_encoder/model.safetensors")
    # parser.add_argument("--t5xxl", type=str, default="/export/scratch/sheid/.cache/hub/models--google--t5-v1_1-xxl/snapshots/3db68a3ef122daf6e605701de53f766d671c19aa/model.safetensors")
    #parser.add_argument("--t5xxl", type=str, default="/export/scratch/sheid/flux/text_encoder_2/model.safetensors")
    parser.add_argument("--ckpt_path", type=str, default="/gpfs/bwfor/work/ws/hd_om233-flux/model_flux/flux/flux1-dev.safetensors")
    parser.add_argument("--ckpt_path_org", type=str, default="/gpfs/bwfor/work/ws/hd_om233-flux/model_flux/flux/flux1-dev.safetensors")
    #parser.add_argument("--ckpt_path", type=str, default="/gpfs/bwfor/work/ws/hd_om233-flux/flux/pix_wave_freeze_double_blocks4_3/test-step00001000.safetensors")
    parser.add_argument("--clip_l", type=str, default="/gpfs/bwfor/work/ws/hd_om233-flux/model_flux/clip/model.safetensors")
    parser.add_argument("--t5xxl", type=str, default="/gpfs/bwfor/work/ws/hd_om233-flux/model_flux/t5xxl/model.safetensors")
    parser.add_argument("--ae", type=str, default="/gpfs/bwfor/work/ws/hd_om233-flux/model_flux/ae/ae.safetensors")
    parser.add_argument("--apply_t5_attn_mask", action="store_true")
    parser.add_argument("--prompt", type=str, default=" A close-up portrait of an elderly man with a weathered face, showing every wrinkle and detail, against a simple, dark background, shot with a shallow depth of field.")
    parser.add_argument("--output_dir", type=str, default="/gpfs/bwfor/work/ws/hd_om233-flux/flux/image/block_analysis/")
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
    #parser.add_argument("--double_blocks", nargs='+', type=int, default=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17])
    parser.add_argument("--single_blocks", nargs='+', type=int, default=[])
    parser.add_argument("--double_blocks", nargs='+', type=int, default=[])
    parser.add_argument("--single_blocks_compress", nargs='+', type=int, default=[])
    parser.add_argument("--double_blocks_compress", nargs='+', type=int, default=[])
    parser.add_argument("--single_blocks_compress_new", nargs='+', type=int, default=[])
    parser.add_argument("--double_blocks_compress_new", nargs='+', type=int, default=[])
    #parser.add_argument("--single_blocks", nargs='+', type=int, default=[])
    parser.add_argument("--image_name", type=str, default="img.png")
    parser.add_argument("--single_flag_attn", action="store_true", help="Flag")
    parser.add_argument("--single_flag_mlp", action="store_true", help="Flag")
    parser.add_argument("--single_flag_mlp2", action="store_true", help="Flag")
    parser.add_argument("--single_flag_mod", action="store_true", help="Flag")

    
    parser.add_argument("--single_comp_mod", nargs='+', type=float, default=[], help="comp")
    parser.add_argument("--single_comp_mlp2", nargs='+', type=float, default=[], help="comp")
    parser.add_argument("--single_comp_attn", nargs='+', type=float, default=[], help="comp")
    parser.add_argument("--single_comp_mlp", nargs='+', type=float, default=[], help="comp")
    
    parser.add_argument("--single_comp_mod_new", nargs='+', type=float, default=[], help="comp")
    parser.add_argument("--single_comp_mlp2_new", nargs='+', type=float, default=[], help="comp")
    parser.add_argument("--single_comp_attn_new", nargs='+', type=float, default=[], help="comp")
    parser.add_argument("--single_comp_mlp_new", nargs='+', type=float, default=[], help="comp")
    
    parser.add_argument("--double_flag_img_attn", action="store_true", help="Flag")
    parser.add_argument("--double_flag_txt_attn", action="store_true", help="Flag")
    parser.add_argument("--double_flag_img_mlp", action="store_true", help="Flag")
    parser.add_argument("--double_flag_txt_mlp", action="store_true", help="Flag")
    parser.add_argument("--double_flag_img_mod", action="store_true", help="Flag")
    parser.add_argument("--double_flag_txt_mod", action="store_true", help="Flag")
    parser.add_argument("--double_flag_img_proj", action="store_true", help="Flag")
    parser.add_argument("--double_flag_txt_proj", action="store_true", help="Flag")
    

    
    parser.add_argument("--double_comp_img_mod", nargs='+', type=float, default=[], help="comp")
    parser.add_argument("--double_comp_img_mlp", nargs='+', type=float, default=[], help="comp")
    parser.add_argument("--double_comp_img_attn",nargs='+', type=float, default=[], help="comp")
    parser.add_argument("--double_comp_txt_mod", nargs='+', type=float, default=[], help="comp")
    parser.add_argument("--double_comp_txt_mlp", nargs='+', type=float, default=[], help="comp")
    parser.add_argument("--double_comp_txt_attn", nargs='+', type=float, default=[], help="comp")
    parser.add_argument("--double_comp_txt_proj", nargs='+', type=float, default=[], help="comp")
    parser.add_argument("--double_comp_img_proj", nargs='+', type=float, default=[], help="comp")
    
    parser.add_argument("--double_comp_img_mod_new", nargs='+', type=float, default=[], help="comp")
    parser.add_argument("--double_comp_img_mlp_new", nargs='+', type=float, default=[], help="comp")
    parser.add_argument("--double_comp_img_attn_new",nargs='+', type=float, default=[], help="comp")
    parser.add_argument("--double_comp_txt_mod_new", nargs='+', type=float, default=[], help="comp")
    parser.add_argument("--double_comp_txt_mlp_new", nargs='+', type=float, default=[], help="comp")
    parser.add_argument("--double_comp_txt_attn_new", nargs='+', type=float, default=[], help="comp")
    parser.add_argument("--double_comp_txt_proj_new", nargs='+', type=float, default=[], help="comp")
    parser.add_argument("--double_comp_img_proj_new", nargs='+', type=float, default=[], help="comp")
    
    parser.add_argument("--double_block_list", nargs='+', type=int, default=[])
    parser.add_argument("--single_block_list", nargs='+', type=int, default=[])
    parser.add_argument("--ref_path", type=str, default="/gpfs/bwfor/work/ws/hd_om233-flux/flux/image/block_investigation_compress_blocks/original")
    parser.add_argument("--logdir", type=str, default="/home/hd/hd_hd/hd_om233/SVD/flux/block_analysis/result.txt")
    parser.add_argument("--prompt_path", type=str, default="/home/hd/hd_hd/hd_om233/partially_removal/100_prompts_laion.json")
    parser.add_argument("--max_blocks_to_remove", type=int, default=15)
    args = parser.parse_args()
  


    
    with open(args.prompt_path, "r") as file:
        data = json.load(file)
        
    prompts = []
    for d in data.values():
        prompts.append(d)
    
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
        
        
    print(loading_device)
    logger.info(f"Loading t5xxl from {args.t5xxl}...")
    t5xxl = flux_utils.load_t5xxl(args.t5xxl, t5xxl_dtype, loading_device)
    #t5xxl = T5EncoderModel.from_pretrained("google/t5-v1_1-xxl")
    t5xxl.eval()
    
    # load clip_l
    logger.info(f"Loading clip_l from {args.clip_l}...")
    clip_l = flux_utils.load_clip_l(args.clip_l, clip_l_dtype, loading_device)
    clip_l.eval()
    t5xxl_max_length =  512
    tokenize_strategy = strategy_flux.FluxTokenizeStrategy(t5xxl_max_length)
    encoding_strategy = strategy_flux.FluxTextEncodingStrategy()

    # AE
    ae = flux_utils.load_ae(args.ae, ae_dtype, loading_device)
    ae.eval()
    
    clip_single = {}
    cmmd_single = {}
    clip_double = {}
    cmmd_double = {}
    # Initialisiere die Listen der zu entfernenden Blöcke
    # Diese Listen enthalten die Blöcke, die permanent entfernt (komprimiert) wurden.
    # Sie müssen aus den args initialisiert werden, falls vorherige Blöcke schon entfernt wurden.
    removed_single_blocks = args.single_blocks_compress.copy()
    removed_double_blocks = args.double_blocks_compress.copy()

    # Initialisiere die Listen der Blöcke, die noch evaluiert werden müssen
    single_blocks_to_evaluate = args.single_block_list.copy()
    double_blocks_to_evaluate = args.double_block_list.copy()

    # HINWEIS: Fügen Sie hier die verbesserte best_network_cmmd_clip Funktion ein.
    # Ich verwende den Namen, den wir im letzten Schritt definiert haben: best_network_cmmd_clip
    # (Ersetzen Sie 'best_network' in der Aufrufanweisung unten durch 'best_network_cmmd_clip'
    # oder verwenden Sie den ursprünglichen Namen, falls Sie nur die Logik aktualisiert haben.)

    MAX_BLOCKS_TO_REMOVE = args.max_blocks_to_remove
    iteration_idx = 0

    while iteration_idx < MAX_BLOCKS_TO_REMOVE and (single_blocks_to_evaluate or double_blocks_to_evaluate):
        iteration_idx += 1
        print(f"\n--- STARTE OPTIMIERUNGS-ITERATION {iteration_idx} von {MAX_BLOCKS_TO_REMOVE} ---")
        
        # Dictionaries für die Ergebnisse dieser Iteration
        cmmd_single = {}
        clip_single = {}
        cmmd_double = {}
        clip_double = {}

        # --- 1. Evaluierung der potenziellen Single Blocks ---
        for s_block in single_blocks_to_evaluate:
            current_removed_single = removed_single_blocks 
            print(f"-> Generiere und evaluiere: Temporäre Entfernung Single Block {s_block}")
            
            # Lade und modifiziere das Modell
            is_schnell, model = flux_utils.load_flow_model(args.ckpt_path_org, None, loading_device)
            model.eval().to(flux_dtype)
            print("Number of original flux model: ", sum(p.numel() for p in model.parameters()))
            model = modify_model(
                model,
                 double_blocks=[],
                 single_blocks=[], 
                single_blocks_comp=args.single_blocks_compress,
                double_blocks_comp=args.double_blocks_compress,
                single_flag_attn=args.single_flag_attn,
                single_flag_mlp=args.single_flag_mlp,
                single_flag_mlp2=args.single_flag_mlp2,
                single_flag_mod=args.single_flag_mod,
                single_comp_mod=args.single_comp_mod,
                single_comp_mlp2=args.single_comp_mlp2,
                single_comp_attn=args.single_comp_attn,
                single_comp_mlp=args.single_comp_mlp,
                double_flag_img_attn=args.double_flag_img_attn,
                double_flag_txt_attn=args.double_flag_txt_attn,
                double_flag_img_mlp= args.double_flag_img_mlp,
                double_flag_txt_mlp=args.double_flag_txt_mlp,
                double_flag_img_mod=args.double_flag_img_mod,
                double_flag_txt_mod=args.double_flag_txt_mod,   
                double_comp_img_mod=args.double_comp_img_mod,
                double_comp_img_mlp=args.double_comp_img_mlp,
                double_comp_img_attn=args.double_comp_img_attn,
                double_comp_txt_mod=args.double_comp_txt_mod,
                double_comp_txt_mlp=args.double_comp_txt_mlp,
                double_comp_txt_attn=args.double_comp_txt_attn,
                double_comp_img_proj=args.double_comp_img_proj,
                double_comp_txt_proj=args.double_comp_txt_proj,
                double_flag_txt_proj=args.double_flag_txt_proj,
                double_flag_img_proj=args.double_flag_img_proj)
            
            for name, param in model.named_parameters():
                if param.is_meta:
                    print(f"Meta tensor found: {name}")
            
            state_dict = load_file(args.ckpt_path)
            model.load_state_dict(state_dict, strict=False)
            if s_block in args.single_blocks_compress:
                idx_single_block = args.single_blocks_compress.index(s_block)
                compression_ratio = args.single_comp_attn[idx_single_block]
                new_compression_ratio = [1 - (1-args.single_comp_attn[idx_single_block])*(1-args.single_comp_attn_new[0])]
            else: 
                new_compression_ratio = args.single_comp_attn_new
            
            print("New compression ratio: ", new_compression_ratio)
            model = modify_model_it(
                model,
                double_blocks=[],
                single_blocks=[], 
                single_blocks_comp_new=[s_block],
                double_blocks_comp_new=[],
                single_blocks_comp = args.single_blocks_compress,
                double_blocks_comp = args.double_blocks_compress,
                single_flag_attn=args.single_flag_attn,
                single_flag_mlp=args.single_flag_mlp,
                single_flag_mlp2=args.single_flag_mlp2,
                single_flag_mod=args.single_flag_mod,
                single_comp_mod=new_compression_ratio,
                single_comp_mlp2=new_compression_ratio,
                single_comp_attn=new_compression_ratio,
                single_comp_mlp=new_compression_ratio,
                double_flag_img_attn=args.double_flag_img_attn,
                double_flag_txt_attn=args.double_flag_txt_attn,
                double_flag_img_mlp= args.double_flag_img_mlp,
                double_flag_txt_mlp=args.double_flag_txt_mlp,
                double_flag_img_mod=args.double_flag_img_mod,
                double_flag_txt_mod=args.double_flag_txt_mod,   
                double_comp_img_mod=args.double_comp_img_mod_new,
                double_comp_img_mlp=args.double_comp_img_mlp_new,
                double_comp_img_attn=args.double_comp_img_attn_new,
                double_comp_txt_mod=args.double_comp_txt_mod_new,
                double_comp_txt_mlp=args.double_comp_txt_mlp_new,
                double_comp_txt_attn=args.double_comp_txt_attn_new,
                double_comp_img_proj=args.double_comp_img_proj_new,
                double_comp_txt_proj=args.double_comp_txt_proj_new,
                double_flag_txt_proj=args.double_flag_txt_proj,
                double_flag_img_proj=args.double_flag_img_proj)
            print("Number of compressed flux model: ", sum(p.numel() for p in model.parameters()))
            output_dir_block = os.path.join(args.output_dir, f"temp_iter{iteration_idx}_single_{s_block}")
            os.makedirs(output_dir_block, exist_ok=True)
            
            print("output_dir_block", output_dir_block)
            
            for idx_prompt, prompt in enumerate(prompts):
                print(output_dir_block+  "prompt" + "_" + str(idx_prompt) + ".png")
                if os.path.isfile(output_dir_block+  "/prompt" + "_" + str(idx_prompt) + ".png" ):
                    print("File exists: " + output_dir_block+  "prompt" + "_" + str(idx_prompt) + ".png" )
                    continue
                # Erzeuge alle Bilder für die Metrikberechnung
                generate_image(
                    model, clip_l, t5xxl, ae, prompt, args.seed, args.width, args.height, args.steps,
                    args.guidance, args.negative_prompt, args.cfg_scale, idx_prompt,output_dir_block= output_dir_block
                )
       
            
            # Berechne Metriken
            try:
                # ANNAHME: compute_cmmd und compute_clip sind implementiert und geben den Metrikwert zurück.
                cmmd_single[s_block] = compute_cmmd(args.ref_path, output_dir_block) 
                clip_single[s_block] = compute_clip(output_dir_block, args.prompt_path)
                print(cmmd_single)
            except Exception as e:
                logger.error(f"Fehler bei Metrikberechnung für Single Block {s_block}: {e}")
                cmmd_single[s_block] = float('inf')
                clip_single[s_block] = 0.0
                
            # del model
            device_utils.clean_memory()
            
            # Löschen der temporären Bilder nach Metrikberechnung (optional, aber empfohlen)
            # import shutil; shutil.rmtree(output_dir_block) 

        # --- 2. Evaluierung der potenziellen Double Blocks ---
        for d_block in double_blocks_to_evaluate:
            current_removed_double = removed_double_blocks 
            print(f"-> Generiere und evaluiere: Temporäre Entfernung Double Block {d_block}")
            
            # Lade und modifiziere das Modell
            is_schnell, model = flux_utils.load_flow_model(args.ckpt_path_org, None, loading_device)
            model.eval().to(flux_dtype)
            
            model = modify_model(
                model,
                 double_blocks=[],
                 single_blocks=[], 
                single_blocks_comp=args.single_blocks_compress,
                double_blocks_comp=args.double_blocks_compress,
                single_flag_attn=args.single_flag_attn,
                single_flag_mlp=args.single_flag_mlp,
                single_flag_mlp2=args.single_flag_mlp2,
                single_flag_mod=args.single_flag_mod,
                single_comp_mod=args.single_comp_mod,
                single_comp_mlp2=args.single_comp_mlp2,
                single_comp_attn=args.single_comp_attn,
                single_comp_mlp=args.single_comp_mlp,
                double_flag_img_attn=args.double_flag_img_attn,
                double_flag_txt_attn=args.double_flag_txt_attn,
                double_flag_img_mlp= args.double_flag_img_mlp,
                double_flag_txt_mlp=args.double_flag_txt_mlp,
                double_flag_img_mod=args.double_flag_img_mod,
                double_flag_txt_mod=args.double_flag_txt_mod,   
                double_comp_img_mod=args.double_comp_img_mod,
                double_comp_img_mlp=args.double_comp_img_mlp,
                double_comp_img_attn=args.double_comp_img_attn,
                double_comp_txt_mod=args.double_comp_txt_mod,
                double_comp_txt_mlp=args.double_comp_txt_mlp,
                double_comp_txt_attn=args.double_comp_txt_attn,
                double_comp_img_proj=args.double_comp_img_proj,
                double_comp_txt_proj=args.double_comp_txt_proj,
                double_flag_txt_proj=args.double_flag_txt_proj,
                double_flag_img_proj=args.double_flag_img_proj)
            
            for name, param in model.named_parameters():
                if param.is_meta:
                    print(f"Meta tensor found: {name}")
            
            state_dict = load_file(args.ckpt_path)
            model.load_state_dict(state_dict, strict=False)
            
            if d_block in args.double_blocks_compress:
                idx_double_block = args.double_blocks_compress.index(d_block)
                compression_ratio = args.double_comp_txt_attn[idx_double_block]
                new_compression_ratio = [1 - (1-args.double_comp_txt_attn[idx_double_block])*(1-args.double_comp_img_attn_new[0])]
            else: 
                new_compression_ratio = args.double_comp_img_attn_new
            model = modify_model_it(
                model,
                double_blocks=[],
                single_blocks=[], 
                single_blocks_comp_new=[],
                double_blocks_comp_new=[d_block],
                single_blocks_comp = args.single_blocks_compress,
                double_blocks_comp = args.double_blocks_compress,
                single_flag_attn=args.single_flag_attn,
                single_flag_mlp=args.single_flag_mlp,
                single_flag_mlp2=args.single_flag_mlp2,
                single_flag_mod=args.single_flag_mod,
                single_comp_mod=args.single_comp_mod_new,
                single_comp_mlp2=args.single_comp_mlp2_new,
                single_comp_attn=args.single_comp_attn_new,
                single_comp_mlp=args.single_comp_mlp_new,
                double_flag_img_attn=args.double_flag_img_attn,
                double_flag_txt_attn=args.double_flag_txt_attn,
                double_flag_img_mlp= args.double_flag_img_mlp,
                double_flag_txt_mlp=args.double_flag_txt_mlp,
                double_flag_img_mod=args.double_flag_img_mod,
                double_flag_txt_mod=args.double_flag_txt_mod,   
                double_comp_img_mod=new_compression_ratio,
                double_comp_img_mlp=new_compression_ratio,
                double_comp_img_attn=new_compression_ratio,
                double_comp_txt_mod=new_compression_ratio,
                double_comp_txt_mlp=new_compression_ratio,
                double_comp_txt_attn=new_compression_ratio,
                double_comp_img_proj=new_compression_ratio,
                double_comp_txt_proj=new_compression_ratio,
                double_flag_txt_proj=args.double_flag_txt_proj,
                double_flag_img_proj=args.double_flag_img_proj)
            print("Number of compressed flux model: ", sum(p.numel() for p in model.parameters()))
            output_dir_block = os.path.join(args.output_dir, f"temp_iter{iteration_idx}_double_{d_block}")
            os.makedirs(output_dir_block, exist_ok=True)
            

            for idx_prompt, prompt in enumerate(prompts):
                # Erzeuge alle Bilder für die Metrikberechnung
                if os.path.isfile(output_dir_block+  "/prompt" + "_" + str(idx_prompt) + ".png" ):
                    print("File exists: " + output_dir_block+  "/prompt" + "_" + str(idx_prompt) + ".png" )
                    continue
      
                generate_image(
                    model, clip_l, t5xxl, ae, prompt, args.seed, args.width, args.height, args.steps,
                    args.guidance, args.negative_prompt, args.cfg_scale, idx_prompt,output_dir_block=output_dir_block
                )
       
                
            # Berechne Metriken
            # try:

            # except Exception as e:
            #     logger.error(f"Fehler bei Metrikberechnung für Double Block {d_block}: {e}")
            #     cmmd_double[d_block] = float('inf')
            #     clip_double[d_block] = 0.0

            del model
            device_utils.clean_memory()
            
            # shutil.rmtree(output_dir_block) # Optional

        # --- 3. Besten Block bestimmen und Listen aktualisieren ---
        if args and hasattr(args, 'logdir'):
            try:
                with open(args.logdir, "a") as f: 
                    f.write(f"\n=======================================================\n")
                    f.write(f"ZWISCHENSTÄNDE FÜR ITERATION {iteration_idx} VOR BLOCKAUSWAHL\n")
                    f.write(f"=======================================================\n")
                    
                    # Protokollierung der Single Block Metriken
                    if cmmd_single and clip_single:
                        f.write("\n### Single Block Scores ###\n")
                        # Die Keys in cmmd_single/clip_single sind die Blöcke, die *evaluiert* wurden.
                        # Wir verwenden die Evaluierungsliste für eine klare Darstellung.
                        for block in single_blocks_to_evaluate:
                            cmmd_val = cmmd_single.get(block, 'FEHLER')
                            clip_val = clip_single.get(block, 'FEHLER')
                            f.write(f"  Single Block {block}: CMMD = {cmmd_val}, CLIP = {clip_val}\n")
                        
                    # Protokollierung der Double Block Metriken
                    if cmmd_double and clip_double:
                        f.write("\n### Double Block Scores ###\n")
                        for block in double_blocks_to_evaluate:
                            cmmd_val = cmmd_double.get(block, 'FEHLER')
                            clip_val = clip_double.get(block, 'FEHLER')
                            f.write(f"  Double Block {block}: CMMD = {cmmd_val}, CLIP = {clip_val}\n")
                    
                    f.write(f"\n-------------------------------------------------------\n")
            except Exception as e:
                logger.error(f"Fehler beim Speichern der Zwischenstände in {args.logdir}: {e}")
        # Rufe die Logik zur Bestimmung des besten Blocks auf
        best_block, block_type = best_network_cmmd_clip( # Verwende die korrigierte Funktion
            iteration_idx,
            cmmd_single=cmmd_single,
            clip_single=clip_single,
            cmmd_double=cmmd_double,
            clip_double=clip_double,
            single_blocks=single_blocks_to_evaluate,
            double_blocks=double_blocks_to_evaluate,
            args=args
        )

        if best_block is None or block_type == "none":
            print("Kein Block zum Entfernen gefunden. Beende Optimierung.")
        
        # Füge den besten Block zu den dauerhaft entfernten Blöcken hinzu und entferne ihn aus der Evaluierungsliste
        if block_type == "single":
            removed_single_blocks.append(best_block)
            single_blocks_to_evaluate.remove(best_block)
            print(f"-> ENTFERNT: Single Block {best_block}. {len(single_blocks_to_evaluate)} Single Blöcke verbleibend.")
        elif block_type == "double":
            removed_double_blocks.append(best_block)
            double_blocks_to_evaluate.remove(best_block)
            print(f"-> ENTFERNT: Double Block {best_block}. {len(double_blocks_to_evaluate)} Double Blöcke verbleibend.")

        with open(args.logdir, "a") as f: 
            f.write("Removed Single Blocks: " + str(removed_single_blocks)+ "\n")
            f.write("Removed Double Blocks: " + str(removed_double_blocks) + "\n")
        device_utils.clean_memory()

    print(f"\n--- OPTIMIERUNG ABGESCHLOSSEN ---")
    print(f"Gesamt entfernte Blöcke: {iteration_idx}")
    print(f"Endgültig entfernte Single Blöcke: {removed_single_blocks}")
    print(f"Endgültig entfernte Double Blöcke: {removed_double_blocks}")