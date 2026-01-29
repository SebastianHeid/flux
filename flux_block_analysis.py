# Minimum Inference Code for FLUX

import argparse
import datetime
import math
import os
import random
from typing import Callable, List, Optional
import yaml 
from box import Box
import accelerate
import einops
from torch.utils.data import DataLoader 
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
import sys
sys.path.append("/export/home/sheid/MasterThesis_Evaluation/datasets")
from eval_dataset import PromptFolderDataset
init_ipex()


from library.utils import setup_logging, str_to_dtype

setup_logging()
import logging

logger = logging.getLogger(__name__)

import networks.lora_flux as lora_flux
from library import flux_models, flux_utils, sd3_utils, strategy_flux


def best_network(idx, cmmd_single, lpips_single, cmmd_double, lpips_double, single_blocks, double_blocks):
    if cmmd_double:
        double_total_position = {}
        double_position_cmmd = {}
        double_position_lpips = {}
        sorted_cmmd_double  = sorted(cmmd_double.items(), key=lambda item: item[1])
        sorted_lpips_double = sorted(lpips_double.items(), key=lambda item: item[1])
        
        for idx, (key, _) in enumerate(sorted_cmmd_double):
            double_position_cmmd[key] = idx
        for idx, (key, _) in enumerate(sorted_lpips_double):
            double_position_lpips[key] = idx
        
        for key in double_position_cmmd:
            double_total_position[key] = double_position_cmmd[key]+double_position_lpips[key] 
        double_best_block = min(double_total_position, key=double_total_position.get)
    
    if cmmd_single:
        single_total_position = {}
        single_position_cmmd = {}
        single_position_lpips = {}
        
        sorted_cmmd_single = sorted(cmmd_single.items(), key=lambda item: item[1])
        sorted_lpips_single = sorted(lpips_single.items(), key=lambda item: item[1])
        
        for idx, (key, _) in enumerate(sorted_cmmd_single):
            single_position_cmmd[key] = idx
        for idx, (key, _) in enumerate(sorted_lpips_single):
            single_position_lpips[key] = idx
            
        for key in single_position_cmmd:
            single_total_position[key] = single_position_cmmd[key]+single_position_lpips[key] 
        single_best_block = min(single_total_position, key=single_total_position.get)
    
    
    if not cmmd_double:
        best_block = single_best_block
        block_type = "single"
    elif not cmmd_single:
        best_block = double_best_block
        block_type = "double"
    elif  min(single_total_position.values()) <= min(double_total_position.values()):
        best_block = single_best_block
        block_type = "single"
    else:
        best_block = double_best_block
        block_type = "double"

    
    with open("./block_analysis/results.txt", "a") as f: 
        f.write(str(idx) + ". block removed from "+str(block_type) + " blocks \n")
        f.write("Best block: " + str(best_block) + "\n")
        
        if cmmd_single:
            for i in range(len(single_blocks)):
                f.write("Block: " + str(single_blocks[i]) + " LPIPS: " + str(lpips_single[single_blocks[i]]) + " CMMD: " + str(cmmd_single[single_blocks[i]]) + " total position: " +str(single_total_position[single_blocks[i]]) + "\n")
        if cmmd_double:
            for i in range(len(double_blocks)):
                f.write("Block: " + str(double_blocks[i]) + " LPIPS: " + str(lpips_double[double_blocks[i]]) + " CMMD: " + str(cmmd_double[double_blocks[i]]) + " total position: " +str(double_total_position[double_blocks[i]]) + "\n")
        
        f.write("\n")
        f.write("\n")
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
    config,
    prompt: str,
    seed: Optional[int],
    image_width: int,
    image_height: int,
    steps: Optional[int],
    guidance: float,
    negative_prompt: Optional[str],
    cfg_scale: float,
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
                        tokenize_strategy, [clip_l, t5xxl], tokens_and_masks, config.apply_t5_attn_mask
                    )
            else:
                with torch.autocast(device_type=device.type, dtype=t5xxl_dtype):
                    _, t5_out, txt_ids, t5_attn_mask = encoding_strategy.encode_tokens(
                        tokenize_strategy, [None, t5xxl], tokens_and_masks, config.apply_t5_attn_mask
                    )
        return l_pooled, t5_out, txt_ids, t5_attn_mask

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

    if config.offload:
        clip_l = clip_l.cpu()
        t5xxl = t5xxl.cpu()
    # del clip_l, t5xxl
    device_utils.clean_memory()

    # generate image
    logger.info("Generating image...")
    model = model.to(device)
    if steps is None:
        steps = 4 if is_schnell else 50

    img_ids = img_ids.to(device)
    t5_attn_mask = t5_attn_mask.to(device) if config.apply_t5_attn_mask else None

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
    if config.offload:
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
    if config.offload:
        ae = ae.cpu()

    x = x.clamp(-1, 1)
    x = x.permute(0, 2, 3, 1)
    img = Image.fromarray((127.5 * (x + 1.0)).float().cpu().numpy().astype(np.uint8)[0])

    # save image
    output_dir = config.output_dir
    #output_path = os.path.join(output_dir, f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    output_path = os.path.join(output_dir)
    img.save(output_path)

    logger.info(f"Saved image to {output_path}")


def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "--config_path",
        type=str,
        const=True,
        default="/home/hd/hd_hd/hd_om233/flux/block_analysis_config/config.yaml",
        nargs="?",
        help="Path to config.yaml file",
    )
    return parser


if __name__ == "__main__":
    target_height = 1024  # 1024
    target_width = 1024  # 1024

    # steps = 50  # 28  # 50
    # guidance_scale = 5
    # seed = 1  # None  # 1

    device = get_preferred_device()
    
    parser = get_parser()
    args = parser.parse_args()
    with open(args.config_path, "r") as file:
        config = Box(yaml.safe_load(file))
    
    seed = config.seed
    guidance_scale = config.guidance
    steps = config.steps
    

    def is_fp8(dt):
        return dt in [torch.float8_e4m3fn, torch.float8_e4m3fnuz, torch.float8_e5m2, torch.float8_e5m2fnuz]

    dtype = str_to_dtype(config.dtype)
    clip_l_dtype = str_to_dtype(config.clip_l_dtype, dtype)
    t5xxl_dtype = str_to_dtype(config.t5xxl_dtype, dtype)
    ae_dtype = str_to_dtype(config.ae_dtype, dtype)
    flux_dtype = str_to_dtype(config.flux_dtype, dtype)

    logger.info(f"Dtypes for clip_l, t5xxl, ae, flux: {clip_l_dtype}, {t5xxl_dtype}, {ae_dtype}, {flux_dtype}")

    loading_device = "cpu" if config.offload else device

    use_fp8 = [is_fp8(d) for d in [dtype, clip_l_dtype, t5xxl_dtype, ae_dtype, flux_dtype]]
    if any(use_fp8):
        accelerator = accelerate.Accelerator(mixed_precision="bf16")
    else:
        accelerator = None

    logger.info(f"Loading t5xxl from {config.t5xxl}...")
    t5xxl = flux_utils.load_t5xxl(config.t5xxl, t5xxl_dtype, loading_device)
    t5xxl.eval()
    
    # load clip_l
    logger.info(f"Loading clip_l from {config.clip_l}...")
    clip_l = flux_utils.load_clip_l(config.clip_l, clip_l_dtype, loading_device)
    clip_l.eval()

    t5xxl_max_length = 512
    tokenize_strategy = strategy_flux.FluxTokenizeStrategy(t5xxl_max_length)
    encoding_strategy = strategy_flux.FluxTextEncodingStrategy()

    # AE
    ae = flux_utils.load_ae(config.ae, ae_dtype, loading_device)
    ae.eval()
    
    with open(config.prompt_path, "r") as file:
        prompts = file.readlines()
    prompts = [prompt.strip() for prompt in prompts]  # Remove empty lines
    
    single_blocks = np.linspace(0,37, num=38, dtype=int)
    double_blocks = np.linspace(0,18, num=19, dtype=int)
    removed_single_blocks = []
    removed_double_blocks = []
    
    # paths
    ref_path = config.generated_images_path + "/original"
    single_path = config.generated_images_path + "/single_block"
    double_path = config.generated_images_path + "/double_block"
    
    # compute original images
    config.double_blocks = []
    config.single_blocks = []
   
    is_schnell, model = flux_utils.load_flow_model(config.ckpt_path, None, loading_device)
    model.eval()
    logger.info(f"Casting model to {flux_dtype}")
    model.to(flux_dtype)  # make sure model is dtype
    for prompt_idx, prompt in enumerate(prompts):
            config.output_dir = ref_path  + "/prompt_"+ str(prompt_idx) + ".png"
            if not os.path.exists(ref_path):
                os.makedirs(ref_path)
            generate_image(
                model,
                clip_l,
                t5xxl,
                ae,
                config, 
                prompt,
                config.seed,
                config.width,
                config.height,
                config.steps,
                config.guidance,
                config.negative_prompt,
                config.cfg_scale,
            )
    
    for removed_blocks in range(config.num_reduced_blocks):
        lpips_single = {}
        cmmd_single = {}
        lpips_double = {}
        cmmd_double = {}
        if config.remove_single_blocks:
            for idx_single_block in single_blocks:
                config.single_blocks = [idx_single_block]
                config.double_blocks = []
                del model
                is_schnell, model = flux_utils.load_flow_model(config.ckpt_path, None, loading_device)
                model.eval()
                logger.info(f"Casting model to {flux_dtype}")
                model.to(flux_dtype)  # make sure model is dtype
                model = modify_model(model,config.double_blocks, config.single_blocks)
                model = modify_model(model,removed_double_blocks, removed_single_blocks)
                for prompt_idx, prompt in enumerate(prompts):
                    config.output_dir = single_path + str(idx_single_block) + "/prompt_"+ str(prompt_idx) + ".png"
                    if not os.path.exists(single_path + str(idx_single_block)):
                        os.makedirs("single_path" + str(idx_single_block))
                    generate_image(
                        model,
                        clip_l,
                        t5xxl,
                        ae,
                        config, 
                        prompt,
                        config.seed,
                        config.width,
                        config.height,
                        config.steps,
                        config.guidance,
                        config.negative_prompt,
                        config.cfg_scale,
                    )
                lpips_single[idx_single_block] = calculate_lpips(ref_path, img_path)
                cmmd_single[idx_single_block] = compute_cmmd(ref_path, img_path)  
                 
            with open("./block_analysis/lpips_single_dict.jsone", "a") as f:
                f.dump(lpips_single)
            with open("./block_analysis/cmmd_single_dict.jsone", "a") as f:
                f.dump(cmmd_single) 
                    
                    
        elif config.remove_double_blocks:
            config.double_blocks = []
            config.single_blocks = []
            for idx_double_block in double_blocks:
                config.double_blocks = [idx_double_block]
                del model
                is_schnell, model = flux_utils.load_flow_model(config.ckpt_path, None, loading_device)
                model.eval()
                logger.info(f"Casting model to {flux_dtype}")
                model.to(flux_dtype)  # make sure model is dtype
                model = modify_model(model,config.double_blocks, config.single_blocks)
                model = modify_model(model,removed_double_blocks, removed_single_blocks)
                for prompt_idx, prompt in enumerate(prompts):
                    config.output_dir = double_path + str(idx_double_block) + "/prompt_"+ str(prompt_idx) + ".png"
                    if not os.path.exists(double_path + str(idx_double_block)):
                        os.makedirs(double_path + str(idx_double_block))
                    generate_image(
                        model,
                        clip_l,
                        t5xxl,
                        ae,
                        config, 
                        prompt,
                        config.seed,
                        config.width,
                        config.height,
                        config.steps,
                        config.guidance,
                        config.negative_prompt,
                        config.cfg_scale,
                    )
                    
            with open("./block_analysis/lpips_double_dict.jsone", "a") as f:
                f.dump(lpips_double)
            with open("./block_analysis/cmmd_double_dict.jsone", "a") as f:
                f.dump(cmmd_double) 
            
        best_block, block_type = best_network(cmmd_single, lpips_single, cmmd_double, lpips_double, single_blocks, double_blocks)
        if block_type=="single":
            removed_single_blocks.append(best_block)
            single_blocks.remove(best_block)
        elif block_type=="double":
            removed_double_blocks.append(best_block)
            double_blocks.remove(best_block)
            
        with open("./block_analysis/results.txt", "a") as f: 
            f.write("Remaining single blocks: " + str(single_blocks) + "\n")
            f.write("Remaining double blocks: " + str(double_blocks) + "\n")
            f.write("\n")
            f.write("\n")
        
    logger.info("Done!")
