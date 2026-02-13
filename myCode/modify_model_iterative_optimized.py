from typing import Tuple, Union, Optional, Literal

import torch
import torch.nn as nn
from GRASP.model_parts import SingleStreamBlockGRASPCompressed, DoubleStreamBlockGRASPCompressed, SingleStreamBlockGRASP, DoubleStreamBlockGRASP
from myCode.place_holder_blocks_optimized import (
    DoubleStreamBlockPruned,
    Identity,
    IdentityD,
    SingleStreamBlockPruned,
)
import numpy as np

def compute_svd_rank_for_compression(n: int, m: int, ratio_x: float) -> int:
    """
    Berechnet den Ziel-Rang (r') für eine SVD-basierte Kompression, 
    um die Parameterzahl um ratio_x zu reduzieren.
    
    Args:
        n (int): Eingabedimension (Zeilen der Matrix W).
        m (int): Ausgabedimension (Spalten der Matrix W).
        ratio_x (float): Gewünschtes Reduktionsverhältnis der Parameter (z.B. 0.6 für 60% Reduktion).

    Returns:
        int: Der Ziel-Rang r' (aufgerundet).
    """
    P_orig = n * m
    P_target = P_orig * (1.0 - ratio_x)
    r_float = P_target / (n + m)
    r_prime = max(1, int(np.ceil(r_float)))
    if ratio_x >= 1.0:
        return 0
    r_prime = min(r_prime, min(n, m))
    return int(r_prime)


def iterative_svd_pruning(
    A: torch.tensor, 
    B: torch.tensor, 
    rank: int,
) -> Tuple[torch.tensor, torch.tensor]:
    W = A @ B
    W_f32 = W.to(torch.float32)
    U, S, Vh = torch.linalg.svd(W_f32, full_matrices=False)
    U = U.to(W.dtype)
    S = S.to(W.dtype)
    Vh = Vh.to(W.dtype)
    U_r = U[:, :rank]  # shape: [in_features, r]
    S_r = S[:rank]  # shape: [r]
    Vh_r = Vh[:rank, :]  # shape: [r, out_features]
    A = U_r @ torch.diag(torch.sqrt(S_r))  # shape: [in_features, r]
    B = torch.diag(torch.sqrt(S_r)) @ Vh_r  # shape: [r, out_features]
    return A, B

def replace_low_rank_matrices_single_stream(block, new_rank_mod, new_rank_mlp2, new_rank_mlp, new_rank_attn, flag_attn, flag_mlp, flag_mlp2, flag_mod):
    if flag_attn:
        new_q_A, new_q_B = iterative_svd_pruning(block.linear_q_A, block.linear_q_B, new_rank_attn)
        block.linear_q_A, block.linear_q_B = nn.Parameter(new_q_A), nn.Parameter(new_q_B)
        
        new_k_A, new_k_B = iterative_svd_pruning(block.linear_k_A, block.linear_k_B, new_rank_attn)
        block.linear_k_A, block.linear_k_B = nn.Parameter(new_k_A), nn.Parameter(new_k_B)
        
        new_v_A, new_v_B = iterative_svd_pruning(block.linear_v_A, block.linear_v_B, new_rank_attn)
        block.linear_v_A, block.linear_v_B = nn.Parameter(new_v_A), nn.Parameter(new_v_B)

    
    if flag_mlp:
        new_mlp_A, new_mlp_B = iterative_svd_pruning(block.linear_mlp_A, block.linear_mlp_B, new_rank_mlp)
        block.linear_mlp_A, block.linear_mlp_B = nn.Parameter(new_mlp_A), nn.Parameter(new_mlp_B)
        
        
    if flag_mlp2:
        new_mlp2_A, new_mlp2_B = iterative_svd_pruning(block.linear_mlp2_A, block.linear_mlp2_B, new_rank_mlp2)
        block.linear_mlp2_A, block.linear_mlp2_B = nn.Parameter(new_mlp2_A), nn.Parameter(new_mlp2_B)
        
    
    if flag_mod:
        new_mod_A, new_mod_B = iterative_svd_pruning(block.modulation.linear_A, block.modulation.linear_B, new_rank_mod)
        block.modulation.linear_A, block.modulation.linear_B = nn.Parameter(new_mod_A), nn.Parameter(new_mod_B)
        
    return block
        
        

def replace_low_rank_matrices_double_stream(block, new_rank_img_attn, new_rank_img_mlp, new_rank_img_proj, new_rank_img_mod,  new_rank_txt_attn, new_rank_txt_mlp, new_rank_txt_proj, new_rank_txt_mod, 
                                            flag_img_attn, flag_img_mlp, flag_img_proj, flag_img_mod, flag_txt_attn, flag_txt_mlp, flag_txt_proj, flag_txt_mod,
                                            ):
    if flag_img_attn:
        new_img_q_A, new_img_q_B = iterative_svd_pruning(block.img_attn.linear_q_A, block.img_attn.linear_q_B, new_rank_img_attn)
        block.img_attn.linear_q_A, block.img_attn.linear_q_B = nn.Parameter(new_img_q_A), nn.Parameter(new_img_q_B)
        
        new_img_k_A, new_img_k_B = iterative_svd_pruning(block.img_attn.linear_k_A, block.img_attn.linear_k_B, new_rank_img_attn)
        block.img_attn.linear_k_A, block.img_attn.linear_k_B = nn.Parameter(new_img_k_A), nn.Parameter(new_img_k_B)
        
        new_img_v_A, new_img_v_B = iterative_svd_pruning(block.img_attn.linear_v_A, block.img_attn.linear_v_B, new_rank_img_attn)
        block.img_attn.linear_v_A, block.img_attn.linear_v_B = nn.Parameter(new_img_v_A), nn.Parameter(new_img_v_B)

    
    if flag_img_mlp:
        new_img_mlp1_A, new_img_mlp1_B = iterative_svd_pruning(block.img_mlp_linear1_A, block.img_mlp_linear1_B, new_rank_img_mlp)
        block.img_mlp_linear1_A, block.img_mlp_linear1_B = nn.Parameter(new_img_mlp1_A), nn.Parameter(new_img_mlp1_B)
        
        new_img_mlp2_A, new_img_mlp2_B = iterative_svd_pruning(block.img_mlp_linear2_A, block.img_mlp_linear2_B, new_rank_img_mlp)
        block.img_mlp_linear2_A, block.img_mlp_linear2_B = nn.Parameter(new_img_mlp2_A), nn.Parameter(new_img_mlp2_B)
        
        
    if flag_img_proj:
        new_proj_A, new_proj_B = iterative_svd_pruning(block.img_attn.proj_linear_A, block.img_attn.proj_linear_B, new_rank_img_proj)
        block.img_attn.proj_linear_A, block.img_attn.proj_linear_B = nn.Parameter(new_proj_A), nn.Parameter(new_proj_B)
        
    
    if flag_img_mod:
        new_mod_A, new_mod_B = iterative_svd_pruning(block.img_mod.linear_A, block.img_mod.linear_B, new_rank_img_mod)
        block.img_mod.linear_A, block.img_mod.linear_B = nn.Parameter(new_mod_A), nn.Parameter(new_mod_B)
        
        
        
    if flag_txt_attn:
        new_img_q_A, new_img_q_B = iterative_svd_pruning(block.txt_attn.linear_q_A, block.txt_attn.linear_q_B, new_rank_txt_attn)
        block.txt_attn.linear_q_A, block.txt_attn.linear_q_B = nn.Parameter(new_img_q_A), nn.Parameter(new_img_q_B)
        
        new_img_k_A, new_img_k_B = iterative_svd_pruning(block.txt_attn.linear_k_A, block.txt_attn.linear_k_B, new_rank_txt_attn)
        block.txt_attn.linear_k_A, block.txt_attn.linear_k_B = nn.Parameter(new_img_k_A), nn.Parameter(new_img_k_B)
        
        new_img_v_A, new_img_v_B = iterative_svd_pruning(block.txt_attn.linear_v_A, block.txt_attn.linear_v_B, new_rank_txt_attn)
        block.txt_attn.linear_v_A, block.txt_attn.linear_v_B = nn.Parameter(new_img_v_A), nn.Parameter(new_img_v_B)

    
    if flag_txt_mlp:
        new_img_mlp1_A, new_img_mlp1_B = iterative_svd_pruning(block.txt_mlp_linear1_A, block.txt_mlp_linear1_B, new_rank_txt_mlp)
        block.txt_mlp_linear1_A, block.txt_mlp_linear1_B = nn.Parameter(new_img_mlp1_A), nn.Parameter(new_img_mlp1_B)
        
        new_img_mlp2_A, new_img_mlp2_B = iterative_svd_pruning(block.txt_mlp_linear2_A, block.txt_mlp_linear2_B, new_rank_txt_mlp)
        block.txt_mlp_linear2_A, block.txt_mlp_linear2_B = nn.Parameter(new_img_mlp2_A), nn.Parameter(new_img_mlp2_B)
        
        
    if flag_txt_proj:
        new_proj_A, new_proj_B = iterative_svd_pruning(block.txt_attn.proj_linear_A, block.txt_attn.proj_linear_B, new_rank_txt_proj)
        block.txt_attn.proj_linear_A, block.txt_attn.proj_linear_B = nn.Parameter(new_proj_A), nn.Parameter(new_proj_B)
        
    
    if flag_txt_mod:
        new_mod_A, new_mod_B = iterative_svd_pruning(block.txt_mod.linear_A, block.txt_mod.linear_B, new_rank_txt_mod)
        block.txt_mod.linear_A, block.txt_mod.linear_B = nn.Parameter(new_mod_A), nn.Parameter(new_mod_B)
    
    return block
              
        
        
def modify_model(model,
                 double_blocks,
                 single_blocks, 
                 single_blocks_comp = [],
                 double_blocks_comp = [],
                 single_flag_attn=False,
                 single_flag_mlp=False,
                 single_flag_mlp2=False,
                 single_flag_mod=False,
                 single_comp_mod=1.0,
                 single_comp_mlp2=1.0,
                 single_comp_attn=1.0,
                 single_comp_mlp=1.0,
                 double_flag_img_attn=False,
                 double_flag_txt_attn=False,
                 double_flag_img_mlp=False,
                 double_flag_txt_mlp=False,
                 double_flag_img_mod=False,
                 double_flag_txt_mod=False,
                 double_flag_txt_proj=False,
                 double_flag_img_proj=False,
                 double_comp_img_mod=1.0,
                 double_comp_img_mlp=1.0,
                 double_comp_img_attn=1.0,
                 double_comp_txt_mod=1.0,
                 double_comp_txt_mlp=1.0,
                 double_comp_txt_attn=1.0,
                 double_comp_txt_proj=1.0,
                 double_comp_img_proj=1.0
                 ):
    
    
 
    
    

    for idx in double_blocks:
        model.double_blocks[idx] = IdentityD(hidden_size=int(3072),mlp_ratio= 4.0, num_heads=int(24))
    for idx in single_blocks:
        model.single_blocks[idx] = Identity(hidden_size=int(3072), num_heads=int(24))
    for j, idx in enumerate(single_blocks_comp):
        single_rank_mod = compute_svd_rank_for_compression(3072, 9216, single_comp_mod[j])
        single_rank_mlp = compute_svd_rank_for_compression(3072, 12288, single_comp_mlp[j])
        single_rank_mlp2 = compute_svd_rank_for_compression(15360, 3072, single_comp_mlp2[j])
        single_rank_attn = compute_svd_rank_for_compression(3072, 3072, single_comp_attn[j])
        model.single_blocks[idx] = SingleStreamBlockPruned(model.single_blocks[idx],
                                                           rank_attn=single_rank_attn,
                                                           rank_mlp=single_rank_mlp,
                                                           rank_mlp2=single_rank_mlp2,
                                                           rank_mod=single_rank_mod,
                                                           flag_mod=single_flag_mod,
                                                           flag_attn=single_flag_attn,
                                                           flag_mlp=single_flag_mlp,
                                                           flag_mlp2=single_flag_mlp2)
    for j, idx in enumerate(double_blocks_comp):
        double_rank_img_attn = compute_svd_rank_for_compression(3072, 3072, double_comp_img_attn[j])
        double_rank_img_mlp = compute_svd_rank_for_compression(3072, 12288, double_comp_img_mlp[j])
        double_rank_img_proj = compute_svd_rank_for_compression(3072, 3072, double_comp_img_proj[j])
        double_rank_img_mod = compute_svd_rank_for_compression(3072, 18432, double_comp_img_mod[j])
        
        double_rank_txt_attn = compute_svd_rank_for_compression(3072, 3072, double_comp_txt_attn[j])
        double_rank_txt_mlp = compute_svd_rank_for_compression(3072, 12288, double_comp_txt_mlp[j])
        double_rank_txt_proj = compute_svd_rank_for_compression(3072, 3072, double_comp_txt_proj[j])
        double_rank_txt_mod = compute_svd_rank_for_compression(3072, 18432, double_comp_txt_mod[j])
        model.double_blocks[idx] = DoubleStreamBlockPruned( model.double_blocks[idx],
                                                            rank_attn_img=double_rank_img_attn,
                                                            rank_attn_txt=double_rank_txt_attn,
                                                            rank_img_mlp_in=double_rank_img_mlp,
                                                            rank_img_mlp_out=double_rank_img_mlp,
                                                            rank_txt_mlp_in=double_rank_txt_mlp,
                                                            rank_txt_mlp_out=double_rank_txt_mlp,
                                                            rank_img_mod=double_rank_img_mod,
                                                            rank_txt_mod=double_rank_txt_mod,
                                                            rank_img_proj=double_rank_img_proj,
                                                            rank_txt_proj=double_rank_txt_proj,
                                                            flag_img_attn=double_flag_img_attn,
                                                            flag_txt_attn=double_flag_txt_attn,
                                                            flag_img_mlp=double_flag_img_mlp,
                                                            flag_txt_mlp=double_flag_txt_mlp,
                                                            flag_img_mod=double_flag_img_mod,
                                                            flag_txt_mod=double_flag_txt_mod,
                                                            flag_img_proj=double_flag_img_proj,
                                                            flag_txt_proj=double_flag_txt_proj,
                                                            )
    return model


def modify_model_it(model,
                 double_blocks,
                 single_blocks, 
                 single_blocks_comp = [],
                 double_blocks_comp = [],
                 single_blocks_comp_new = [],
                 double_blocks_comp_new = [],
                 single_flag_attn=False,
                 single_flag_mlp=False,
                 single_flag_mlp2=False,
                 single_flag_mod=False,
                 single_comp_mod=1.0,
                 single_comp_mlp2=1.0,
                 single_comp_attn=1.0,
                 single_comp_mlp=1.0,
                 double_flag_img_attn=False,
                 double_flag_txt_attn=False,
                 double_flag_img_mlp=False,
                 double_flag_txt_mlp=False,
                 double_flag_img_mod=False,
                 double_flag_txt_mod=False,
                 double_flag_txt_proj=False,
                 double_flag_img_proj=False,
                 double_comp_img_mod=1.0,
                 double_comp_img_mlp=1.0,
                 double_comp_img_attn=1.0,
                 double_comp_txt_mod=1.0,
                 double_comp_txt_mlp=1.0,
                 double_comp_txt_attn=1.0,
                 double_comp_txt_proj=1.0,
                 double_comp_img_proj=1.0
                 ):
    # Example modification: replace a specific layer with an Identity layer

    
    

    for idx in double_blocks:
        model.double_blocks[idx] = IdentityD(hidden_size=int(3072),mlp_ratio= 4.0, num_heads=int(24))
    for idx in single_blocks:
        model.single_blocks[idx] = Identity(hidden_size=int(3072), num_heads=int(24))

    for j, idx in enumerate(single_blocks_comp_new):
        single_rank_mod = compute_svd_rank_for_compression(3072, 9216, single_comp_mod[j])
        single_rank_mlp = compute_svd_rank_for_compression(3072, 12288, single_comp_mlp[j])
        single_rank_mlp2 = compute_svd_rank_for_compression(15360, 3072, single_comp_mlp2[j])
        single_rank_attn = compute_svd_rank_for_compression(3072, 3072, single_comp_attn[j])
        if idx in single_blocks_comp:
            model.single_blocks[idx] = replace_low_rank_matrices_single_stream(model.single_blocks[idx],
                                                                            single_rank_mod,
                                                                            single_rank_mlp2,
                                                                            single_rank_mlp,
                                                                            single_rank_attn,
                                                                            single_flag_attn,
                                                                            single_flag_mlp,
                                                                            single_flag_mlp2,
                                                                            single_flag_mod)
        else: 
            model.single_blocks[idx] = SingleStreamBlockPruned(model.single_blocks[idx],
                                                           rank_attn=single_rank_attn,
                                                           rank_mlp=single_rank_mlp,
                                                           rank_mlp2=single_rank_mlp2,
                                                           rank_mod=single_rank_mod,
                                                           flag_mod=single_flag_mod,
                                                           flag_attn=single_flag_attn,
                                                           flag_mlp=single_flag_mlp,
                                                           flag_mlp2=single_flag_mlp2)
            
    for j, idx in enumerate(double_blocks_comp_new):
        double_rank_img_attn = compute_svd_rank_for_compression(3072, 3072, double_comp_img_attn[j])
        double_rank_img_mlp = compute_svd_rank_for_compression(3072, 12288, double_comp_img_mlp[j])
        double_rank_img_proj = compute_svd_rank_for_compression(3072, 3072, double_comp_img_proj[j])
        double_rank_img_mod = compute_svd_rank_for_compression(3072, 18432, double_comp_img_mod[j])
        
        double_rank_txt_attn = compute_svd_rank_for_compression(3072, 3072, double_comp_txt_attn[j])
        double_rank_txt_mlp = compute_svd_rank_for_compression(3072, 12288, double_comp_txt_mlp[j])
        double_rank_txt_proj = compute_svd_rank_for_compression(3072, 3072, double_comp_txt_proj[j])
        double_rank_txt_mod = compute_svd_rank_for_compression(3072, 18432, double_comp_txt_mod[j])
        if idx in double_blocks_comp:
            model.double_blocks[idx] = replace_low_rank_matrices_double_stream(model.double_blocks[idx],
                                                                            double_rank_img_attn,
                                                                            double_rank_img_mlp,
                                                                            double_rank_img_proj,                                                                   
                                                                            double_rank_img_mod,
                                                                            double_rank_txt_attn,
                                                                            double_rank_txt_mlp,
                                                                            double_rank_txt_proj,
                                                                            double_rank_txt_mod, 
                                                                            double_flag_img_attn,
                                                                            double_flag_img_mlp,
                                                                            double_flag_img_proj,
                                                                            double_flag_img_mod,
                                                                            double_flag_txt_attn,
                                                                            double_flag_txt_mlp,
                                                                            double_flag_txt_proj,
                                                                            double_flag_txt_mod,
                                                                                )
        else: 
            model.double_blocks[idx] = DoubleStreamBlockPruned( model.double_blocks[idx],
                                                            rank_attn_img=double_rank_img_attn,
                                                            rank_attn_txt=double_rank_txt_attn,
                                                            rank_img_mlp_in=double_rank_img_mlp,
                                                            rank_img_mlp_out=double_rank_img_mlp,
                                                            rank_txt_mlp_in=double_rank_txt_mlp,
                                                            rank_txt_mlp_out=double_rank_txt_mlp,
                                                            rank_img_mod=double_rank_img_mod,
                                                            rank_txt_mod=double_rank_txt_mod,
                                                            rank_img_proj=double_rank_img_proj,
                                                            rank_txt_proj=double_rank_txt_proj,
                                                            flag_img_attn=double_flag_img_attn,
                                                            flag_txt_attn=double_flag_txt_attn,
                                                            flag_img_mlp=double_flag_img_mlp,
                                                            flag_txt_mlp=double_flag_txt_mlp,
                                                            flag_img_mod=double_flag_img_mod,
                                                            flag_txt_mod=double_flag_txt_mod,
                                                            flag_img_proj=double_flag_img_proj,
                                                            flag_txt_proj=double_flag_txt_proj,
                                                            )
    return model


