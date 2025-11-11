from copy import deepcopy
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from einops import rearrange
from library.flux_models import (
    DoubleStreamBlock,
    Modulation,
    ModulationOut,
    QKNorm,
    SingleStreamBlock,
    attention,
)
from torch import Tensor

#     def forward(self, img, txt, vec, pe, txt_attention_mask= None):
#         return img, txt
from torch.utils.checkpoint import checkpoint

# class Identity(SingleStreamBlock):
#     def __init__(self, hidden_size, num_heads):
#         super().__init__(hidden_size=hidden_size, num_heads=num_heads)

#     def forward(self, x, *args, **kargs):
#         return x

# class IdentityD(DoubleStreamBlock):
#     def __init__(self, hidden_size, mlp_ratio, num_heads):
#         super().__init__(hidden_size=hidden_size,mlp_ratio= mlp_ratio, num_heads=num_heads)

import torch
from typing import Union, Tuple

def decompose_linear_to_svd_soft(
    W: torch.Tensor,
    r: int,
    reverse: bool = False,
    return_full: bool = False,
    soft_threshold: bool = True,
    alpha: float = 0.1,  # Steepness of soft cutoff
    preserve_energy: float = 0.999,  # Alternative: energy-based rank selection
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Decomposes a weight matrix into two LoRA-style matrices using SVD.
    
    Args:
        W: Weight matrix to decompose
        r: Target rank for decomposition
        reverse: If True, use complementary singular values
        return_full: If True, return reconstructed matrix instead of factors
        soft_threshold: If True, use soft weighting instead of hard cutoff
        alpha: Steepness parameter for soft cutoff (higher = harder cutoff)
        preserve_energy: Minimum energy to preserve (0.999 = 99.9%)
    
    Returns:
        (A, B): Low-rank factors such that A @ B ≈ W, or reconstructed W if return_full=True
    """
    orig_dtype = W.dtype
    device = W.device
    
    # Convert to float32 for SVD
    W_32 = W.to(torch.float32)
    
    # Perform SVD
    U, S, Vh = torch.linalg.svd(W_32.T, full_matrices=False)
    
    # === SOFT THRESHOLDING LOGIC ===
    if soft_threshold:
        # Calculate cumulative energy
        energy = S ** 2
        cumsum_energy = torch.cumsum(energy, dim=0)
        total_energy = cumsum_energy[-1]
        
        # Option 1: Energy-based adaptive rank (overrides r if needed)
        if preserve_energy is not None:
            threshold = preserve_energy * total_energy
            adaptive_r = torch.searchsorted(cumsum_energy, threshold).item() + 1
            effective_r = min(adaptive_r, r, len(S))
            print(f"Adaptive rank: {effective_r} (target: {r}, energy: {cumsum_energy[effective_r-1]/total_energy:.4f})")
        else:
            effective_r = min(r, len(S))
        
        # Create soft weighting mask
        indices = torch.arange(len(S), device=device, dtype=torch.float32)
        
        if not reverse:
            # Soft sigmoid cutoff around rank r
            # weights = 1 for i << r, weights → 0 for i >> r
            weights = torch.sigmoid(-alpha * (indices - effective_r))
        else:
            # Inverted: keep high indices
            weights = torch.sigmoid(alpha * (indices - effective_r))
        
        # Apply soft weights to singular values
        S_weighted = S * weights
        
        # Optional: Remove near-zero components to save memory
        threshold_val = 1e-6 * S[0]  # Relative to largest singular value
        keep_mask = S_weighted > threshold_val
        keep_r = keep_mask.sum().item()
        
        U_r = U[:, :keep_r]
        S_r = S_weighted[:keep_r]
        Vh_r = Vh[:keep_r, :]
        
    else:
        # === ORIGINAL HARD CUTOFF ===
        if not reverse:
            U_r = U[:, :r]
            S_r = S[:r]
            Vh_r = Vh[:r, :]
        else:
            U_r = U[:, r:]
            S_r = S[r:]
            Vh_r = Vh[r:, :]
    
    # Reconstruct or form low-rank factors
    if return_full:
        W_rec = (U_r @ torch.diag(S_r) @ Vh_r).T
        return W_rec.to(dtype=orig_dtype, device=device)
    
    # Split singular values symmetrically
    sqrt_S = torch.sqrt(S_r)
    A = U_r @ torch.diag(sqrt_S)
    B = torch.diag(sqrt_S) @ Vh_r
    
    # Convert back to original dtype
    A = A.to(dtype=orig_dtype, device=device)
    B = B.to(dtype=orig_dtype, device=device)
    
    return A, B


def decompose_linear_to_svd(
    W: torch.tensor,
    r: int,
    reverse: bool = False,
    return_full: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Decomposes a torch.nn.Linear layer into two LoRA-style matrices using truncated SVD.
    Works safely for bfloat16 weights (internally converts to float32 for SVD).

    Args:
        linear_layer (nn.Linear): The original linear layer to decompose.
        r (int): Rank for decomposition (r < min(in_features, out_features))
        reverse (bool): If True, use the complementary singular values.
        return_full (bool): If True, return the reconstructed matrix instead of factors.

    Returns:
        (A, B): Low-rank factors such that A @ B ≈ W
                Both returned in the same dtype as the original layer (e.g. bfloat16).
    """
    # Get weight
    orig_dtype = W.dtype
    device = W.device

    # Convert to float32 for SVD (since bfloat16 not supported)
    W_32 = W.to(torch.float32)

    # Perform SVD on transposed weight (shape: [in_features, out_features])
    U, S, Vh = torch.linalg.svd(W_32.T, full_matrices=False)

    # Truncate to rank-r
    if not reverse:
        U_r = U[:, :r]
        S_r = S[:r]
        Vh_r = Vh[:r, :]
    else:
        U_r = U[:, r:]
        S_r = S[r:]
        Vh_r = Vh[r:, :]

    # Reconstruct or form low-rank factors
    if return_full:
        W_rec = (U_r @ torch.diag(S_r) @ Vh_r).T
        return W_rec.to(dtype=orig_dtype, device=device)

    A = U_r @ torch.diag(torch.sqrt(S_r))
    B = torch.diag(torch.sqrt(S_r)) @ Vh_r

    # Convert back to original dtype (e.g. bfloat16)
    A = A.to(dtype=orig_dtype, device=device)
    B = B.to(dtype=orig_dtype, device=device)

    return A, B


class Identity(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()

    def forward(self, x, *args, **kargs):
        return x
    
    def enable_gradient_checkpointing(self,cpu_offload=False):
        self.gradient_checkpointing = True

    def disable_gradient_checkpointing(self, cpu_offload=False):
        self.gradient_checkpointing = False
    
class IdentityD(nn.Module):
    def __init__(self, hidden_size, mlp_ratio, num_heads):
        super().__init__()

    def forward(self, img, txt, vec, pe, txt_attention_mask= None):
        return img, txt
    
    def enable_gradient_checkpointing(self,cpu_offload=False):
        self.gradient_checkpointing = True

    def disable_gradient_checkpointing(self, cpu_offload=False):
        self.gradient_checkpointing = False





class SingleStreamBlockPruned(nn.Module):
    """
    A DiT block with parallel linear layers as described in
    https://arxiv.org/abs/2302.05442 and adapted modulation interface.
    """

    def __init__(
        self,
        block, 
        rank_attn=256,
        rank_mlp=512,
        rank_mlp2=512,
        rank_mod=256,
        flag_attn=False,
        flag_mlp=False,
        flag_mlp2=False,
        flag_mod=False
    ):
        super().__init__()
        self.hidden_dim = block.hidden_size
        hidden_size = block.hidden_size
        self.num_heads = block.num_heads
        qk_scale = block.qk_scale
        mlp_ratio = block.mlp_ratio
        head_dim = hidden_size // self.num_heads
        self.scale = block.scale
        self.flag_attn = flag_attn
        self.flag_mlp = flag_mlp
        self.flag_mlp2 = flag_mlp2
        self.flag_mod = flag_mod
        self.mlp_hidden_dim = block.mlp_hidden_dim
        
        # qkv and mlp_in

        W = block.linear1.weight.data  # shape: [(3*hidden_size + mlp_hidden_dim), in_features]
        b = block.linear1.bias.data if block.linear1.bias is not None else None
        qkv_W, mlp_W = torch.split(W,[3 * hidden_size, self.mlp_hidden_dim],dim=0)
        if b is not None:
            qkv_b, mlp_b = torch.split(b,[3 * hidden_size,self.mlp_hidden_dim],dim=0)
        else:
            qkv_b = mlp_b = None
        
        
        
        if flag_attn:
            q_W, k_W, v_W = torch.split(qkv_W, [hidden_size, hidden_size, hidden_size], dim=0)
            if qkv_b is not None:
                q_b, k_b, v_b = torch.split(qkv_b, [hidden_size, hidden_size, hidden_size], dim=0)
            else:
                q_b = k_b = v_b = None
            # --- Q ---
            linear_q_A, linear_q_B = decompose_linear_to_svd(q_W, r=rank_attn)
            self.linear_q_A, self.linear_q_B = nn.Parameter(linear_q_A), nn.Parameter(linear_q_B)
            self.bias_q = nn.Parameter(deepcopy(q_b))
            
            # ---K ---
            linear_k_A, linear_k_B = decompose_linear_to_svd(k_W, r=rank_attn)
            self.linear_k_A, self.linear_k_B = nn.Parameter(linear_k_A), nn.Parameter(linear_k_B)
            self.bias_k = nn.Parameter(deepcopy(k_b))
            
            # --- V ---
            linear_v_A, linear_v_B = decompose_linear_to_svd(v_W, r=rank_attn)
            self.linear_v_A, self.linear_v_B = nn.Parameter(linear_v_A), nn.Parameter(linear_v_B)
            self.bias_v = nn.Parameter(deepcopy(v_b))
        
        elif flag_mlp: 
            self.qkv_W = nn.Parameter(qkv_W)
            self.qkv_b =nn.Parameter(qkv_b)
        
        
        # --- MLP ---
        if flag_mlp:
            linear_mlp_A, linear_mlp_B = decompose_linear_to_svd(mlp_W, r=rank_mlp)
            self.linear_mlp_A, self.linear_mlp_B = nn.Parameter(linear_mlp_A), nn.Parameter(linear_mlp_B)
            self.bias_mlp = nn.Parameter(deepcopy(mlp_b))
        elif flag_attn: 
            self.mlp_W = nn.Parameter(mlp_W)
            self.mlp_b = nn.Parameter(mlp_b)
        
        if not flag_mlp and not flag_attn: 
            self.linear1 = block.linear1
        
        # proj and mlp_out
        if flag_mlp2:
            W2 = block.linear2.weight.data
            b2 = block.linear2.bias.data if block.linear2.bias is not None else None
            linear_mlp2_A, linear_mlp2_B = decompose_linear_to_svd(W2, r=rank_mlp2)
            self.linear_mlp2_A, self.linear_mlp2_B = nn.Parameter(linear_mlp2_A), nn.Parameter(linear_mlp2_B)
            self.bias_mlp2 = nn.Parameter(deepcopy(b2))
        else:
            self.linear2 = block.linear2
        #self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size)

        self.norm = block.norm

        self.hidden_size = block.hidden_size
        self.pre_norm = block.pre_norm

        self.mlp_act = block.mlp_act
        
        if flag_mod:
            self.modulation = ModulationSVD(block.modulation, rank_mod)
        else: 
            self.modulation = block.modulation

        self.gradient_checkpointing = block.gradient_checkpointing
        self.cpu_offload_checkpointing = block.cpu_offload_checkpointing

    def enable_gradient_checkpointing(self, cpu_offload: bool = False):
        self.gradient_checkpointing = True
        self.cpu_offload_checkpointing = cpu_offload

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False
        self.cpu_offload_checkpointing = False

    def _forward(self, x: Tensor, vec: Tensor, pe: Tensor, txt_attention_mask: Optional[Tensor] = None) -> Tensor:
        mod, _ = self.modulation(vec)
        x_mod = (1 + mod.scale) * self.pre_norm(x) + mod.shift
        if self.flag_mlp:
            mlp =  (x_mod @ self.linear_mlp_A @ self.linear_mlp_B) + self.bias_mlp
        elif self.flag_attn: 
            mlp = x_mod @ self.mlp_W.T + self.mlp_b
            
        
        if self.flag_attn:
            q = (x_mod @ self.linear_q_A @ self.linear_q_B) + self.bias_q
            v = (x_mod @ self.linear_v_A @ self.linear_v_B) + self.bias_v
            k = (x_mod @ self.linear_k_A @ self.linear_k_B) + self.bias_k
            q = rearrange(q, "B L (H D) -> B H L D", H=self.num_heads)
            k = rearrange(k, "B L (H D) -> B H L D", H=self.num_heads)
            v = rearrange(v, "B L (H D) -> B H L D", H=self.num_heads)
        elif self.flag_mlp:
            qkv = x_mod @ self.qkv_W.T + self.qkv_b
            q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        
        if not self.flag_attn and not self.flag_mlp:
            qkv, mlp = torch.split(self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)
            q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        #q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)

        # make attention mask if not None
        attn_mask = None
        if txt_attention_mask is not None:
            # F.scaled_dot_product_attention expects attn_mask to be bool for binary mask
            attn_mask = txt_attention_mask.to(torch.bool)  # b, seq_len
            attn_mask = torch.cat(
                (
                    attn_mask,
                    torch.ones(
                        attn_mask.shape[0], x.shape[1] - txt_attention_mask.shape[1], device=attn_mask.device, dtype=torch.bool
                    ),
                ),
                dim=1,
            )  # b, seq_len + img_len = x_len

            # broadcast attn_mask to all heads
            attn_mask = attn_mask[:, None, None, :].expand(-1, q.shape[1], q.shape[2], -1)

        # compute attention
        attn = attention(q, k, v, pe=pe, attn_mask=attn_mask)

        # compute activation in mlp stream, cat again and run second linear layer
        #output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
        if self.flag_mlp2: 
            xfinal = torch.cat((attn, self.mlp_act(mlp)), 2)
            output = (xfinal @ self.linear_mlp2_A @ self.linear_mlp2_B) + self.bias_mlp2
        else: 
            output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
        return x + mod.gate * output

    def forward(self, x: Tensor, vec: Tensor, pe: Tensor, txt_attention_mask: Optional[Tensor] = None) -> Tensor:
        if self.training and self.gradient_checkpointing:
            if not self.cpu_offload_checkpointing:
                return checkpoint(self._forward, x, vec, pe, txt_attention_mask, use_reentrant=False)

            # cpu offload checkpointing

            def create_custom_forward(func):
                def custom_forward(*inputs):
                    cuda_inputs = to_cuda(inputs)
                    outputs = func(*cuda_inputs)
                    return to_cpu(outputs)

                return custom_forward

            return torch.utils.checkpoint.checkpoint(
                create_custom_forward(self._forward), x, vec, pe, txt_attention_mask, use_reentrant=False
            )
        else:
            return self._forward(x, vec, pe, txt_attention_mask)


class SelfAttention_(nn.Module):
    def __init__(self, attn_block, rank_attn):
        super().__init__()
        self.num_heads = attn_block.num_heads
        dim = attn_block.dim
        head_dim = attn_block.dim // attn_block.num_heads
        qkv_W = attn_block.qkv.weight.data  # shape: [(3*hidden_size + mlp_hidden_dim), in_features]
        qkv_b = attn_block.qkv.bias.data if attn_block.qkv.bias is not None else None

        
        q_W, k_W, v_W = torch.split(qkv_W, [attn_block.dim, attn_block.dim, attn_block.dim], dim=0)
        q_b, k_b, v_b = torch.split(qkv_b, [attn_block.dim, attn_block.dim, attn_block.dim], dim=0)
        # --- Q ---
        linear_q_A, linear_q_B = decompose_linear_to_svd(q_W, r=rank_attn)
        self.linear_q_A, self.linear_q_B = nn.Parameter(linear_q_A), nn.Parameter(linear_q_B)
        self.bias_q = nn.Parameter(deepcopy(q_b))
        
        # ---K ---
        linear_k_A, linear_k_B = decompose_linear_to_svd(k_W, r=rank_attn)
        self.linear_k_A, self.linear_k_B = nn.Parameter(linear_k_A), nn.Parameter(linear_k_B)
        self.bias_k = nn.Parameter(deepcopy(k_b))
        
        # --- V ---
        linear_v_A, linear_v_B = decompose_linear_to_svd(v_W, r=rank_attn)
        self.linear_v_A, self.linear_v_B = nn.Parameter(linear_v_A), nn.Parameter(linear_v_B)
        self.bias_v = nn.Parameter(deepcopy(v_b))
        
        
        self.norm = QKNorm(head_dim)
        self.proj = attn_block.proj

    # this is not called from DoubleStreamBlock/SingleStreamBlock because they uses attention function directly
    def forward(self, x: Tensor, pe: Tensor) -> Tensor:
        q = (x @ self.linear_q_A @ self.linear_q_B) + self.bias_q
        v = (x @ self.linear_v_A @ self.linear_v_B) + self.bias_v
        k = (x @ self.linear_k_A @ self.linear_k_B) + self.bias_k
        
        q = rearrange(q, "B L (H D) -> B H L D", H=self.num_heads)
        k = rearrange(k, "B L (H D) -> B H L D", H=self.num_heads)
        v = rearrange(v, "B L (H D) -> B H L D", H=self.num_heads)
        q, k = self.norm(q, k, v)
        x = attention(q, k, v, pe=pe)
        x = self.proj(x)
        return x


from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F


def to_cuda(x):
    if isinstance(x, torch.Tensor):
        return x.cuda()
    elif isinstance(x, (list, tuple)):
        return [to_cuda(elem) for elem in x]
    elif isinstance(x, dict):
        return {k: to_cuda(v) for k, v in x.items()}
    else:
        return x


def to_cpu(x):
    if isinstance(x, torch.Tensor):
        return x.cpu()
    elif isinstance(x, (list, tuple)):
        return [to_cpu(elem) for elem in x]
    elif isinstance(x, dict):
        return {k: to_cpu(v) for k, v in x.items()}
    else:
        return x

class ModulationSVD(nn.Module):
    def __init__(self, block_modulation: nn.Module, rank_mod: int):
        super().__init__()
        self.is_double = block_modulation.is_double
        self.multiplier = 6 if self.is_double else 3

        # === Extract weights and bias ===
        lin_W = block_modulation.lin.weight.data  # shape: [multiplier * dim, dim]
        lin_b = block_modulation.lin.bias.data if block_modulation.lin.bias is not None else None

        # === Apply provided SVD decomposition ===
        linear_A, linear_B = decompose_linear_to_svd(lin_W, r=rank_mod)

        # === Register low-rank factors as parameters ===
        self.linear_A = nn.Parameter(linear_A)  # [out_features, r]
        self.linear_B = nn.Parameter(linear_B)  # [r, in_features]
        self.bias = nn.Parameter(deepcopy(lin_b)) if lin_b is not None else None

    def forward(self, vec: torch.Tensor):
        """
        vec: [B, dim]
        returns: (ModulationOut, ModulationOut | None)
        """
        # Low-rank linear forward pass: (silu(vec) @ B.T @ A.T)
        x = F.silu(vec)
        out = (x @ self.linear_A @ self.linear_B)
        if self.bias is not None:
            out = out + self.bias

        out = out[:, None, :].chunk(self.multiplier, dim=-1)

        return (
            ModulationOut(*out[:3]),
            ModulationOut(*out[3:]) if self.is_double else None,
        )




class DoubleStreamBlockPruned(nn.Module):
    def __init__(self,
                 block,
                 rank_attn_img=512,
                 rank_attn_txt=512,
                 rank_img_mlp_in=512,
                 rank_img_mlp_out=512,
                 rank_txt_mlp_in=512,
                 rank_txt_mlp_out=512,
                 rank_img_mod= 512,
                 rank_txt_mod=512,
                 flag_img_attn=False,
                 flag_txt_attn=False,
                 flag_img_mlp=False,
                 flag_txt_mlp=False,
                 flag_img_mod=False,
                 flag_txt_mod = False):
        super().__init__()
        self.mlp_ratio = block.mlp_ratio 
        self.hidden_size = block.hidden_size
        self.qkv_bias = block.qkv_bias
        hidden_size = block.hidden_size
        mlp_hidden_dim = int(hidden_size * block.mlp_ratio)
        self.num_heads = block.num_heads
        self.hidden_size = block.hidden_size
        self.flag_img_attn = flag_img_attn
        self.flag_txt_attn = flag_txt_attn
        self.flag_img_mlp = flag_img_mlp
        self.flag_txt_mlp = flag_txt_mlp
        self.flag_img_mod = flag_img_mod
        self.flag_txt_mod = flag_txt_mod
        self.img_norm1 = block.img_norm1

        if flag_img_mod:
            self.img_mod = ModulationSVD(block.img_mod, rank_img_mod)
        else: 
            self.img_mod = block.img_mod
        
        

        if flag_img_attn:
            self.img_attn = SelfAttention_(block.img_attn, rank_attn_img)
        else: 
            self.img_attn = block.img_attn

        self.img_norm2 = block.img_norm2
     
        if flag_img_mlp:
            img_mlp_l1, gelu, img_mlp_l2 = block.img_mlp
            # extract weights and biases
            img_mlp_W1 = img_mlp_l1.weight.data
            img_mlp_b1 = img_mlp_l1.bias.data if img_mlp_l1.bias is not None else None
            img_mlp_W2 = img_mlp_l2.weight.data
            img_mlp_b2 = img_mlp_l2.bias.data if img_mlp_l2.bias is not None else None

            # --- Apply SVD decomposition ---
            img_mlp_A1, img_mlp_B1 = decompose_linear_to_svd(img_mlp_W1, r=rank_img_mlp_in)
            img_mlp_A2, img_mlp_B2 = decompose_linear_to_svd(img_mlp_W2, r=rank_img_mlp_out)

            # --- Register decomposed parameters ---
            self.img_mlp_linear1_A = nn.Parameter(img_mlp_A1)
            self.img_mlp_linear1_B = nn.Parameter(img_mlp_B1)
            self.img_mlp_bias1 = nn.Parameter(deepcopy(img_mlp_b1)) if img_mlp_b1 is not None else None

            self.img_mlp_linear2_A = nn.Parameter(img_mlp_A2)
            self.img_mlp_linear2_B = nn.Parameter(img_mlp_B2)
            self.img_mlp_bias2 = nn.Parameter(deepcopy(img_mlp_b2)) if img_mlp_b2 is not None else None

            # store GELU activation
            self.img_mlp_act = deepcopy(gelu)    
        else:
            self.img_mlp = block.img_mlp
        

        if self.flag_txt_mod:
            self.txt_mod = ModulationSVD(block.txt_mod, rank_txt_mod)
        else:
            self.txt_mod = block.txt_mod
        self.txt_norm1 = block.txt_norm1
        
       
        if flag_txt_attn:
            self.txt_attn = SelfAttention_(block.txt_attn, rank_attn_txt)
        else:
            self.txt_attn = block.txt_attn

        self.txt_norm2 = block.txt_norm2
        
 
        if flag_txt_mlp:
            txt_mlp_l1, gelu, txt_mlp_l2 = block.txt_mlp

            # extract weights and biases
            txt_mlp_W1 = txt_mlp_l1.weight.data
            txt_mlp_b1 = txt_mlp_l1.bias.data if txt_mlp_l1.bias is not None else None
            txt_mlp_W2 = txt_mlp_l2.weight.data
            txt_mlp_b2 = txt_mlp_l2.bias.data if txt_mlp_l2.bias is not None else None

            # --- Apply SVD decomposition ---
            txt_mlp_A1, txt_mlp_B1 = decompose_linear_to_svd(txt_mlp_W1, r=rank_txt_mlp_in)
            txt_mlp_A2, txt_mlp_B2 = decompose_linear_to_svd(txt_mlp_W2, r=rank_txt_mlp_out)

            # --- Register decomposed parameters ---
            self.txt_mlp_linear1_A = nn.Parameter(txt_mlp_A1)
            self.txt_mlp_linear1_B = nn.Parameter(txt_mlp_B1)
            self.txt_mlp_bias1 = nn.Parameter(deepcopy(txt_mlp_b1)) if txt_mlp_b1 is not None else None

            self.txt_mlp_linear2_A = nn.Parameter(txt_mlp_A2)
            self.txt_mlp_linear2_B = nn.Parameter(txt_mlp_B2)
            self.txt_mlp_bias2 = nn.Parameter(deepcopy(txt_mlp_b2)) if txt_mlp_b2 is not None else None

            # store GELU activation
            self.txt_mlp_act = deepcopy(gelu)

            
        else:
            self.txt_mlp = block.txt_mlp

        self.gradient_checkpointing = False
        self.cpu_offload_checkpointing = False

    def enable_gradient_checkpointing(self, cpu_offload: bool = False):
        self.gradient_checkpointing = True
        self.cpu_offload_checkpointing = cpu_offload

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False
        self.cpu_offload_checkpointing = False

    def _forward(
        self, img: Tensor, txt: Tensor, vec: Tensor, pe: Tensor, txt_attention_mask: Optional[Tensor] = None
    ) -> tuple[Tensor, Tensor]:
        img_mod1, img_mod2 = self.img_mod(vec)
        txt_mod1, txt_mod2 = self.txt_mod(vec)

        # prepare image for attention
        img_modulated = self.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        if self.flag_img_attn:
            q = (img_modulated @ self.img_attn.linear_q_A @ self.img_attn.linear_q_B) + self.img_attn.bias_q
            v = (img_modulated @ self.img_attn.linear_v_A @ self.img_attn.linear_v_B) + self.img_attn.bias_v
            k = (img_modulated @ self.img_attn.linear_k_A @ self.img_attn.linear_k_B) + self.img_attn.bias_k
            
            img_q = rearrange(q, "B L (H D) -> B H L D", H=self.num_heads)
            img_k = rearrange(k, "B L (H D) -> B H L D", H=self.num_heads)
            img_v = rearrange(v, "B L (H D) -> B H L D", H=self.num_heads)
        else:
            img_qkv = self.img_attn.qkv(img_modulated)
            img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

        # prepare txt for attention
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        if self.flag_txt_attn:
            q = (txt_modulated @ self.txt_attn.linear_q_A @ self.txt_attn.linear_q_B) + self.txt_attn.bias_q
            v = (txt_modulated @ self.txt_attn.linear_v_A @ self.txt_attn.linear_v_B) + self.txt_attn.bias_v
            k = (txt_modulated @ self.txt_attn.linear_k_A @ self.txt_attn.linear_k_B) + self.txt_attn.bias_k
            
            txt_q = rearrange(q, "B L (H D) -> B H L D", H=self.num_heads)
            txt_k = rearrange(k, "B L (H D) -> B H L D", H=self.num_heads)
            txt_v = rearrange(v, "B L (H D) -> B H L D", H=self.num_heads)
        else:
            txt_qkv = self.txt_attn.qkv(txt_modulated)
            txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

        # run actual attention
        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)

        # make attention mask if not None
        attn_mask = None
        if txt_attention_mask is not None:
            # F.scaled_dot_product_attention expects attn_mask to be bool for binary mask
            attn_mask = txt_attention_mask.to(torch.bool)  # b, seq_len
            attn_mask = torch.cat(
                (attn_mask, torch.ones(attn_mask.shape[0], img.shape[1], device=attn_mask.device, dtype=torch.bool)), dim=1
            )  # b, seq_len + img_len

            # broadcast attn_mask to all heads
            attn_mask = attn_mask[:, None, None, :].expand(-1, q.shape[1], q.shape[2], -1)

        attn = attention(q, k, v, pe=pe, attn_mask=attn_mask)
        txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1] :]

        # calculate the img blocks
        img = img + img_mod1.gate * self.img_attn.proj(img_attn)
        
        if self.flag_img_mlp:
            img_in = (1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift
            img_in = (img_in @ self.img_mlp_linear1_A @ self.img_mlp_linear1_B) + self.img_mlp_bias1
            img_in = self.img_mlp_act(img_in)

            # --- Second Linear ---
            img_in = (img_in @ self.img_mlp_linear2_A @ self.img_mlp_linear2_B) +  self.img_mlp_bias2
            img = img + img_mod2.gate * img_in
            
        else:
            img = img + img_mod2.gate * self.img_mlp((1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift)

        # calculate the txt blocks
        txt = txt + txt_mod1.gate * self.txt_attn.proj(txt_attn)
        if self.flag_txt_mlp:
            txt_in = (1 + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift
            txt_in = (txt_in @ self.txt_mlp_linear1_A @ self.txt_mlp_linear1_B) + self.txt_mlp_bias1
            txt_in = self.txt_mlp_act(txt_in)
            
            txt_in = (txt_in @ self.txt_mlp_linear2_A @ self.txt_mlp_linear2_B) +  self.txt_mlp_bias2
            txt = txt + txt_mod2.gate * txt_in
        else:
            txt = txt + txt_mod2.gate * self.txt_mlp((1 + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift)
        return img, txt

    def forward(
        self, img: Tensor, txt: Tensor, vec: Tensor, pe: Tensor, txt_attention_mask: Optional[Tensor] = None
    ) -> tuple[Tensor, Tensor]:
        if self.training and self.gradient_checkpointing:
            if not self.cpu_offload_checkpointing:
                return checkpoint(self._forward, img, txt, vec, pe, txt_attention_mask, use_reentrant=False)
            # cpu offload checkpointing

            def create_custom_forward(func):
                def custom_forward(*inputs):
                    cuda_inputs = to_cuda(inputs)
                    outputs = func(*cuda_inputs)
                    return to_cpu(outputs)

                return custom_forward

            return torch.utils.checkpoint.checkpoint(
                create_custom_forward(self._forward), img, txt, vec, pe, txt_attention_mask, use_reentrant=False
            )

        else:
            return self._forward(img, txt, vec, pe, txt_attention_mask)