import math
from copy import deepcopy
from typing import List, Literal, Optional, Union, Tuple

import torch
import torch as th
import torch.nn as nn

import torch.nn.functional as F
from library.flux_models import (
    DoubleStreamBlock,
    Modulation,
    ModulationOut,
    QKNorm,
    SingleStreamBlock,
    attention,
)

class SVDLinear(nn.Module):
    def __init__(self, U: torch.Tensor, S: torch.Tensor, Vh: torch.Tensor, bias: Optional[torch.Tensor], sigma_fuse: Literal["UV", "U", "V"] = "UV"):
        '''
        **__Args__:**
            U: Left Singular Vectors after rank truncation, which is shape of [rank, out_features]
            S: Diagonal Matrix of singular values, which is shape of [rank, rank]
            Vh: Right Singular Vectors after rank truncation, which is shape of [in_features, rank]
            bias: bias
        '''
        super(SVDLinear, self).__init__()
        
        in_features = Vh.shape[1]
        out_features = U.shape[0]
        hidden_size = S.shape[0]

        self.InLinear = nn.Linear(in_features=in_features, out_features=hidden_size, bias=False)
        self.OutLinear = nn.Linear(in_features=hidden_size, out_features=out_features, bias=True if bias is not None else False)

        if bias is not None:
            self.OutLinear.bias.data = bias
        
        if sigma_fuse == "UV":
            self.InLinear.weight.data = Vh.mul(S.sqrt().view(-1, 1)).contiguous()
            self.OutLinear.weight.data = U.mul(S.sqrt()).contiguous()
        elif sigma_fuse == "U":
            self.InLinear.weight.data = Vh.contiguous()
            self.OutLinear.weight.data = U.mul(S).contiguous()
        elif sigma_fuse == "V":
            self.InLinear.weight.data = Vh.mul(S.view(-1, 1)).contiguous()
        else:
            raise ValueError(f"value of sigma_fuse {sigma_fuse} not support")
    
    def forward(self, x: torch.Tensor):
        output = self.OutLinear(self.InLinear(x))
        return output
    
    
# class GRASPLayer(nn.Module):
#     def __init__(self, U: torch.Tensor, S: torch.Tensor, Vh: torch.Tensor, bias: Optional[torch.Tensor], compression_ratio: Optional[float]):
#         super(GRASPLayer, self).__init__()
#         self.U = nn.Parameter(U.clone().detach().requires_grad_(False))
#         self.S = nn.Parameter(S.clone().detach().requires_grad_(True))
#         self.Vh = nn.Parameter(Vh.clone().detach().requires_grad_(False))

#         self.in_features = self.Vh.shape[1]
#         self.out_features = self.U.shape[0]

#         self.bias =  nn.Parameter(bias.clone().detach().requires_grad_(False))
#         self.compression_ratio = compression_ratio

#     def forward(self, x: torch.Tensor):
#         b, s, d = x.shape
#         sigma = torch.diag(self.S)
#         W_reconstructed =  torch.mm(self.U, torch.mm(sigma, self.Vh))
#         return (torch.mm(x.reshape(-1, x.shape[-1]), W_reconstructed.t()) + self.bias).view(b, s, -1)
    
    
    


import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional
from copy import deepcopy
from einops import rearrange
from torch.utils.checkpoint import checkpoint
import sys 

def grasp_decompose_linear_to_svd(
        linear_layer: nn.Linear,
        r: int,
        reverse: bool = False,
        return_full: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Decomposes a torch.nn.Linear layer into two LoRA-style matrices using truncated SVD.

    Args:
        linear_layer (nn.Linear): The original linear layer to decompose.
        r (int): Rank for decomposition (r < min(in_features, out_features))

    Returns:
        A (nn.Parameter): Left matrix (in_features x r)
        B (nn.Parameter): Right matrix (r x out_features)
    """
    # Get original weight (shape: out_features x in_features)
    W = linear_layer # shape: [out_features, in_features]

    # Perform full SVD on the transposed weight to get shape (in_features x out_features)
    # This lets us get A (in_features x r) and B (r x out_features)
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)

    # Truncate to rank-r
    if not reverse:
        U_r = U[:, :r]  # shape: [in_features, r]
        S_r = S[:r]  # shape: [r]
        Vh_r = Vh[:r, :]  # shape: [r, out_features]
    else:
        U_r = U[:, r:]  # shape: [in_features, full - r]
        S_r = S[r:]  # shape: [full - r]
        Vh_r = Vh[r:, :]  # shape: [full - r, out_features]

  
    
    return  U_r, S_r, Vh_r
# 1. Die von Ihnen bereitgestellte GRASPLayer-Klasse
# (Diese Klasse ist korrekt und unverändert)
class GRASPLayer(nn.Module):
    def __init__(self, U: torch.Tensor, S: torch.Tensor, Vh: torch.Tensor, bias: Optional[torch.Tensor], compression_ratio: Optional[float] = None):
        super(GRASPLayer, self).__init__()
        self.U = nn.Parameter(U.clone().detach().requires_grad_(False))
        self.S = nn.Parameter(S.clone().detach().requires_grad_(True))
        self.Vh = nn.Parameter(Vh.clone().detach().requires_grad_(False))

        self.in_features = self.Vh.shape[1]
        self.out_features = self.U.shape[0]

        if bias is not None:
            self.bias =  nn.Parameter(bias.clone().detach().requires_grad_(False))
        else:
            self.register_parameter('bias', None)
            
        self.compression_ratio = compression_ratio # Dieser Wert wird später von dynamic_svd_selection verwendet

    def forward(self, x: torch.Tensor):
        # Die Forward-Methode rekonstruiert das volle Gewicht
        # (oder sollte es zumindest, um gradientenfähig zu sein)
        is_3d = x.dim() == 3
        if is_3d:
            b, s, d = x.shape
            x_reshaped = x.reshape(-1, d)
        else:
            x_reshaped = x

        sigma = torch.diag(self.S)
        W_reconstructed =  torch.mm(self.U, torch.mm(sigma, self.Vh))
        
        output = torch.mm(x_reshaped, W_reconstructed.t())
        if self.bias is not None:
            output += self.bias
        
        if is_3d:
            return output.view(b, s, -1)
        return output

class ModulationSVD(nn.Module):
    def __init__(self, block_modulation: nn.Module, rank_mod: int=0):
        super().__init__()
        self.is_double = block_modulation.is_double
        self.multiplier = 6 if self.is_double else 3

        # === Extract weights and bias ===
        lin_W = block_modulation.lin.weight.data  # Form: [out_features, in_features]
        lin_b = block_modulation.lin.bias.data if block_modulation.lin.bias is not None else None

        # === Führe SVD mit vollem Rang durch ===
        # (Wir ignorieren rank_mod hier, da GRASPLayer die vollen Matrizen erwartet)
        U, S, Vh = torch.linalg.svd(lin_W.float(), full_matrices=False)

        # === Erstelle den GRASPLayer ===
        # Dieser Layer kapselt jetzt U, S, Vh und den Bias
        self.lin_grasp = GRASPLayer(U, S, Vh, lin_b)

        # Die alten nn.Parameter (linear_A, linear_B, bias) sind
        # jetzt alle im self.lin_grasp enthalten.

    def forward(self, vec: torch.Tensor):
        """
        vec: [B, dim_in]
        returns: (ModulationOut, ModulationOut | None)
        """
        # 1. Silu-Aktivierung auf den Input
        x = F.silu(vec) # Form: [B, dim_in]
        
        # 2. Aufruf des GRASPLayer. 
        #    Dieser berechnet intern: (x @ W_reconstructed.T + bias)
        #    Output-Form: [B, out_features] (wobei out_features = multiplier * dim)
        out = self.lin_grasp(x)

        # (Anmerkung: Ihr alter Code 'out = (x @ self.linear_A @ self.linear_B)'
        #  war mathematisch wahrscheinlich nicht korrekt, da die Dimensionen
        #  nicht passten. Diese neue Version 'self.lin_grasp(x)' 
        #  implementiert die Standard-Linear-Layer-Operation.)

        # 3. Aufteilung in Chunks (unverändert)
        out = out[:, None, :].chunk(self.multiplier, dim=-1)

        return (
            ModulationOut(*out[:3]),
            ModulationOut(*out[3:]) if self.is_double else None,
        )
        
        
class SingleStreamBlockGRASP(nn.Module):
    """
    Ein DiT-Block, der GRASPLayer mit vollem Rang initialisiert.
    """

    def __init__(
        self,
        block, 
        rank_attn=256,   # Diese Rank-Argumente sind jetzt für __init__ irrelevant
        rank_mlp=512,    # Sie werden erst in dynamic_svd_selection wichtig
        rank_mlp2=512,   # (z.B. über ein compression_ratio)
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
        # ... (andere Attribute) ...
        self.scale = block.scale
        self.flag_attn = flag_attn
        self.flag_mlp = flag_mlp
        self.flag_mlp2 = flag_mlp2
        self.flag_mod = flag_mod
        self.mlp_hidden_dim = block.mlp_hidden_dim
        
        # qkv und mlp_in
        W = block.linear1.weight.data
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

            # --- Q (GRASPLayer) ---
            # Führe SVD durch und übergebe die VOLLEN Matrizen
            U_q, S_q, Vh_q = torch.linalg.svd(q_W.float(), full_matrices=False)
            self.q_proj = GRASPLayer(U_q, S_q, Vh_q, q_b)
            
            # --- K (GRASPLayer) ---
            U_k, S_k, Vh_k = torch.linalg.svd(k_W.float(), full_matrices=False)
            self.k_proj = GRASPLayer(U_k, S_k, Vh_k, k_b)
            
            # --- V (GRASPLayer) ---
            U_v, S_v, Vh_v = torch.linalg.svd(v_W.float(), full_matrices=False)
            self.v_proj = GRASPLayer(U_v, S_v, Vh_v, v_b)
        
        elif flag_mlp: 
            self.qkv_W = nn.Parameter(qkv_W)
            self.qkv_b = nn.Parameter(qkv_b) if qkv_b is not None else None
        
        
        # --- MLP ---
        if flag_mlp:
            # --- MLP_in (GRASPLayer) ---
            U_mlp, S_mlp, Vh_mlp = torch.linalg.svd(mlp_W.float(), full_matrices=False)
            self.mlp_in = GRASPLayer(U_mlp, S_mlp, Vh_mlp, mlp_b)

        elif flag_attn: 
            self.mlp_W = nn.Parameter(mlp_W)
            self.mlp_b = nn.Parameter(mlp_b) if mlp_b is not None else None
        
        if not flag_mlp and not flag_attn: 
            self.linear1 = block.linear1
        
        # proj and mlp_out
        if flag_mlp2:
            W2 = block.linear2.weight.data
            b2 = block.linear2.bias.data if block.linear2.bias is not None else None
            
            # --- MLP_out (GRASPLayer) ---
            U_mlp2, S_mlp2, Vh_mlp2 = torch.linalg.svd(W2.float(), full_matrices=False)
            self.mlp_out = GRASPLayer(U_mlp2, S_mlp2, Vh_mlp2, b2)
        else:
            self.linear2 = block.linear2

        self.norm = block.norm
        self.hidden_size = block.hidden_size
        self.pre_norm = block.pre_norm
        self.mlp_act = block.mlp_act
        
        if flag_mod:
            self.modulation = ModulationSVD(block.modulation)
        else: 
            self.modulation = block.modulation

        self.gradient_checkpointing = block.gradient_checkpointing
        self.cpu_offload_checkpointing = block.cpu_offload_checkpointing

    # ... (Rest der Klasse: enable/disable_gradient_checkpointing, _forward, forward) ...
    # ... (Diese Methoden sind identisch mit der vorherigen Version, da
    # ...  sie einfach die .forward() des GRASPLayer aufrufen) ...

    def enable_gradient_checkpointing(self, cpu_offload: bool = False):
        self.gradient_checkpointing = True
        self.cpu_offload_checkpointing = cpu_offload

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False
        self.cpu_offload_checkpointing = False

    def _forward(self, x: Tensor, vec: Tensor, pe: Tensor, txt_attention_mask: Optional[Tensor] = None) -> Tensor:
        mod, _ = self.modulation(vec)
        x_mod = (1 + mod.scale) * self.pre_norm(x) + mod.shift

        # --- MLP Forward ---
        if self.flag_mlp:
            mlp = self.mlp_in(x_mod) # Aufruf des GRASPLayer
        elif self.flag_attn: 
            mlp = torch.mm(x_mod.reshape(-1, x_mod.shape[-1]), self.mlp_W.t())
            if self.mlp_b is not None:
                mlp += self.mlp_b
            mlp = mlp.view(x_mod.shape[0], x_mod.shape[1], -1)
        
        # --- QKV Forward ---
        if self.flag_attn:
            q = self.q_proj(x_mod) # Aufruf des GRASPLayer
            k = self.k_proj(x_mod) # Aufruf des GRASPLayer
            v = self.v_proj(x_mod) # Aufruf des GRASPLayer
            q = rearrange(q, "B L (H D) -> B H L D", H=self.num_heads)
            k = rearrange(k, "B L (H D) -> B H L D", H=self.num_heads)
            v = rearrange(v, "B L (H D) -> B H L D", H=self.num_heads)
        elif self.flag_mlp:
            qkv = torch.mm(x_mod.reshape(-1, x_mod.shape[-1]), self.qkv_W.t())
            if self.qkv_b is not None:
                qkv += self.qkv_b
            qkv = qkv.view(x_mod.shape[0], x_mod.shape[1], -1)
            q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        
        if not self.flag_attn and not self.flag_mlp:
            qkv, mlp = torch.split(self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)
            q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        
        q, k = self.norm(q, k, v)

        # --- Attention-Maske und -Berechnung (unverändert) ---
        attn_mask = None
        if txt_attention_mask is not None:
            attn_mask = txt_attention_mask.to(torch.bool)
            attn_mask = torch.cat(
                (
                    attn_mask,
                    torch.ones(
                        attn_mask.shape[0], x.shape[1] - txt_attention_mask.shape[1], device=attn_mask.device, dtype=torch.bool
                    ),
                ),
                dim=1,
            ) 
            attn_mask = attn_mask[:, None, None, :].expand(-1, q.shape[1], q.shape[2], -1)

        attn = attention(q, k, v, pe=pe, attn_mask=attn_mask) # `attention` muss definiert sein

        # --- MLP_out Forward ---
        if self.flag_mlp2: 
            xfinal = torch.cat((attn, self.mlp_act(mlp)), 2)
            output = self.mlp_out(xfinal) # Aufruf des GRASPLayer
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

class DoubleStreamBlockGRASP(nn.Module):
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
        
        # --- Image Stream ---
        self.img_norm1 = block.img_norm1

        if flag_img_mod:
            # ModulationSVD verwendet bereits GRASPLayer (aus vorigem Chat)
            self.img_mod = ModulationSVD(block.img_mod, 0) # rank ist irrelevant
        else: 
            self.img_mod = block.img_mod
        
        if flag_img_attn:
            # Ersetze SelfAttention_ durch GRASPLayer für Q, K, V
            qkv_W = block.img_attn.qkv.weight.data
            qkv_b = block.img_attn.qkv.bias.data if block.img_attn.qkv.bias is not None else None
            q_W, k_W, v_W = torch.split(qkv_W, [hidden_size, hidden_size, hidden_size], dim=0)
            
            if qkv_b is not None:
                q_b, k_b, v_b = torch.split(qkv_b, [hidden_size, hidden_size, hidden_size], dim=0)
            else:
                q_b = k_b = v_b = None

            U_q, S_q, Vh_q = torch.linalg.svd(q_W.float(), full_matrices=False)
            self.img_q_proj = GRASPLayer(U_q, S_q, Vh_q, q_b)
            
            U_k, S_k, Vh_k = torch.linalg.svd(k_W.float(), full_matrices=False)
            self.img_k_proj = GRASPLayer(U_k, S_k, Vh_k, k_b)
            
            U_v, S_v, Vh_v = torch.linalg.svd(v_W.float(), full_matrices=False)
            self.img_v_proj = GRASPLayer(U_v, S_v, Vh_v, v_b)
            
            self.img_attn_norm = deepcopy(block.img_attn.norm)
            self.img_attn_proj = deepcopy(block.img_attn.proj)
        else: 
            self.img_attn = block.img_attn

        self.img_norm2 = block.img_norm2
     
        if flag_img_mlp:
            img_mlp_l1, gelu, img_mlp_l2 = block.img_mlp
            
            img_mlp_W1 = img_mlp_l1.weight.data
            img_mlp_b1 = img_mlp_l1.bias.data if img_mlp_l1.bias is not None else None
            img_mlp_W2 = img_mlp_l2.weight.data
            img_mlp_b2 = img_mlp_l2.bias.data if img_mlp_l2.bias is not None else None

            # --- Ersetze MLP Layer 1 mit GRASPLayer ---
            U1, S1, Vh1 = torch.linalg.svd(img_mlp_W1.float(), full_matrices=False)
            self.img_mlp_l1 = GRASPLayer(U1, S1, Vh1, img_mlp_b1)

            # --- Ersetze MLP Layer 2 mit GRASPLayer ---
            U2, S2, Vh2 = torch.linalg.svd(img_mlp_W2.float(), full_matrices=False)
            self.img_mlp_l2 = GRASPLayer(U2, S2, Vh2, img_mlp_b2)

            self.img_mlp_act = deepcopy(gelu)    
        else:
            self.img_mlp = block.img_mlp
        
        # --- Text Stream ---
        if self.flag_txt_mod:
            self.txt_mod = ModulationSVD(block.txt_mod, 0) # rank ist irrelevant
        else:
            self.txt_mod = block.txt_mod
        self.txt_norm1 = block.txt_norm1
        
        if flag_txt_attn:
            # Ersetze SelfAttention_ durch GRASPLayer für Q, K, V
            qkv_W = block.txt_attn.qkv.weight.data
            qkv_b = block.txt_attn.qkv.bias.data if block.txt_attn.qkv.bias is not None else None
            q_W, k_W, v_W = torch.split(qkv_W, [hidden_size, hidden_size, hidden_size], dim=0)

            if qkv_b is not None:
                q_b, k_b, v_b = torch.split(qkv_b, [hidden_size, hidden_size, hidden_size], dim=0)
            else:
                q_b = k_b = v_b = None

            U_q, S_q, Vh_q = torch.linalg.svd(q_W.float(), full_matrices=False)
            self.txt_q_proj = GRASPLayer(U_q, S_q, Vh_q, q_b)
            
            U_k, S_k, Vh_k = torch.linalg.svd(k_W.float(), full_matrices=False)
            self.txt_k_proj = GRASPLayer(U_k, S_k, Vh_k, k_b)
            
            U_v, S_v, Vh_v = torch.linalg.svd(v_W.float(), full_matrices=False)
            self.txt_v_proj = GRASPLayer(U_v, S_v, Vh_v, v_b)

            self.txt_attn_norm = deepcopy(block.txt_attn.norm)
            self.txt_attn_proj = deepcopy(block.txt_attn.proj)
        else:
            self.txt_attn = block.txt_attn

        self.txt_norm2 = block.txt_norm2
        
        if flag_txt_mlp:
            txt_mlp_l1, gelu, txt_mlp_l2 = block.txt_mlp

            txt_mlp_W1 = txt_mlp_l1.weight.data
            txt_mlp_b1 = txt_mlp_l1.bias.data if txt_mlp_l1.bias is not None else None
            txt_mlp_W2 = txt_mlp_l2.weight.data
            txt_mlp_b2 = txt_mlp_l2.bias.data if txt_mlp_l2.bias is not None else None

            # --- Ersetze MLP Layer 1 mit GRASPLayer ---
            U1, S1, Vh1 = torch.linalg.svd(txt_mlp_W1.float(), full_matrices=False)
            self.txt_mlp_l1 = GRASPLayer(U1, S1, Vh1, txt_mlp_b1)

            # --- Ersetze MLP Layer 2 mit GRASPLayer ---
            U2, S2, Vh2 = torch.linalg.svd(txt_mlp_W2.float(), full_matrices=False)
            self.txt_mlp_l2 = GRASPLayer(U2, S2, Vh2, txt_mlp_b2)

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
            # Aufruf der neuen GRASPLayer
            q = self.img_q_proj(img_modulated)
            k = self.img_k_proj(img_modulated)
            v = self.img_v_proj(img_modulated)
            
            img_q = rearrange(q, "B L (H D) -> B H L D", H=self.num_heads)
            img_k = rearrange(k, "B L (H D) -> B H L D", H=self.num_heads)
            img_v = rearrange(v, "B L (H D) -> B H L D", H=self.num_heads)
            # Aufruf der kopierten Norm
            img_q, img_k = self.img_attn_norm(img_q, img_k, img_v)
        else:
            img_qkv = self.img_attn.qkv(img_modulated)
            img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
            img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

        # prepare txt for attention
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        if self.flag_txt_attn:
            # Aufruf der neuen GRASPLayer
            q = self.txt_q_proj(txt_modulated)
            k = self.txt_k_proj(txt_modulated)
            v = self.txt_v_proj(txt_modulated)
            
            txt_q = rearrange(q, "B L (H D) -> B H L D", H=self.num_heads)
            txt_k = rearrange(k, "B L (H D) -> B H L D", H=self.num_heads)
            txt_v = rearrange(v, "B L (H D) -> B H L D", H=self.num_heads)
            # Aufruf der kopierten Norm
            txt_q, txt_k = self.txt_attn_norm(txt_q, txt_k, txt_v)
        else:
            txt_qkv = self.txt_attn.qkv(txt_modulated)
            txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
            txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

        # --- run actual attention (unverändert) ---
        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)

        attn_mask = None
        if txt_attention_mask is not None:
            attn_mask = txt_attention_mask.to(torch.bool) 
            attn_mask = torch.cat(
                (attn_mask, torch.ones(attn_mask.shape[0], img.shape[1], device=attn_mask.device, dtype=torch.bool)), dim=1
            )
            attn_mask = attn_mask[:, None, None, :].expand(-1, q.shape[1], q.shape[2], -1)

        attn = attention(q, k, v, pe=pe, attn_mask=attn_mask)
        txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1] :]

        # --- calculate the img blocks ---
        if self.flag_img_attn:
            img = img + img_mod1.gate * self.img_attn_proj(img_attn) # Aufruf des kopierten Proj
        else:
            img = img + img_mod1.gate * self.img_attn.proj(img_attn)
        
        if self.flag_img_mlp:
            img_in = (1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift
            # Aufruf der neuen GRASPLayer
            img_in = self.img_mlp_l1(img_in)
            img_in = self.img_mlp_act(img_in)
            img_in = self.img_mlp_l2(img_in)
            img = img + img_mod2.gate * img_in
        else:
            img = img + img_mod2.gate * self.img_mlp((1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift)

        # --- calculate the txt blocks ---
        if self.flag_txt_attn:
            txt = txt + txt_mod1.gate * self.txt_attn_proj(txt_attn) # Aufruf des kopierten Proj
        else:
            txt = txt + txt_mod1.gate * self.txt_attn.proj(txt_attn)
            
        if self.flag_txt_mlp:
            txt_in = (1 + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift
            # Aufruf der neuen GRASPLayer
            txt_in = self.txt_mlp_l1(txt_in)
            txt_in = self.txt_mlp_act(txt_in)
            txt_in = self.txt_mlp_l2(txt_in)
            txt = txt + txt_mod2.gate * txt_in
        else:
            txt = txt + txt_mod2.gate * self.txt_mlp((1 + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift)
        return img, txt

    def forward(
        self, img: Tensor, txt: Tensor, vec: Tensor, pe: Tensor, txt_attention_mask: Optional[Tensor] = None
    ) -> tuple[Tensor, Tensor]:
        # --- Checkpointing (unverändert) ---
        if self.training and self.gradient_checkpointing:
            if not self.cpu_offload_checkpointing:
                return checkpoint(self._forward, img, txt, vec, pe, txt_attention_mask, use_reentrant=False)
            
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
        
        
        
class ModulationSVDCompressed(nn.Module):
    def __init__(self, block_modulation: nn.Module, rank_mod: int=0):
        super().__init__()
        self.is_double = block_modulation.is_double
        self.multiplier = 6 if self.is_double else 3

        # === Extract weights and bias ===
        lin_W = block_modulation.lin.weight.data  # Form: [out_features, in_features]
        lin_b = block_modulation.lin.bias.data if block_modulation.lin.bias is not None else None

        # === Führe SVD mit vollem Rang durch ===
        # (Wir ignorieren rank_mod hier, da GRASPLayer die vollen Matrizen erwartet)
        U, S, Vh =grasp_decompose_linear_to_svd(lin_W.float(), r=rank_mod)

        # === Erstelle den GRASPLayer ===
        # Dieser Layer kapselt jetzt U, S, Vh und den Bias
        self.lin_grasp = SVDLinear(U, S, Vh, lin_b)

        # Die alten nn.Parameter (linear_A, linear_B, bias) sind
        # jetzt alle im self.lin_grasp enthalten.

    def forward(self, vec: torch.Tensor):
        """
        vec: [B, dim_in]
        returns: (ModulationOut, ModulationOut | None)
        """
        # 1. Silu-Aktivierung auf den Input
        x = F.silu(vec) # Form: [B, dim_in]
        
        # 2. Aufruf des GRASPLayer. 
        #    Dieser berechnet intern: (x @ W_reconstructed.T + bias)
        #    Output-Form: [B, out_features] (wobei out_features = multiplier * dim)
        out = self.lin_grasp(x)

        # (Anmerkung: Ihr alter Code 'out = (x @ self.linear_A @ self.linear_B)'
        #  war mathematisch wahrscheinlich nicht korrekt, da die Dimensionen
        #  nicht passten. Diese neue Version 'self.lin_grasp(x)' 
        #  implementiert die Standard-Linear-Layer-Operation.)

        # 3. Aufteilung in Chunks (unverändert)
        out = out[:, None, :].chunk(self.multiplier, dim=-1)

        return (
            ModulationOut(*out[:3]),
            ModulationOut(*out[3:]) if self.is_double else None,
        )    
      
class SingleStreamBlockGRASPCompressed(nn.Module):
    """
    Ein DiT-Block, der GRASPLayer mit vollem Rang initialisiert.
    """

    def __init__(
        self,
        block, 
        rank_attn=256,   # Diese Rank-Argumente sind jetzt für __init__ irrelevant
        rank_mlp=512,    # Sie werden erst in dynamic_svd_selection wichtig
        rank_mlp2=512,   # (z.B. über ein compression_ratio)
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
        # ... (andere Attribute) ...
        self.scale = block.scale
        self.flag_attn = flag_attn
        self.flag_mlp = flag_mlp
        self.flag_mlp2 = flag_mlp2
        self.flag_mod = flag_mod
        self.mlp_hidden_dim = block.mlp_hidden_dim
        
        # qkv und mlp_in
        W = block.linear1.weight.data
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

            # --- Q (SVDLinear) ---
            U_q, S_q, Vh_q = grasp_decompose_linear_to_svd(q_W.float(), r=rank_attn)
            self.q_proj = SVDLinear(U_q, S_q, Vh_q, q_b)

            # --- K (SVDLinear) ---
            U_k, S_k, Vh_k = grasp_decompose_linear_to_svd(k_W.float(), r=rank_attn)
            self.k_proj = SVDLinear(U_k, S_k, Vh_k, k_b)

            # --- V (SVDLinear) ---
            U_v, S_v, Vh_v = grasp_decompose_linear_to_svd(v_W.float(), r=rank_attn)
            self.v_proj = SVDLinear(U_v, S_v, Vh_v, v_b)

        elif flag_mlp:
            self.qkv_W = nn.Parameter(qkv_W)
            self.qkv_b = nn.Parameter(qkv_b) if qkv_b is not None else None

        # --- MLP_in ---
        if flag_mlp:
            U_mlp, S_mlp, Vh_mlp = grasp_decompose_linear_to_svd(mlp_W.float(), r=rank_mlp)
            self.mlp_in = SVDLinear(U_mlp, S_mlp, Vh_mlp, mlp_b)

        elif flag_attn:
            self.mlp_W = nn.Parameter(mlp_W)
            self.mlp_b = nn.Parameter(mlp_b) if mlp_b is not None else None

        if not flag_mlp and not flag_attn:
            self.linear1 = block.linear1

        # --- MLP_out ---
        if flag_mlp2:
            W2 = block.linear2.weight.data
            b2 = block.linear2.bias.data if block.linear2.bias is not None else None

            U_mlp2, S_mlp2, Vh_mlp2 = grasp_decompose_linear_to_svd(W2.float(), r=rank_mlp2)
            self.mlp_out = SVDLinear(U_mlp2, S_mlp2, Vh_mlp2, b2)
        else:
            self.linear2 = block.linear2


        self.norm = block.norm
        self.hidden_size = block.hidden_size
        self.pre_norm = block.pre_norm
        self.mlp_act = block.mlp_act
        
        if flag_mod:
            self.modulation = ModulationSVDCompressed(block.modulation,rank_mod )
        else: 
            self.modulation = block.modulation

        self.gradient_checkpointing = block.gradient_checkpointing
        self.cpu_offload_checkpointing = block.cpu_offload_checkpointing

    # ... (Rest der Klasse: enable/disable_gradient_checkpointing, _forward, forward) ...
    # ... (Diese Methoden sind identisch mit der vorherigen Version, da
    # ...  sie einfach die .forward() des GRASPLayer aufrufen) ...

    def enable_gradient_checkpointing(self, cpu_offload: bool = False):
        self.gradient_checkpointing = True
        self.cpu_offload_checkpointing = cpu_offload

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False
        self.cpu_offload_checkpointing = False

    def _forward(self, x: Tensor, vec: Tensor, pe: Tensor, txt_attention_mask: Optional[Tensor] = None) -> Tensor:
        mod, _ = self.modulation(vec)
        x_mod = (1 + mod.scale) * self.pre_norm(x) + mod.shift

        # --- MLP Forward ---
        if self.flag_mlp:
            mlp = self.mlp_in(x_mod) # Aufruf des GRASPLayer
        elif self.flag_attn: 
            mlp = torch.mm(x_mod.reshape(-1, x_mod.shape[-1]), self.mlp_W.t())
            if self.mlp_b is not None:
                mlp += self.mlp_b
            mlp = mlp.view(x_mod.shape[0], x_mod.shape[1], -1)
        
        # --- QKV Forward ---
        if self.flag_attn:
            q = self.q_proj(x_mod) # Aufruf des GRASPLayer
            k = self.k_proj(x_mod) # Aufruf des GRASPLayer
            v = self.v_proj(x_mod) # Aufruf des GRASPLayer
            q = rearrange(q, "B L (H D) -> B H L D", H=self.num_heads)
            k = rearrange(k, "B L (H D) -> B H L D", H=self.num_heads)
            v = rearrange(v, "B L (H D) -> B H L D", H=self.num_heads)
        elif self.flag_mlp:
            qkv = torch.mm(x_mod.reshape(-1, x_mod.shape[-1]), self.qkv_W.t())
            if self.qkv_b is not None:
                qkv += self.qkv_b
            qkv = qkv.view(x_mod.shape[0], x_mod.shape[1], -1)
            q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        
        if not self.flag_attn and not self.flag_mlp:
            qkv, mlp = torch.split(self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)
            q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        
        q, k = self.norm(q, k, v)

        # --- Attention-Maske und -Berechnung (unverändert) ---
        attn_mask = None
        if txt_attention_mask is not None:
            attn_mask = txt_attention_mask.to(torch.bool)
            attn_mask = torch.cat(
                (
                    attn_mask,
                    torch.ones(
                        attn_mask.shape[0], x.shape[1] - txt_attention_mask.shape[1], device=attn_mask.device, dtype=torch.bool
                    ),
                ),
                dim=1,
            ) 
            attn_mask = attn_mask[:, None, None, :].expand(-1, q.shape[1], q.shape[2], -1)

        attn = attention(q, k, v, pe=pe, attn_mask=attn_mask) # `attention` muss definiert sein

        # --- MLP_out Forward ---
        if self.flag_mlp2: 
            xfinal = torch.cat((attn, self.mlp_act(mlp)), 2)
            output = self.mlp_out(xfinal) # Aufruf des GRASPLayer
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
        
        

class DoubleStreamBlockGRASPCompressed(nn.Module):
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
        
        # --- Image Stream ---
        self.img_norm1 = block.img_norm1

        if flag_img_mod:
            self.img_mod = ModulationSVDCompressed(block.img_mod, rank_img_mod)  # rank irrelevant
        else: 
            self.img_mod = block.img_mod
        
        if flag_img_attn:
            qkv_W = block.img_attn.qkv.weight.data
            qkv_b = block.img_attn.qkv.bias.data if block.img_attn.qkv.bias is not None else None
            q_W, k_W, v_W = torch.split(qkv_W, [hidden_size, hidden_size, hidden_size], dim=0)
            if qkv_b is not None:
                q_b, k_b, v_b = torch.split(qkv_b, [hidden_size, hidden_size, hidden_size], dim=0)
            else:
                q_b = k_b = v_b = None

            # --- Replace GRASPLayer with SVDLinear ---
            U_q, S_q, Vh_q = grasp_decompose_linear_to_svd(q_W.float(), r=rank_attn_img)
            self.img_q_proj = SVDLinear(U_q, S_q, Vh_q, q_b)

            U_k, S_k, Vh_k = grasp_decompose_linear_to_svd(k_W.float(), r=rank_attn_img)
            self.img_k_proj = SVDLinear(U_k, S_k, Vh_k, k_b)

            U_v, S_v, Vh_v = grasp_decompose_linear_to_svd(v_W.float(), r=rank_attn_img)
            self.img_v_proj = SVDLinear(U_v, S_v, Vh_v, v_b)
            
            self.img_attn_norm = deepcopy(block.img_attn.norm)
            self.img_attn_proj = deepcopy(block.img_attn.proj)
        else: 
            self.img_attn = block.img_attn

        self.img_norm2 = block.img_norm2
     
        if flag_img_mlp:
            img_mlp_l1, gelu, img_mlp_l2 = block.img_mlp
            
            img_mlp_W1 = img_mlp_l1.weight.data
            img_mlp_b1 = img_mlp_l1.bias.data if img_mlp_l1.bias is not None else None
            img_mlp_W2 = img_mlp_l2.weight.data
            img_mlp_b2 = img_mlp_l2.bias.data if img_mlp_l2.bias is not None else None

            U1, S1, Vh1 = grasp_decompose_linear_to_svd(img_mlp_W1.float(), r=rank_img_mlp_in)
            self.img_mlp_l1 = SVDLinear(U1, S1, Vh1, img_mlp_b1)

            U2, S2, Vh2 = grasp_decompose_linear_to_svd(img_mlp_W2.float(), r=rank_img_mlp_out)
            self.img_mlp_l2 = SVDLinear(U2, S2, Vh2, img_mlp_b2)

            self.img_mlp_act = deepcopy(gelu)    
        else:
            self.img_mlp = block.img_mlp
        
        # --- Text Stream ---
        if self.flag_txt_mod:
            self.txt_mod = ModulationSVDCompressed(block.txt_mod, rank_txt_mod)
        else:
            self.txt_mod = block.txt_mod
        self.txt_norm1 = block.txt_norm1
        
        if flag_txt_attn:
            qkv_W = block.txt_attn.qkv.weight.data
            qkv_b = block.txt_attn.qkv.bias.data if block.txt_attn.qkv.bias is not None else None
            q_W, k_W, v_W = torch.split(qkv_W, [hidden_size, hidden_size, hidden_size], dim=0)

            if qkv_b is not None:
                q_b, k_b, v_b = torch.split(qkv_b, [hidden_size, hidden_size, hidden_size], dim=0)
            else:
                q_b = k_b = v_b = None

            U_q, S_q, Vh_q = grasp_decompose_linear_to_svd(q_W.float(), r=rank_attn_txt)
            self.txt_q_proj = SVDLinear(U_q, S_q, Vh_q, q_b)
            
            U_k, S_k, Vh_k = grasp_decompose_linear_to_svd(k_W.float(), r=rank_attn_txt)
            self.txt_k_proj = SVDLinear(U_k, S_k, Vh_k, k_b)
            
            U_v, S_v, Vh_v = grasp_decompose_linear_to_svd(v_W.float(), r=rank_attn_txt)
            self.txt_v_proj = SVDLinear(U_v, S_v, Vh_v, v_b)

            self.txt_attn_norm = deepcopy(block.txt_attn.norm)
            self.txt_attn_proj = deepcopy(block.txt_attn.proj)
        else:
            self.txt_attn = block.txt_attn

        self.txt_norm2 = block.txt_norm2
        
        if flag_txt_mlp:
            txt_mlp_l1, gelu, txt_mlp_l2 = block.txt_mlp

            txt_mlp_W1 = txt_mlp_l1.weight.data
            txt_mlp_b1 = txt_mlp_l1.bias.data if txt_mlp_l1.bias is not None else None
            txt_mlp_W2 = txt_mlp_l2.weight.data
            txt_mlp_b2 = txt_mlp_l2.bias.data if txt_mlp_l2.bias is not None else None

            U1, S1, Vh1 = grasp_decompose_linear_to_svd(txt_mlp_W1.float(), r=rank_txt_mlp_in)
            self.txt_mlp_l1 = SVDLinear(U1, S1, Vh1, txt_mlp_b1)

            U2, S2, Vh2 = grasp_decompose_linear_to_svd(txt_mlp_W2.float(), r=rank_txt_mlp_out)
            self.txt_mlp_l2 = SVDLinear(U2, S2, Vh2, txt_mlp_b2)

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
            # Aufruf der neuen GRASPLayer
            q = self.img_q_proj(img_modulated)
            k = self.img_k_proj(img_modulated)
            v = self.img_v_proj(img_modulated)
            
            img_q = rearrange(q, "B L (H D) -> B H L D", H=self.num_heads)
            img_k = rearrange(k, "B L (H D) -> B H L D", H=self.num_heads)
            img_v = rearrange(v, "B L (H D) -> B H L D", H=self.num_heads)
            # Aufruf der kopierten Norm
            img_q, img_k = self.img_attn_norm(img_q, img_k, img_v)
        else:
            img_qkv = self.img_attn.qkv(img_modulated)
            img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
            img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

        # prepare txt for attention
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        if self.flag_txt_attn:
            # Aufruf der neuen GRASPLayer
            q = self.txt_q_proj(txt_modulated)
            k = self.txt_k_proj(txt_modulated)
            v = self.txt_v_proj(txt_modulated)
            
            txt_q = rearrange(q, "B L (H D) -> B H L D", H=self.num_heads)
            txt_k = rearrange(k, "B L (H D) -> B H L D", H=self.num_heads)
            txt_v = rearrange(v, "B L (H D) -> B H L D", H=self.num_heads)
            # Aufruf der kopierten Norm
            txt_q, txt_k = self.txt_attn_norm(txt_q, txt_k, txt_v)
        else:
            txt_qkv = self.txt_attn.qkv(txt_modulated)
            txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
            txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

        # --- run actual attention (unverändert) ---
        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)

        attn_mask = None
        if txt_attention_mask is not None:
            attn_mask = txt_attention_mask.to(torch.bool) 
            attn_mask = torch.cat(
                (attn_mask, torch.ones(attn_mask.shape[0], img.shape[1], device=attn_mask.device, dtype=torch.bool)), dim=1
            )
            attn_mask = attn_mask[:, None, None, :].expand(-1, q.shape[1], q.shape[2], -1)

        attn = attention(q, k, v, pe=pe, attn_mask=attn_mask)
        txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1] :]

        # --- calculate the img blocks ---
        if self.flag_img_attn:
            img = img + img_mod1.gate * self.img_attn_proj(img_attn) # Aufruf des kopierten Proj
        else:
            img = img + img_mod1.gate * self.img_attn.proj(img_attn)
        
        if self.flag_img_mlp:
            img_in = (1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift
            # Aufruf der neuen GRASPLayer
            img_in = self.img_mlp_l1(img_in)
            img_in = self.img_mlp_act(img_in)
            img_in = self.img_mlp_l2(img_in)
            img = img + img_mod2.gate * img_in
        else:
            img = img + img_mod2.gate * self.img_mlp((1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift)

        # --- calculate the txt blocks ---
        if self.flag_txt_attn:
            txt = txt + txt_mod1.gate * self.txt_attn_proj(txt_attn) # Aufruf des kopierten Proj
        else:
            txt = txt + txt_mod1.gate * self.txt_attn.proj(txt_attn)
            
        if self.flag_txt_mlp:
            txt_in = (1 + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift
            # Aufruf der neuen GRASPLayer
            txt_in = self.txt_mlp_l1(txt_in)
            txt_in = self.txt_mlp_act(txt_in)
            txt_in = self.txt_mlp_l2(txt_in)
            txt = txt + txt_mod2.gate * txt_in
        else:
            txt = txt + txt_mod2.gate * self.txt_mlp((1 + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift)
        return img, txt

    def forward(
        self, img: Tensor, txt: Tensor, vec: Tensor, pe: Tensor, txt_attention_mask: Optional[Tensor] = None
    ) -> tuple[Tensor, Tensor]:
        # --- Checkpointing (unverändert) ---
        if self.training and self.gradient_checkpointing:
            if not self.cpu_offload_checkpointing:
                return checkpoint(self._forward, img, txt, vec, pe, txt_attention_mask, use_reentrant=False)
            
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