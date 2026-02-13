import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from einops import rearrange
from typing import Optional
from typing import Dict, List, Optional, Tuple, Union
from copy import deepcopy
from library.flux_models import (
    DoubleStreamBlock,
    Modulation,
    ModulationOut,
    QKNorm,
    SingleStreamBlock,
    attention,
)

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


# ==========================================
# 1. Helper: Optimierter SVD Layer Wrapper
# ==========================================
class SVDLinear(nn.Module):
    """
    Ersetzt die manuelle Berechnung (x @ A @ B + bias) durch zwei nn.Linear Module.
    Vorteil: Erlaubt PyTorch 2.0+ (torch.compile), Operationen zu 'fusen' 
    und reduziert den Overhead im Vergleich zu rohen Matrix-Multiplikationen.
    """
    def __init__(self, A_weight, B_weight, bias=None):
        super().__init__()
        # Wir nehmen an: A_weight ist [in, r], B_weight ist [r, out] (für x @ A @ B)
        # nn.Linear speichert Gewichte als [out, in]. Daher transponieren wir hier (.T).
        
        in_features, rank = A_weight.shape
        rank_, out_features = B_weight.shape
        assert rank == rank_, f"Rank mismatch: A={A_weight.shape}, B={B_weight.shape}"

        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_A.weight.data = A_weight.T.contiguous()
        
        self.lora_B = nn.Linear(rank, out_features, bias=True if bias is not None else False)
        self.lora_B.weight.data = B_weight.T.contiguous()
        
        if bias is not None:
            self.lora_B.bias.data = bias

    def forward(self, x):
        # x -> [B, L, r] -> [B, L, out]
        return self.lora_B(self.lora_A(x))

# ==========================================
# 2. Optimized Attention (Vertical Fusion)
# ==========================================
class SelfAttention_Optimized(nn.Module):
    def __init__(self, attn_block, rank_attn, rank_proj, flag_proj):
        super().__init__()
        self.num_heads = attn_block.num_heads
        dim = attn_block.dim
        head_dim = dim // attn_block.num_heads
        
        # --- Gewichte extrahieren ---
        qkv_W = attn_block.qkv.weight.data
        qkv_b = attn_block.qkv.bias.data if attn_block.qkv.bias is not None else torch.zeros(3*dim, device=qkv_W.device)
        
        # Split in Q, K, V
        q_W, k_W, v_W = torch.split(qkv_W, [dim, dim, dim], dim=0)
        q_b, k_b, v_b = torch.split(qkv_b, [dim, dim, dim], dim=0)

        # --- SVD Berechnung ---
        # decompose_linear_to_svd muss A[in, r], B[r, out] liefern
        qa, qb = decompose_linear_to_svd(q_W, r=rank_attn)
        ka, kb = decompose_linear_to_svd(k_W, r=rank_attn)
        va, vb = decompose_linear_to_svd(v_W, r=rank_attn)
        
        # --- OPTIMIERUNG: Fused Input Projection ---
        # Statt x dreimal zu laden (x@qA, x@kA, x@vA), kleben wir die A-Matrizen zusammen.
        # Shape: [in, 3*rank]
        # x wird nur 1x geladen und gegen diese Matrix gerechnet.
        self.rank_attn = rank_attn
        self.fused_A = nn.Linear(dim, 3 * rank_attn, bias=False)
        fused_W = torch.cat([qa, ka, va], dim=1) # [in, 3r]
        self.fused_A.weight.data = fused_W.T.contiguous() # nn.Linear braucht [3r, in]
        
        # --- Output Projection (B-Matrizen) ---
        # Diese müssen getrennt bleiben, da die Outputs (q, k, v) getrennt weiterverarbeitet werden.
        self.q_B = nn.Linear(rank_attn, dim, bias=True)
        self.q_B.weight.data = qb.T.contiguous()
        self.q_B.bias.data = q_b
        
        self.k_B = nn.Linear(rank_attn, dim, bias=True)
        self.k_B.weight.data = kb.T.contiguous()
        self.k_B.bias.data = k_b
        
        self.v_B = nn.Linear(rank_attn, dim, bias=True)
        self.v_B.weight.data = vb.T.contiguous()
        self.v_B.bias.data = v_b
        
        self.norm = attn_block.norm # Übernehme QKNorm
        
        # --- Output Projection des Blocks ---
        if flag_proj:
            proj_W = attn_block.proj.weight.data
            proj_b = attn_block.proj.bias.data if attn_block.proj.bias is not None else None
            pa, pb = decompose_linear_to_svd(proj_W, r=rank_proj)
            self.proj = SVDLinear(pa, pb, proj_b)
        else:
            self.proj = attn_block.proj

    def forward(self, x: torch.Tensor, pe: torch.Tensor) -> torch.Tensor:
        # 1. Fused Projection: Reduziert Memory Reads drastisch
        # Output Shape: [B, L, 3*rank]
        shared_compressed = self.fused_A(x)
        
        # 2. Split (Memory-View Operation, sehr billig)
        q_in, k_in, v_in = shared_compressed.split(self.rank_attn, dim=-1)
        
        # 3. Individuelle Projektionen auf Full Rank
        q = self.q_B(q_in)
        k = self.k_B(k_in)
        v = self.v_B(v_in)
        
        # 4. Standard Attention Logik
        q = rearrange(q, "B L (H D) -> B H L D", H=self.num_heads)
        k = rearrange(k, "B L (H D) -> B H L D", H=self.num_heads)
        v = rearrange(v, "B L (H D) -> B H L D", H=self.num_heads)
        
        q, k = self.norm(q, k, v)
        
        # Annahme: 'attention' Funktion ist global verfügbar
        x = attention(q, k, v, pe=pe)
        x = self.proj(x)
        return x

# ==========================================
# 3. Optimized Modulation
# ==========================================
class ModulationSVD_Optimized(nn.Module):
    def __init__(self, block_modulation, rank_mod):
        super().__init__()
        self.is_double = block_modulation.is_double
        self.multiplier = 6 if self.is_double else 3

        lin_W = block_modulation.lin.weight.data 
        lin_b = block_modulation.lin.bias.data if block_modulation.lin.bias is not None else None
        
        # SVD
        la, lb = decompose_linear_to_svd(lin_W, r=rank_mod)
        self.svd_layer = SVDLinear(la, lb, lin_b)

    def forward(self, vec: torch.Tensor):
        x = F.silu(vec)
        out = self.svd_layer(x) # Nutzt SVDLinear
        out = out[:, None, :].chunk(self.multiplier, dim=-1)
        return (
            ModulationOut(*out[:3]),
            ModulationOut(*out[3:]) if self.is_double else None,
        )

# ==========================================
# 4. Main Block: DoubleStreamBlockPruned
# ==========================================
class DoubleStreamBlockPruned(nn.Module):
    def __init__(self,
                 block,
                 rank_attn_img=512, rank_attn_txt=512,
                 rank_img_mlp_in=512, rank_img_mlp_out=512,
                 rank_txt_mlp_in=512, rank_txt_mlp_out=512,
                 rank_img_mod=512, rank_txt_mod=512,
                 rank_img_proj=512, rank_txt_proj=512,
                 flag_img_attn=False, flag_txt_attn=False,
                 flag_img_mlp=False, flag_txt_mlp=False,
                 flag_img_mod=False, flag_txt_mod=False,
                 flag_img_proj=False, flag_txt_proj=False):
        super().__init__()
        
        # --- Config Copy ---
        self.hidden_size = block.hidden_size
        self.num_heads = block.num_heads
        
        # Flags speichern
        self.flag_img_attn = flag_img_attn
        self.flag_txt_attn = flag_txt_attn
        self.flag_img_mlp = flag_img_mlp
        self.flag_txt_mlp = flag_txt_mlp
        self.flag_img_mod = flag_img_mod
        self.flag_txt_mod = flag_txt_mod
        self.flag_img_proj = flag_img_proj
        self.flag_txt_proj = flag_txt_proj

        # --- Modulation ---
        if flag_img_mod:
            self.img_mod = ModulationSVD_Optimized(block.img_mod, rank_img_mod)
        else: 
            self.img_mod = block.img_mod
            
        if flag_txt_mod:
            self.txt_mod = ModulationSVD_Optimized(block.txt_mod, rank_txt_mod)
        else:
            self.txt_mod = block.txt_mod

        # --- Norms ---
        self.img_norm1 = block.img_norm1
        self.img_norm2 = block.img_norm2
        self.txt_norm1 = block.txt_norm1
        self.txt_norm2 = block.txt_norm2

        # --- Attention ---
        if flag_img_attn:
            self.img_attn = SelfAttention_Optimized(block.img_attn, rank_attn_img, rank_img_proj, flag_img_proj)
        else: 
            self.img_attn = block.img_attn
            
        if flag_txt_attn:
            self.txt_attn = SelfAttention_Optimized(block.txt_attn, rank_attn_txt, rank_txt_proj, flag_txt_proj)
        else:
            self.txt_attn = block.txt_attn

        # --- MLP Image ---
        if flag_img_mlp:
            l1, gelu, l2 = block.img_mlp
            # Layer 1
            a1, b1 = decompose_linear_to_svd(l1.weight.data, r=rank_img_mlp_in)
            self.img_mlp_1 = SVDLinear(a1, b1, l1.bias.data if l1.bias is not None else None)
            
            # Activation
            self.img_mlp_act = deepcopy(gelu)
            
            # Layer 2
            a2, b2 = decompose_linear_to_svd(l2.weight.data, r=rank_img_mlp_out)
            self.img_mlp_2 = SVDLinear(a2, b2, l2.bias.data if l2.bias is not None else None)
        else:
            self.img_mlp = block.img_mlp

        # --- MLP Text ---
        if flag_txt_mlp:
            l1, gelu, l2 = block.txt_mlp
            # Layer 1
            a1, b1 = decompose_linear_to_svd(l1.weight.data, r=rank_txt_mlp_in)
            self.txt_mlp_1 = SVDLinear(a1, b1, l1.bias.data if l1.bias is not None else None)
            
            # Activation
            self.txt_mlp_act = deepcopy(gelu)
            
            # Layer 2
            a2, b2 = decompose_linear_to_svd(l2.weight.data, r=rank_txt_mlp_out)
            self.txt_mlp_2 = SVDLinear(a2, b2, l2.bias.data if l2.bias is not None else None)
        else:
            self.txt_mlp = block.txt_mlp

        self.gradient_checkpointing = False

    def _forward(self, img, txt, vec, pe, txt_attention_mask=None):
        img_mod1, img_mod2 = self.img_mod(vec)
        txt_mod1, txt_mod2 = self.txt_mod(vec)

        # === 1. Image Attention ===
        img_modulated = self.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        
   
        
        # A) Prepare Image QKV
        if self.flag_img_attn: # SVD Optimized
            # Wir rufen NICHT forward auf, sondern holen uns QKV manuell
            shared = self.img_attn.fused_A(img_modulated)
            qi, ki, vi = shared.split(self.img_attn.rank_attn, dim=-1)
            img_q = self.img_attn.q_B(qi)
            img_k = self.img_attn.k_B(ki)
            img_v = self.img_attn.v_B(vi)
            # Rearrange & Norm
            img_q = rearrange(img_q, "B L (H D) -> B H L D", H=self.num_heads)
            img_k = rearrange(img_k, "B L (H D) -> B H L D", H=self.num_heads)
            img_v = rearrange(img_v, "B L (H D) -> B H L D", H=self.num_heads)
            img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)
        else:
            img_qkv = self.img_attn.qkv(img_modulated)
            img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
            img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

        # B) Prepare Text QKV
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        
        if self.flag_txt_attn: # SVD Optimized
            shared = self.txt_attn.fused_A(txt_modulated)
            qt, kt, vt = shared.split(self.txt_attn.rank_attn, dim=-1)
            txt_q = self.txt_attn.q_B(qt)
            txt_k = self.txt_attn.k_B(kt)
            txt_v = self.txt_attn.v_B(vt)
            
            txt_q = rearrange(txt_q, "B L (H D) -> B H L D", H=self.num_heads)
            txt_k = rearrange(txt_k, "B L (H D) -> B H L D", H=self.num_heads)
            txt_v = rearrange(txt_v, "B L (H D) -> B H L D", H=self.num_heads)
            txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)
        else:
            txt_qkv = self.txt_attn.qkv(txt_modulated)
            txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
            txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

        # C) Joint Attention Execution
        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)

        # Mask Logic (unchanged)
        attn_mask = None
        if txt_attention_mask is not None:
            attn_mask = txt_attention_mask.to(torch.bool)
            attn_mask = torch.cat(
                (attn_mask, torch.ones(attn_mask.shape[0], img.shape[1], device=attn_mask.device, dtype=torch.bool)), dim=1
            )
            attn_mask = attn_mask[:, None, None, :].expand(-1, q.shape[1], q.shape[2], -1)

        attn = attention(q, k, v, pe=pe, attn_mask=attn_mask)
        txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1] :]

        # === 2. Post-Attention Projections & Residuals ===
        
        # --- Image ---
        if self.flag_img_attn: # SVD Proj
            img = img + img_mod1.gate * self.img_attn.proj(img_attn) # proj ist SVDLinear
        else:
            img = img + img_mod1.gate * self.img_attn.proj(img_attn)

        # --- Text ---
        if self.flag_txt_attn: # SVD Proj
            txt = txt + txt_mod1.gate * self.txt_attn.proj(txt_attn)
        else:
            txt = txt + txt_mod1.gate * self.txt_attn.proj(txt_attn)

        # === 3. MLP Blocks ===

        # --- Image MLP ---
        if self.flag_img_mlp:
            img_in = (1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift
            # SVDLinear Pipeline
            img_in = self.img_mlp_1(img_in)
            img_in = self.img_mlp_act(img_in)
            img_in = self.img_mlp_2(img_in)
            
            img = img + img_mod2.gate * img_in
        else:
            img = img + img_mod2.gate * self.img_mlp((1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift)

        # --- Text MLP ---
        if self.flag_txt_mlp:
            txt_in = (1 + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift
            # SVDLinear Pipeline
            txt_in = self.txt_mlp_1(txt_in)
            txt_in = self.txt_mlp_act(txt_in)
            txt_in = self.txt_mlp_2(txt_in)
            
            txt = txt + txt_mod2.gate * txt_in
        else:
            txt = txt + txt_mod2.gate * self.txt_mlp((1 + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift)

        return img, txt

    def forward(self, img, txt, vec, pe, txt_attention_mask=None):
        # Hier optional checkpointing Logic einfügen
        return self._forward(img, txt, vec, pe, txt_attention_mask)
    
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from einops import rearrange
from typing import Optional



class SingleStreamBlockPruned(nn.Module):
    """
    Optimized DiT SingleStreamBlock with SVD pruning capabilities.
    Features: Vertical Fusion for QKV A-matrices and clean modularization.
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
        
        # --- Config Copy ---
        self.hidden_dim = block.hidden_size
        self.hidden_size = block.hidden_size
        self.num_heads = block.num_heads
        self.mlp_hidden_dim = block.mlp_hidden_dim
        self.scale = block.scale
        
        self.flag_attn = flag_attn
        self.flag_mlp = flag_mlp
        self.flag_mlp2 = flag_mlp2
        self.flag_mod = flag_mod
        
        # --- Components ---
        self.norm = block.norm
        self.pre_norm = block.pre_norm
        self.mlp_act = block.mlp_act # meist GELU
        
        # 1. Modulation
        if flag_mod:
            self.modulation = ModulationSVD_Optimized(block.modulation, rank_mod)
        else: 
            self.modulation = block.modulation

        # 2. Linear 1 (QKV + MLP In)
        # Original shape: [3*hidden + mlp_hidden, hidden]
        full_W = block.linear1.weight.data
        full_b = block.linear1.bias.data if block.linear1.bias is not None else None
        
        # Split in QKV part and MLP part
        # dim=0 is Output dimension in Linear Layer weights
        qkv_W, mlp_W = torch.split(full_W, [3 * self.hidden_size, self.mlp_hidden_dim], dim=0)
        
        if full_b is not None:
            qkv_b, mlp_b = torch.split(full_b, [3 * self.hidden_size, self.mlp_hidden_dim], dim=0)
        else:
            qkv_b = mlp_b = None

        # --- Handling QKV (Attention) ---
        if flag_attn:
            # Split Q, K, V
            q_W, k_W, v_W = torch.split(qkv_W, [self.hidden_size]*3, dim=0)
            if qkv_b is not None:
                q_b, k_b, v_b = torch.split(qkv_b, [self.hidden_size]*3, dim=0)
            else:
                q_b = k_b = v_b = None # Sollte 0 sein, aber sicherheitshalber
                
            # SVD Decomp
            qa, qb_ = decompose_linear_to_svd(q_W, r=rank_attn)
            ka, kb_ = decompose_linear_to_svd(k_W, r=rank_attn)
            va, vb_ = decompose_linear_to_svd(v_W, r=rank_attn)
            
            # --- OPTIMIZATION: Fused QKV Input (A-Matrix) ---
            # Wir bauen EINE Matrix für x @ A_fused.
            # Shape A_fused: [hidden, 3*rank]
            # Transpose beachten für nn.Linear weight: [3*rank, hidden]
            self.rank_attn = rank_attn
            self.qkv_A_fused = nn.Linear(self.hidden_size, 3 * rank_attn, bias=False)
            fused_W = torch.cat([qa, ka, va], dim=1) 
            self.qkv_A_fused.weight.data = fused_W.T.contiguous()
            
            # B-Matrices (Output Projection) bleiben separat
            self.q_B = nn.Linear(rank_attn, self.hidden_size, bias=True)
            self.q_B.weight.data = qb_.T.contiguous()
            if q_b is not None: self.q_B.bias.data = q_b

            self.k_B = nn.Linear(rank_attn, self.hidden_size, bias=True)
            self.k_B.weight.data = kb_.T.contiguous()
            if k_b is not None: self.k_B.bias.data = k_b
            
            self.v_B = nn.Linear(rank_attn, self.hidden_size, bias=True)
            self.v_B.weight.data = vb_.T.contiguous()
            if v_b is not None: self.v_B.bias.data = v_b
            
        else:
            # Fallback: Dense QKV
            self.qkv_dense = nn.Linear(self.hidden_size, 3 * self.hidden_size)
            self.qkv_dense.weight.data = qkv_W
            if qkv_b is not None: self.qkv_dense.bias.data = qkv_b

        # --- Handling MLP (Input Part) ---
        if flag_mlp:
            ma, mb = decompose_linear_to_svd(mlp_W, r=rank_mlp)
            self.mlp_in_svd = SVDLinear(ma, mb, mlp_b)
        else:
            self.mlp_in_dense = nn.Linear(self.hidden_size, self.mlp_hidden_dim)
            self.mlp_in_dense.weight.data = mlp_W
            if mlp_b is not None: self.mlp_in_dense.bias.data = mlp_b

        # 3. Linear 2 (Output Part)
        # Input dim: hidden + mlp_hidden (wegen concat)
        if flag_mlp2:
            W2 = block.linear2.weight.data
            b2 = block.linear2.bias.data if block.linear2.bias is not None else None
            l2a, l2b = decompose_linear_to_svd(W2, r=rank_mlp2)
            self.linear2 = SVDLinear(l2a, l2b, b2)
        else:
            self.linear2 = block.linear2

        self.gradient_checkpointing = False
        self.cpu_offload_checkpointing = False

    def _forward(self, x: torch.Tensor, vec: torch.Tensor, pe: torch.Tensor, txt_attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        mod, _ = self.modulation(vec)
        x_mod = (1 + mod.scale) * self.pre_norm(x) + mod.shift
        
        # --- 1. Compute Q, K, V ---
        if self.flag_attn:
            # Optimized Fused Path
            # 1x Input Load für alle 3
            shared = self.qkv_A_fused(x_mod) # [B, L, 3*rank]
            qi, ki, vi = shared.split(self.rank_attn, dim=-1)
            
            q = self.q_B(qi)
            k = self.k_B(ki)
            v = self.v_B(vi)
            
            # Rearrange standard DiT
            q = rearrange(q, "B L (H D) -> B H L D", H=self.num_heads)
            k = rearrange(k, "B L (H D) -> B H L D", H=self.num_heads)
            v = rearrange(v, "B L (H D) -> B H L D", H=self.num_heads)
        else:
            # Dense Path
            qkv = self.qkv_dense(x_mod)
            q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
            
        q, k = self.norm(q, k, v)

        # --- 2. Compute MLP Part ---
        if self.flag_mlp:
            mlp = self.mlp_in_svd(x_mod)
        else:
            mlp = self.mlp_in_dense(x_mod)

        # --- 3. Attention ---
        # Mask creation (Standard Logic)
        attn_mask = None
        if txt_attention_mask is not None:
            attn_mask = txt_attention_mask.to(torch.bool)
            attn_mask = torch.cat(
                (attn_mask, torch.ones(attn_mask.shape[0], x.shape[1] - txt_attention_mask.shape[1], device=attn_mask.device, dtype=torch.bool)), 
                dim=1
            )
            attn_mask = attn_mask[:, None, None, :].expand(-1, q.shape[1], q.shape[2], -1)

        attn = attention(q, k, v, pe=pe, attn_mask=attn_mask)

        # --- 4. Output Projection ---
        # Concat Attention Output + MLP Activation
        x_concat = torch.cat((attn, self.mlp_act(mlp)), dim=2)
        
        # Linear2 (handled by SVDLinear or Dense internally)
        output = self.linear2(x_concat)
        
        return x + mod.gate * output

    def forward(self, x, vec, pe, txt_attention_mask=None):
        if self.training and self.gradient_checkpointing:
             # (Checkpointing Logik hier einkürzen für Übersicht, bleibt identisch)
             return self._forward(x, vec, pe, txt_attention_mask)
        else:
            return self._forward(x, vec, pe, txt_attention_mask)