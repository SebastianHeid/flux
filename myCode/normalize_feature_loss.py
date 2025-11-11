from typing import List

import torch
import torch as th
import torch.nn.functional as F


def normalize_feature_loss(feature_loss: torch.Tensor):
    sum_feat_loss = feature_loss.sum()
    normalized_feature_loss = feature_loss / sum_feat_loss
    return normalized_feature_loss

# def normalization_feature_loss( feat_teacher):
#         num_stages = len(feat_teacher)
#         stage_norms = []
#         for t_feat in feat_teacher:
#             stage_norm = th.norm(t_feat, p=2)
#             stage_norms.append(stage_norm)
#         stage_norms = th.stack(stage_norms)
#         norm_sum = stage_norms.sum()
#         alphas = norm_sum / ((stage_norms + 10**(-8)) * num_stages)
#         return alphas



import torch as th

# def normalization_feature_loss(feat_losses: th.Tensor, eps: float = 1e-8):
#     """
#     Normalize feature losses across layers to balance their contribution.

#     Args:
#         feat_losses: Tensor of shape [num_layers, batch_size]
#                      or [num_layers] if already averaged across batch.
#         eps: Small constant for numerical stability.

#     Returns:
#         Weighted per-batch feature loss tensor (same shape as input).
#     """
#     # Compute L2 norm per layer (averaged across batch if 2D)
#     if feat_losses.dim() == 2:
#         stage_norms = feat_losses.norm(p=2, dim=1)  # shape [num_layers]
#     else:
#         stage_norms = feat_losses.abs() + eps        # shape [num_layers]
    
#     # Normalize inversely proportional to magnitude (large norms → smaller weight)
#     inv_norms = 1.0 / (stage_norms + eps)
#     inv_norms = inv_norms.clamp(max=10.0)
#     alphas = inv_norms / inv_norms.sum() * len(inv_norms)  # normalized weights
    
#     # Apply weights to each layer loss
#     if feat_losses.dim() == 2:
#         weighted = alphas[:, None] * feat_losses
#     else:
#         weighted = alphas * feat_losses

#     return weighted


# def normalization_feature_loss(feat_losses: th.Tensor, eps: float = 1e-8, max_scale: float = 10.0):
#     """
#     Normalize feature losses across layers to balance their contribution.
#     """
#     if feat_losses.dim() == 2:
#         # Compute mean L2 norm per layer
#         stage_norms = feat_losses.norm(p=2, dim=1)
#     else:
#         stage_norms = feat_losses.abs() + eps

#     # Inverse normalization (large loss → smaller weight)
#     inv_norms = 1.0 / (stage_norms + eps)

#     # Clamp extreme values to avoid exploding weights
#     inv_norms = inv_norms.clamp(max=max_scale)

#     # Normalize to keep mean weight ~1
#     alphas = inv_norms / (inv_norms.sum() + eps) * len(inv_norms)
#     alphas = alphas.detach()  # avoid gradient feedback through weighting

#     if feat_losses.dim() == 2:
#         weighted = alphas[:, None] * feat_losses
#     else:
#         weighted = alphas * feat_losses

#     return weighted


# def normalization_feature_loss(feat_losses: th.Tensor, eps: float = 1e-8, max_scale: float = 10.0):
#     """
#     Normalize feature losses across layers to balance their contribution
#     and stabilize overall loss magnitude.
#     """
#     if feat_losses.dim() == 2:
#         stage_norms = feat_losses.norm(p=2, dim=1)
#     else:
#         stage_norms = feat_losses.abs() + eps

#     inv_norms = 1.0 / (stage_norms + eps)
#     inv_norms = inv_norms.clamp(max=max_scale)

#     # Layer balancing
#     alphas = inv_norms / (inv_norms.sum() + eps) * len(inv_norms)
#     alphas = alphas.detach()

#     if feat_losses.dim() == 2:
#         weighted = alphas[:, None] * feat_losses
#     else:
#         weighted = alphas * feat_losses

#     # 🔹 Global normalization (keep total loss magnitude stable)
#     global_norm = weighted.norm(p=2)
#     weighted = weighted / (global_norm + eps)

#     return weighted


# import torch as th

# def normalization_feature_loss(feat_losses: th.Tensor, eps: float = 1e-8):
#     """
#     Normalize feature losses across layers so that their total sum = 1.
#     This prevents exploding losses and ensures balanced layer contribution.

#     Args:
#         feat_losses: Tensor of shape [num_layers, batch_size] or [num_layers].
#         eps: Small constant for numerical stability.

#     Returns:
#         Normalized feature losses where sum(feat_losses_norm) = 1.
#     """
#     # Ensure positive values
#     feat_losses = feat_losses.abs()

#     # Sum across layers (and batch if present)
#     total = feat_losses.sum()

#     # Avoid division by zero
#     if total < eps:
#         return feat_losses * 0.0  # all zeros if loss is degenerate

#     # Normalize so that the sum = 1
#     normalized = feat_losses / (total + eps)

#     return normalized



def normalization_feature_loss(feat_losses: th.Tensor, eps: float = 1e-8):
    """
    Normalize feature losses across layers to balance contributions,
    but preserve global loss magnitude so it can decrease during training.
    """
    # feat_losses shape: [num_layers, batch_size] or [num_layers]
    feat_losses = feat_losses.abs()

    # Compute relative layer weights (sum to 1)
    layer_weights = feat_losses / (feat_losses.sum(dim=0, keepdim=True) + eps)

    # Scale back by the *mean* magnitude so global scale is preserved
    global_scale = feat_losses.mean()
    weighted = layer_weights * global_scale

    return weighted


import torch as th


def normalized_feature_distillation_loss(
    teacher_feats: list[th.Tensor],
    student_feats: list[th.Tensor],
    base_distill_loss: th.Tensor | None = None,
    eps: float = 1e-8,
):
    """
    Implements Eq. (8)–(9) feature normalization for distillation.

    Args:
        teacher_feats: list of teacher feature tensors [L] (each [B, C, H, W])
        student_feats: list of student feature tensors [L] (same shapes)
        base_distill_loss: optional scalar tensor (|L_distill|), 
                           e.g., global L2 or noise-pred loss from student-teacher difference
        eps: numerical stability constant

    Returns:
        total_feature_loss: scalar tensor
        per_layer_losses: list of per-layer scalar losses
    """
    L = len(teacher_feats)
    device = teacher_feats[0].device

    # Compute per-layer feature losses
    per_layer_losses = []
    teacher_norms = []

    for t_feat, s_feat in zip(teacher_feats, student_feats):
        diff = t_feat - s_feat
        loss_l = (diff.pow(2).flatten(start_dim=1).mean(dim=1)).mean()  # L2^2 loss
        norm_t =(t_feat.pow(2).flatten(start_dim=1).mean(dim=1)).sqrt().mean()  # ||f_tea^l||_2
        per_layer_losses.append(loss_l)
        teacher_norms.append(norm_t)

    feat_losses = th.stack(per_layer_losses)       # [L]
    teacher_norms = th.stack(teacher_norms)       # [L]

    # Global teacher normalization factor
    teacher_norm_mean = teacher_norms.mean()

    # Base distillation loss magnitude
    with th.no_grad():
        base_mag = (feat_losses.mean() if base_distill_loss is None else base_distill_loss)
        base_mag = base_mag.detach().abs()

        # base_mag weights w_l according to Eq. (9)
        w = (base_mag / (feat_losses + eps)) * \
            (teacher_norms.sum() / (L * teacher_norms + eps))

    # Weighted sum
    total_feature_loss = (w * feat_losses).sum()
    
    
    return total_feature_loss





def compute_color_consistency_loss(student, teacher,
                                   w_meanstd=1.0,
                                   w_cov=0.1,
                                   w_range=0.05,
                                   eps=1e-6):
    """
    Match color statistics between student and teacher.
    Works on tensors shaped [B, C, H, W].
    """
    # Safety & dtype
    assert student.ndim == 4 and teacher.ndim == 4
    B, C, H, W = student.shape
    teacher = teacher.detach()                    # no grad through teacher
    s = student.float()
    t = teacher.float()

    loss = s.new_tensor(0.0)

    # -------- 1) Per-image, per-channel mean/std (instance stats) --------
    # shape: [B, C]
    s_mean = s.mean(dim=(2,3))
    t_mean = t.mean(dim=(2,3))

    # std with eps & unbiased=False
    s_var = s.var(dim=(2,3), unbiased=False)
    t_var = t.var(dim=(2,3), unbiased=False)
    s_std = torch.sqrt(s_var + eps)
    t_std = torch.sqrt(t_var + eps)

    loss_mean = F.mse_loss(s_mean, t_mean)
    loss_std  = F.mse_loss(s_std,  t_std)
    loss += w_meanstd * (loss_mean + loss_std)

    # -------- 2) Centered covariance / Gram matching (per-image) --------
    # flatten spatial: [B, C, HW]
    s_flat = s.reshape(B, C, -1)
    t_flat = t.reshape(B, C, -1)

    # center per image & channel
    s_center = s_flat - s_flat.mean(dim=2, keepdim=True)
    t_center = t_flat - t_flat.mean(dim=2, keepdim=True)

    # covariance (or centered Gram). Scale by HW-1 to be invariant to size
    denom = max(H*W - 1, 1)
    s_cov = torch.bmm(s_center, s_center.transpose(1, 2)) / denom   # [B, C, C]
    t_cov = torch.bmm(t_center, t_center.transpose(1, 2)) / denom

    loss_cov = F.mse_loss(s_cov, t_cov)
    loss += w_cov * loss_cov

    # -------- 3) Range control (robust alternative to hard min/max) --------
    # Option A (robust): penalize if student exceeds teacher's per-image channel range
    # teacher ranges per image/channel
    t_min = t.amin(dim=(2,3))
    t_max = t.amax(dim=(2,3))

    s_min = s.amin(dim=(2,3))
    s_max = s.amax(dim=(2,3))

    # hinge: only penalize when outside teacher's range
    under = F.relu(t_min - s_min)      # student too low
    over  = F.relu(s_max - t_max)      # student too high
    loss_range = (under.mean() + over.mean())

    loss += w_range * loss_range

    # (If you really want exact min/max matching, replace range block with:
    # loss += w_range * 0.5 * (F.mse_loss(s_min, t_min) + F.mse_loss(s_max, t_max))
    # but gradients will be sparse.)

    return loss.unsqueeze(0) 