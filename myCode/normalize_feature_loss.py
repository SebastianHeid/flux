from typing import List

import torch
import torch as th


def normalize_feature_loss(feature_loss: torch.Tensor):
    sum_feat_loss = feature_loss.sum()
    normalized_feature_loss = feature_loss / sum_feat_loss
    return normalized_feature_loss

def normalization_feature_loss( feat_teacher):
        num_stages = len(feat_teacher)
        stage_norms = []
        for t_feat in feat_teacher:
            stage_norm = th.norm(t_feat, p=2)
            stage_norms.append(stage_norm)
        stage_norms = th.stack(stage_norms)
        norm_sum = stage_norms.sum()
        alphas = norm_sum / ((stage_norms + 10**(-8)) * num_stages)
        return alphas