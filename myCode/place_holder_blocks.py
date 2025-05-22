import torch
import torch.nn as nn
from library.flux_models import DoubleStreamBlock, SingleStreamBlock

# class Identity(SingleStreamBlock):
#     def __init__(self, hidden_size, num_heads):
#         super().__init__(hidden_size=hidden_size, num_heads=num_heads)

#     def forward(self, x, *args, **kargs):
#         return x

# class IdentityD(DoubleStreamBlock):
#     def __init__(self, hidden_size, mlp_ratio, num_heads):
#         super().__init__(hidden_size=hidden_size,mlp_ratio= mlp_ratio, num_heads=num_heads)

#     def forward(self, img, txt, vec, pe, txt_attention_mask= None):
#         return img, txt


class Identity(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()

    def forward(self, x, *args, **kargs):
        return x
    
class IdentityD(nn.Module):
    def __init__(self, hidden_size, mlp_ratio, num_heads):
        super().__init__()

    def forward(self, img, txt, vec, pe, txt_attention_mask= None):
        return img, txt
