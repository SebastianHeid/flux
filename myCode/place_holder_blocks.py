import torch
import torch.nn as nn


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kargs):
        return x







class IdentityD(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, img, txt, vec, pe, txt_attention_mask= None):
        return img, txt
