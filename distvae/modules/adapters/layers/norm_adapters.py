import torch
import torch.nn as nn
from distvae.models.layers.normalization import PatchGroupNorm

class GroupNormAdapter(nn.Module):
    def __init__(self, group_norm: nn.GroupNorm):
        super().__init__()
        self.group_norm = PatchGroupNorm(
            num_groups=group_norm.num_groups, 
            num_channels=group_norm.num_channels, 
            eps=group_norm.eps, 
            affine=group_norm.affine
        )
        if group_norm.affine:
            self.group_norm.weight = group_norm.weight
            self.group_norm.bias = group_norm.bias

    def forward(self, x):
        return self.group_norm(x)