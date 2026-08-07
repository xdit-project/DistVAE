import torch
import torch.nn as nn
from typing import Optional

from distvae.models.layers.normalization import PatchGroupNorm


class GroupNormAdapter(nn.Module):
    """A GroupNorm that sums its statistics across the ranks the feature map is split over

    Takes the axis of that split rather than reading it from the distributed environment. The
    environment holds one axis for the whole process, written by whichever adapter was built
    last, so an encoder split on height and a decoder split on width in the same process would
    leave one of them reducing along an axis it does not own - and a group norm given the wrong
    axis does not fail, it returns numbers that are wrong by the ratio of the two. Left unsaid,
    the axis still falls back to the environment, which is what the layers built outside an
    adapter rely on.
    """

    def __init__(self, group_norm: nn.GroupNorm, patch_dim: Optional[int] = None):
        super().__init__()
        self.group_norm = PatchGroupNorm(
            num_groups=group_norm.num_groups,
            num_channels=group_norm.num_channels,
            eps=group_norm.eps,
            affine=group_norm.affine,
            patch_dim=patch_dim,
        )
        if group_norm.affine:
            self.group_norm.weight = group_norm.weight
            self.group_norm.bias = group_norm.bias

    def forward(self, x):
        return self.group_norm(x)
