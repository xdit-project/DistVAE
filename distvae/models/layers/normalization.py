import math
import numbers
from typing import Optional

import torch
import torch.nn as nn
import torch.distributed as dist
from torch import Tensor

from diffusers.models.activations import get_activation
from distvae.utils import DistributedEnv, ParallelContext, normalize_patch_dim


class PatchGroupNorm(nn.GroupNorm):
    r"""Applies Group Normalization over a mini-batch of inputs.

    This layer implements the operation as described in
    the paper `Group Normalization <https://arxiv.org/abs/1803.08494>`__

    .. math::
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The input channels are separated into :attr:`num_groups` groups, each containing
    ``num_channels / num_groups`` channels. :attr:`num_channels` must be divisible by
    :attr:`num_groups`. The mean and standard-deviation are calculated
    separately over the each group. :math:`\gamma` and :math:`\beta` are learnable
    per-channel affine transform parameter vectors of size :attr:`num_channels` if
    :attr:`affine` is ``True``.
    The standard-deviation is calculated via the biased estimator, equivalent to
    `torch.var(input, unbiased=False)`.

    This layer uses statistics computed from input data in both training and
    evaluation modes.

    Args:
        num_groups (int): number of groups to separate the channels into
        num_channels (int): number of channels expected in input
        eps: a value added to the denominator for numerical stability. Default: 1e-5
        affine: a boolean value that when set to ``True``, this module
            has learnable per-channel affine parameters initialized to ones (for weights)
            and zeros (for biases). Default: ``True``.

    Shape:
        - Input: :math:`(N, C, *)` where :math:`C=\text{num\_channels}`
        - Output: :math:`(N, C, *)` (same shape as input)

    Examples::

        >>> input = torch.randn(20, 6, 10, 10)
        >>> # Separate 6 channels into 3 groups
        >>> m = nn.GroupNorm(3, 6)
        >>> # Separate 6 channels into 6 groups (equivalent with InstanceNorm)
        >>> m = nn.GroupNorm(6, 6)
        >>> # Put all 6 channels into a single group (equivalent with LayerNorm)
        >>> m = nn.GroupNorm(1, 6)
        >>> # Activating the module
        >>> output = m(input)
    """

    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: float = 1e-5,
        affine: bool = True,
        device=None,
        dtype=None,
        patch_dim: Optional[int] = None,
        parallel_context: Optional[ParallelContext] = None,
    ) -> None:
        self.parallel_context = parallel_context
        self.patch_dim = parallel_context.patch_dim if parallel_context is not None else patch_dim
        super().__init__(
            num_groups=num_groups,
            num_channels=num_channels,
            eps=eps,
            affine=affine,
            device=device,
            dtype=dtype
        )
    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        ndim = x.ndim
        shape = x.shape
        axis = DistributedEnv.get_patch_dim() if self.patch_dim is None else self.patch_dim
        patch_dim = ndim + normalize_patch_dim(axis, ndim, spatial_only=True)

        vae_group = (
            self.parallel_context.group
            if self.parallel_context is not None
            else DistributedEnv.get_vae_group()
        )
        group_world_size = (
            self.parallel_context.world_size
            if self.parallel_context is not None
            else DistributedEnv.get_group_world_size()
        )
        x = x.detach()
        channels_per_group = shape[1] // self.num_groups

        x = x.view(shape[0], self.num_groups, -1, *shape[2: ])
        reduced = tuple(range(2, x.ndim))
        # [bs, num_groups, 1, 1, 1] for 4D input, one more 1 for 5D.
        per_group = (shape[0], self.num_groups, *([1] * (x.ndim - 2)))

        # This rank's row count travels with its group sums. Both are sums over the same group of
        # ranks, so combining them changes no arithmetic, and sent alone the row count costs a
        # whole round trip to move one number. Float32 holds a row count exactly either way.
        # Support 4D (N,C,H,W) and 5D (N,C,F,H,W); patch dim is first spatial (index 2).
        totals = torch.empty(
            1 + shape[0] * self.num_groups, dtype=torch.float32, device=x.device
        )
        totals[0] = shape[patch_dim]
        totals[1:] = x.sum(dim=reduced, dtype=torch.float32).flatten()
        # Summing one rank's numbers across one rank returns them unchanged, so on a single-rank
        # group both reductions here are the identity. They are still real collectives, though:
        # a decode of a VAE with twenty-five group norms issued fifty of them to talk to nobody.
        if group_world_size > 1:
            dist.all_reduce(totals, group=vae_group)

        patch_size = totals[0]
        nelements = (
            channels_per_group *
            math.prod(shape[2: patch_dim]) *
            patch_size *
            math.prod(shape[patch_dim + 1: ])
        )
        group_sum = totals[1:].view(shape[0], self.num_groups)
        E = (group_sum / nelements).view(per_group).to(x.dtype)

        # Squared about the mean of the whole group rather than this rank's share of it. A rank
        # holding a brighter patch has a mean of its own, and deviations measured from that one
        # leave out how far the patch itself sits from the middle, so the summed variance comes
        # out short of the variance the unsharded norm computes.
        group_square_sum = ((x - E) ** 2).sum(dim=reduced, dtype=torch.float32)
        if group_world_size > 1:
            dist.all_reduce(group_square_sum, group=vae_group)
        # Divided by the count, not one less than it, which is the estimator nn.GroupNorm uses.
        var = (group_square_sum / nelements).view(per_group).to(x.dtype)

        x = (x - E) / torch.sqrt(var + self.eps)
        x = x.view(shape[0], -1, *shape[2: ])
        if self.weight is not None and self.bias is not None:
            weight = self.weight.view(1, -1, *([1] * (ndim - 2)))
            bias = self.bias.view(1, -1, *([1] * (ndim - 2)))
            x = x * weight + bias

        return x


class RMSNorm(nn.Module):
    def __init__(self, dim, eps: float, elementwise_affine: bool = True):
        super().__init__()

        self.eps = eps

        if isinstance(dim, numbers.Integral):
            dim = (dim,)

        self.dim = torch.Size(dim)

        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.weight = None

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)

        if self.weight is not None:
            # convert into half-precision if necessary
            if self.weight.dtype in [torch.float16, torch.bfloat16]:
                hidden_states = hidden_states.to(self.weight.dtype)
            hidden_states = hidden_states * self.weight
        else:
            hidden_states = hidden_states.to(input_dtype)

        return hidden_states