import math
import numbers
from typing import Optional

import torch
import torch.nn as nn
import torch.distributed as dist
from torch import Tensor

from diffusers.models.activations import get_activation
from distvae.utils import DistributedEnv


class PatchAdaGroupNorm(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        out_dim: int,
        num_groups: int,
        act_fn: Optional[str] = None,
        eps: float = 1e-5,
        patch_dim: int = -2,
    ):
        super().__init__()
        self.patch_dim = patch_dim
        self.num_groups = num_groups
        self.eps = eps

        if act_fn is None:
            self.act = None
        else:
            self.act = get_activation(act_fn)

        self.linear = nn.Linear(embedding_dim, out_dim * 2)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        patch_dim = self.patch_dim if self.patch_dim >= 0 else x.ndim + self.patch_dim

        if self.act:
            emb = self.act(emb)
        emb = self.linear(emb)
        # Support 4D (N,C,H,W) and 5D (N,C,F,H,W); patch dim is always first spatial (index 2).
        emb = emb[:, :, None, None, None] if x.ndim == 5 else emb[:, :, None, None]
        scale, shift = emb.chunk(2, dim=1)

        world_size = DistributedEnv.get_world_size()
        patch_size_list = [torch.empty([1], dtype=torch.int64) for _ in range(world_size)]
        dist.all_gather(
            patch_size_list,
            torch.tensor([x.shape[patch_dim]], dtype=torch.int64),
            group=DistributedEnv.get_vae_group()
        )
        patch_size = torch.tensor(patch_size_list).sum().item()

        channels_per_group = x.shape[1] // self.num_groups
        nelements = (
            channels_per_group *
            math.prod(x.shape[2: patch_dim]) * patch_size * math.prod(x.shape[patch_dim + 1:])
        )
        partial_sum = x.sum_to_size(x.shape[0], self.num_groups)
        partial_sum_list = [
            torch.empty([x.shape[0], self.num_groups], dtype=x.dtype, device=x.device)
            for _ in range(world_size)
        ]
        dist.all_gather(partial_sum_list, partial_sum, group=DistributedEnv.get_vae_group())
        group_sum = torch.tensor(partial_sum_list, device=x.device).sum(dim=0)
        E = group_sum / nelements
        partial_var = ((x - E) ** 2).sum_to_size(x.shape[0], self.num_groups)
        partial_var_list = [
            torch.empty([x.shape[0], self.num_groups], dtype=x.dtype, device=x.device)
            for _ in range(world_size)
        ]
        dist.all_gather(partial_var_list, partial_var, group=DistributedEnv.get_vae_group())
        group_var = torch.tensor(partial_var_list, device=x.device).sum(dim=0)
        var = group_var / nelements

        x = (x - E) / torch.sqrt(var + self.eps)
        x = x * (1 + scale) + shift

        return x


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
        patch_dim: int = -2,
    ) -> None:
        self.patch_dim = patch_dim
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
        patch_dim = self.patch_dim if self.patch_dim >= 0 else ndim + self.patch_dim

        vae_group = DistributedEnv.get_vae_group()
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