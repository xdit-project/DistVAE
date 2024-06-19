import numbers
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch import Tensor

from diffusers.models.activations import get_activation

#TODO test and fix
class PatchAdaGroupNorm(nn.Module):
    def __init__(
        self, embedding_dim: int, out_dim: int, num_groups: int, act_fn: Optional[str] = None, eps: float = 1e-5
    ):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps

        if act_fn is None:
            self.act = None
        else:
            self.act = get_activation(act_fn)

        self.linear = nn.Linear(embedding_dim, out_dim * 2)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        if self.act:
            emb = self.act(emb)
        emb = self.linear(emb)
        emb = emb[:, :, None, None]
        scale, shift = emb.chunk(2, dim=1)

        world_size = dist.get_world_size()
        height_list = [torch.empty([1], dtype=torch.int64) for _ in range(world_size)]
        dist.all_gather(height_list, torch.tensor([x.shape[2]], dtype=torch.int64))
        height = torch.tensor(height_list).sum()

        channels_per_group = x.shape[1] // self.num_groups
        
        partial_sum = x.sum_to_size(x.shape[0], self.num_groups)
        partial_sum_list = [torch.empty([x.shape[0], self.num_groups], dtype=x.dtype, device=x.device) for _ in range(world_size)]
        dist.all_gather(partial_sum_list, partial_sum)
        group_sum = torch.tensor(partial_sum_list).sum(dim=0)
        E = group_sum / (channels_per_group * height * x.shape[-1])
        
        partial_var = ((x - E) ** 2).sum_to_size(x.shape[0], self.num_groups)
        partial_var_list = [torch.empty([x.shape[0], self.num_groups], dtype=x.dtype, device=x.device) for _ in range(world_size)]
        dist.all_gather(partial_var_list, partial_var)
        group_var = torch.tensor(partial_var_list).sum(dim=0)
        var = group_var / (channels_per_group * height * x.shape[-1])

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

    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5, affine: bool = True,
                 device=None, dtype=None) -> None:
        super().__init__(num_groups=num_groups, num_channels=num_channels, eps=eps, 
                         affine=affine, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        # get height
        height = torch.tensor(x.shape[-2], dtype=torch.int64, device=x.device)
        dist.all_reduce(height)

        channels_per_group = x.shape[1] // self.num_groups
        nelements_rank = channels_per_group * x.shape[-2] * x.shape[-1]
        nelements = channels_per_group * height * x.shape[-1]

        x = x.view(x.shape[0], self.num_groups, -1, x.shape[-2], x.shape[-1])
        group_sum = x.mean(dim=(2,3,4), dtype=torch.float32)
        group_sum = group_sum * nelements_rank
        dist.all_reduce(group_sum)
        # shape: [bs, num_groups, 1, 1, 1]
        E = (group_sum / nelements)[:, :, None, None, None].to(x.dtype)

        group_var_sum = torch.empty((x.shape[0], self.num_groups), dtype=torch.float32, device=x.device)
        torch.var(x, dim=(2,3,4), out=group_var_sum)
        group_var_sum = group_var_sum * nelements_rank
        dist.all_reduce(group_var_sum)
        # shape: [bs, num_groups, 1, 1, 1]
        var = (group_var_sum / nelements)[:, :, None, None, None].to(x.dtype)

        x = (x - E) / torch.sqrt(var + self.eps)
        x = x.view(x.shape[0], -1, x.shape[-2], x.shape[-1])
        x = x * self.weight[None, :, None, None] + self.bias[None, :, None, None]
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