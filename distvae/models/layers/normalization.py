import math
import torch
import torch.nn as nn
import torch.distributed as dist
from torch import Tensor

from distvae.utils import ParallelContext, normalize_patch_dim


class PatchGroupNorm(nn.GroupNorm):
    """Inference-only GroupNorm over spatial shards held by a VAE process group.

    Each rank supplies its local, potentially uneven shard. Group sums and squared
    deviations are reduced across ``parallel_context.group``, so every rank normalizes
    its shard with the statistics of the complete unsharded tensor. The biased variance
    estimator and affine transform match :class:`torch.nn.GroupNorm`.

    ``parallel_context`` identifies the process group and spatial patch dimension.
    ``forward`` runs without gradient tracking and returns a tensor with the same local
    shape as its input.
    """

    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: float = 1e-5,
        affine: bool = True,
        device=None,
        dtype=None,
        parallel_context: ParallelContext = None,
    ) -> None:
        if not isinstance(parallel_context, ParallelContext):
            raise TypeError("PatchGroupNorm requires a ParallelContext")
        self.parallel_context = parallel_context
        self.patch_dim = parallel_context.patch_dim
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
        patch_dim = ndim + normalize_patch_dim(self.patch_dim, ndim, spatial_only=True)
        vae_group = self.parallel_context.group
        group_world_size = self.parallel_context.world_size
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