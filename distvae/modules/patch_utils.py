from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from distvae.utils import DistributedEnv


def gather_patches(patch: torch.Tensor, patch_dim: int) -> Tuple[List[torch.Tensor], List[int]]:
    """All-gather patches that need not be the same size along patch_dim

    dist.all_gather insists every rank contributes the same shape, so a rank holding fewer rows
    than its neighbours cannot take part directly. Each rank pads its patch out to the widest
    before the collective and the padding is sliced off on the far side, so it exists only for
    the length of the transfer and never reaches a convolution.

    Returns each rank's patch in rank order, and the sizes, which callers need to locate their
    own rows within the whole.
    """
    group = DistributedEnv.get_vae_group()
    world_size = DistributedEnv.get_group_world_size()

    gathered_sizes = [
        torch.empty(1, dtype=torch.int64, device=patch.device) for _ in range(world_size)
    ]
    dist.all_gather(
        gathered_sizes,
        torch.tensor([patch.shape[patch_dim]], dtype=torch.int64, device=patch.device),
        group=group,
    )
    sizes = [int(size.item()) for size in gathered_sizes]
    widest = max(sizes)

    padded = patch
    if patch.shape[patch_dim] < widest:
        # torch.nn.functional.pad counts its pairs from the last dimension backwards.
        pad = [0] * (2 * patch.ndim)
        pad[2 * (patch.ndim - patch_dim - 1) + 1] = widest - patch.shape[patch_dim]
        padded = F.pad(patch, tuple(pad))

    buffers = [torch.empty_like(padded) for _ in range(world_size)]
    dist.all_gather(buffers, padded.contiguous(), group=group)

    return [
        buffer.narrow(patch_dim, 0, size) for buffer, size in zip(buffers, sizes)
    ], sizes


class Patchify(nn.Module):
    """Hands each rank one contiguous band of rows along the patch dimension

    Bands are cut in whole multiples of scale_factor, the amount the VAE narrows or widens this
    axis by, so that every band begins on the grid the strided convolutions downstream step
    along and the rows a rank produces are its own. Bands therefore differ in size when they do
    not divide evenly, which is why the gathers pad for transport.

    Padding the tensor up to a size that did divide would be simpler and is what this used to
    do, but it is not the same computation: after the first convolution the pad is no longer
    zeros but the network's answer to zeros, and it reaches the kept rows through every
    receptive field and every attention that follows, however much is cropped afterwards.
    """

    def __init__(
        self,
        patch_dim: int = -2,
        scale_factor: int = 1,
    ):
        super().__init__()
        self.group_world_size = DistributedEnv.get_group_world_size()
        self.rank_in_vae_group = DistributedEnv.get_rank_in_vae_group()
        self.patch_dim = patch_dim
        self.scale_factor = scale_factor

    def forward(self, hidden_state):
        patch_dim = self.patch_dim if self.patch_dim >= 0 else hidden_state.ndim + self.patch_dim
        size = hidden_state.shape[patch_dim]
        factor = max(1, self.scale_factor)
        if size % factor:
            raise ValueError(
                f"Cannot split {size} rows into multiples of {factor}: the VAE narrows this "
                f"axis by {factor}, so a band that is not a whole multiple of it would land "
                f"between output rows."
            )
        units = size // factor
        if units < self.group_world_size:
            raise ValueError(
                f"Cannot split {size} rows across {self.group_world_size} ranks: that leaves "
                f"{units} band{'' if units == 1 else 's'} of {factor} rows to go round. Use at "
                f"most {units} rank{'' if units == 1 else 's'} for this VAE."
            )
        # The ranks that come first each take one extra band where the count does not divide.
        band, remainder = divmod(units, self.group_world_size)
        rank = self.rank_in_vae_group
        start = (rank * band + min(rank, remainder)) * factor
        length = (band + (1 if rank < remainder else 0)) * factor
        return hidden_state.narrow(patch_dim, start, length).clone()


class DePatchify(nn.Module):
    def __init__(self, patch_dim: int = -2):
        super().__init__()
        self.patch_dim = patch_dim

    def forward(self, patch_hidden_state):
        patch_dim = (
            self.patch_dim if self.patch_dim >= 0 else patch_hidden_state.ndim + self.patch_dim
        )
        patches, _ = gather_patches(patch_hidden_state, patch_dim)
        return torch.cat(patches, dim=patch_dim)
