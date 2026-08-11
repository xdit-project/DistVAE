from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from distvae.models.layers.conv_mixin import PatchConvMixin
from distvae.utils import ParallelContext, normalize_patch_dim

def _patch_axis(conv) -> int:
    """Which entry of a convolution's per-axis tuples describes the axis being split"""
    patch_dim = conv.patch_dim
    if patch_dim < 0:
        patch_dim += conv._patch_ndim()
    return patch_dim - 2


def widest_halo(module: nn.Module) -> int:
    """The most rows any convolution in here will ask a neighbour for

    Neither halo width ever exceeds half the kernel, whatever the stride and the padding: the
    step count either side of a boundary is a ceiling of the same quantity the width is then
    measured back from, and what survives that algebra is `kernel_size // 2` with the stride and
    the padding cancelled out. So the widest kernel over the stack bounds every exchange the run
    will make, and being a property of the weights rather than of the image, it can be read once
    and reread never.
    """
    widest = 0
    # Every patched convolution, by the mixin that gives them their halo rather than by the two
    # plain subclasses: AsymmetricZeroPadConv2d exchanges a halo like the others and is neither
    # of them, so naming the subclasses left its kernel out of the bound this guard is built from.
    for conv in module.modules():
        if not isinstance(conv, PatchConvMixin):
            continue
        kernel = conv.kernel_size
        if isinstance(kernel, tuple):
            kernel = kernel[_patch_axis(conv)]
        widest = max(widest, kernel // 2)
    return widest


def gather_patches(
    patch: torch.Tensor,
    parallel_context: ParallelContext,
) -> Tuple[List[torch.Tensor], List[int]]:
    """All-gather patches that need not be the same size along patch_dim

    dist.all_gather insists every rank contributes the same shape, so a rank holding fewer rows
    than its neighbours cannot take part directly. Each rank pads its patch out to the widest
    before the collective and the padding is sliced off on the far side, so it exists only for
    the length of the transfer and never reaches a convolution.

    Returns each rank's patch in rank order, and the sizes, which callers need to locate their
    own rows within the whole.
    """
    if not isinstance(parallel_context, ParallelContext):
        raise TypeError("gather_patches requires a ParallelContext")
    patch_dim = patch.ndim + normalize_patch_dim(
        parallel_context.patch_dim, patch.ndim, spatial_only=True
    )
    group = parallel_context.group
    world_size = parallel_context.world_size

    # One rank already holds the whole thing, so there is nothing to collect and no other size to
    # discover. Both gathers below would be round trips whose answer is the argument. Callers
    # concatenate what comes back, and cat copies, so handing back the input itself aliases nothing.
    if world_size == 1:
        return [patch], [patch.shape[patch_dim]]

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

    Padding to an even split changes the computation: convolution and attention propagate the
    network's response to padded values into retained rows before any final crop.

    This is also where a band too thin to lend its neighbour a halo is caught, because it is the
    one place every rank works the same sum from the same numbers. The convolutions cannot do it:
    each holds only its own band, bands differ by a unit, and a rank that stopped on its own
    would leave its neighbours waiting on rows from a rank that is no longer sending them.
    """

    def __init__(
        self,
        parallel_context: ParallelContext,
        scale_factor: int = 1,
        halo: int = 0,
    ):
        super().__init__()
        if not isinstance(parallel_context, ParallelContext):
            raise TypeError("Patchify requires a ParallelContext")
        self.parallel_context = parallel_context
        self.group_world_size = parallel_context.world_size
        self.rank_in_vae_group = parallel_context.rank
        self.patch_dim = parallel_context.patch_dim
        self.scale_factor = scale_factor
        self.halo = halo

    def forward(self, hidden_state):
        patch_dim = hidden_state.ndim + normalize_patch_dim(
            self.patch_dim, hidden_state.ndim, spatial_only=True
        )
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
        # A unit is the narrowest a band gets: an encoder is on its way down to one row per unit
        # and a decoder is on its way up from one. So the thinnest band anyone will hold at any
        # point in the run is `band` rows, and a halo wider than that is a rank reaching past its
        # neighbour into a rank it does not border. Erring towards refusal for an encoder whose
        # widest kernel sits early, where the rows have not been spent yet.
        if self.halo > band:
            fits = units // self.halo
            raise ValueError(
                f"Cannot split {size} rows across {self.group_world_size} ranks: that leaves "
                f"{band} row{'' if band == 1 else 's'} per rank at the narrowest, and this VAE "
                f"has a convolution reaching {self.halo} rows past a band into its neighbour's. "
                f"Use at most {fits} rank{'' if fits == 1 else 's'} for this VAE, or tile it "
                f"instead."
            )
        rank = self.rank_in_vae_group
        start = (rank * band + min(rank, remainder)) * factor
        length = (band + (1 if rank < remainder else 0)) * factor
        return hidden_state.narrow(patch_dim, start, length).clone()


class DePatchify(nn.Module):
    def __init__(self, parallel_context: ParallelContext):
        super().__init__()
        if not isinstance(parallel_context, ParallelContext):
            raise TypeError("DePatchify requires a ParallelContext")
        self.parallel_context = parallel_context
        self.patch_dim = parallel_context.patch_dim

    def forward(self, patch_hidden_state):
        patch_dim = patch_hidden_state.ndim + normalize_patch_dim(
            self.patch_dim, patch_hidden_state.ndim, spatial_only=True
        )
        patches, _ = gather_patches(patch_hidden_state, self.parallel_context)
        return torch.cat(patches, dim=patch_dim)
