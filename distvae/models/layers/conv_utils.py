"""Shared helpers for patch-parallel 2D/3D convolution.

This module provides utilities used by PatchConv2d and PatchConv3d: patch boundary
indices, halo widths, chunk-boundary alignment (correct_end/correct_start), crop
slices for trimming conv output to the local patch, padding adjustment at rank
boundaries, and halo exchange between neighboring ranks.
"""

import math
import os
from typing import List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch import Tensor

from distvae.utils import DistributedEnv, ParallelContext


def get_world_size_and_rank(parallel_context: Optional[ParallelContext] = None):
    """Return distributed group and rank info from DistributedEnv.

    Returns:
        Tuple of (group_world_size, global_rank, rank_in_group, local_rank).
    """
    if parallel_context is not None:
        return (
            parallel_context.world_size,
            dist.get_rank() if dist.is_initialized() else 0,
            parallel_context.rank,
            int(os.environ.get("LOCAL_RANK", 0)),
        )
    group_world_size = DistributedEnv.get_group_world_size()
    global_rank = DistributedEnv.get_global_rank()
    rank_in_group = DistributedEnv.get_rank_in_vae_group()
    local_rank = DistributedEnv.get_local_rank()
    return group_world_size, global_rank, rank_in_group, local_rank


def calc_patch_index(patch_list: List[Tensor]):
    """Build cumulative patch boundaries from per-rank patch sizes.

    Args:
        patch_list: List of 1-element tensors; patch_list[i].item() is the size
            of rank i's patch along the patch dimension.

    Returns:
        List of length len(patch_list) + 1. height_index[i] is the global start
        index of patch i; height_index[-1] is the total length. The first element
        is always 0.
    """
    patch_list = torch.cat(patch_list, dim=0)
    return torch.cat(
        [
            torch.zeros(1, device=patch_list.device, dtype=patch_list.dtype),
            torch.cumsum(patch_list, dim=0),
        ],
        dim=0,
    ).cpu().tolist()


def calc_bottom_halo_width(rank, height_index, kernel_size, padding=0, stride=1):
    """Width of halo below this rank's patch (needed so next rank can run conv).

    The halo is the extra input region below the patch that the next rank needs
    for its convolution. height_index gives global patch boundaries; rank is this
    rank's index. The last rank has no "bottom" neighbor and returns 0. The
    formula computes how many output steps occur before the boundary, then
    converts to the required input width.

    Args:
        rank: This rank's index (0 to world_size - 1).
        height_index: Cumulative patch boundaries from calc_patch_index.
        kernel_size, padding, stride: Conv parameters along the patch dimension.

    Returns:
        Number of rows (or patch-dim elements) to receive from the next rank.
    """
    assert rank >= 0, "rank should not be smaller than 0"
    assert rank < len(height_index) - 1, "rank should be smaller than the length of height_index - 1"
    assert padding >= 0, "padding should not be smaller than 0"
    assert stride > 0, "stride should be larger than 0"
    world_size = len(height_index) - 1
    if rank == world_size - 1:
        return 0
    nstep_before_bottom = (height_index[rank + 1] + padding - (kernel_size - 1) // 2 + stride - 1) // stride
    assert nstep_before_bottom > 0, "nstep_before_bottom should be larger than 0"
    bottom_halo_width =  (nstep_before_bottom - 1) * stride + kernel_size - padding - height_index[rank + 1]
    return max(0, bottom_halo_width)


def calc_top_halo_width(rank, height_index, kernel_size, padding=0, stride=1):
    """Width of halo above this rank's patch (needed from previous rank).

    The halo is the extra input region above the patch that this rank needs for
    convolution. Rank 0 has no "top" neighbor and returns 0. The formula
    computes the number of output steps that fall before the patch start, then
    converts to the required input width above the patch.

    Args:
        rank: This rank's index.
        height_index: Cumulative patch boundaries from calc_patch_index.
        kernel_size, padding, stride: Conv parameters along the patch dimension.

    Returns:
        Number of rows (or patch-dim elements) to receive from the previous rank.
    """
    assert rank >= 0, "rank should not be smaller than 0"
    assert rank < len(height_index) - 1, "rank should be smaller than the length of height_index - 1"
    assert padding >= 0, "padding should not be smaller than 0"
    assert stride > 0, "stride should be larger than 0"

    if rank == 0:
        return 0
    nstep_before_top = (height_index[rank] + padding - (kernel_size - 1) // 2 + stride - 1) // stride
    top_halo_width = height_index[rank] - (nstep_before_top * stride - padding)
    return top_halo_width


def calc_halo_width(rank, height_index, kernel_size, padding=0, stride=1):
    """Compute (top_halo_width, bottom_halo_width) for this rank along the patch dimension.

    The halo is the region used for convolution but not included in this rank's
    output. The first rank forces top to 0; the last rank (world_size - 1, inferred
    from len(height_index) - 1 or DistributedEnv.get_group_world_size()) forces
    bottom to 0.

    Returns:
        Tuple (top_halo_width, bottom_halo_width) in patch-dim elements.
    """
    halo_width = [
        calc_top_halo_width(rank, height_index, kernel_size, padding, stride),
        calc_bottom_halo_width(rank, height_index, kernel_size, padding, stride)
    ]
    if rank == 0:
        halo_width[0] = 0
    elif rank == len(height_index) - 2:
        halo_width[1] = 0
    return tuple(halo_width)


def calc_halo_width_unit_stride(rank, world_size, kernel_size):
    """Compute (top, bottom) halo widths for a stride-1 conv, asking no other rank anything.

    Under unit stride every term that mentions where a patch sits cancels out of
    calc_top_halo_width and calc_bottom_halo_width, and the halo comes down to the kernel:
    a rank needs the (kernel_size - 1) // 2 rows above it that its first output row reads,
    and kernel_size // 2 rows below it for its last. Padding cancels too, because it shifts
    the output grid and the patch start by the same amount.

    That matters because the alternative is an all_gather of one integer per convolution,
    and on a Wan decode those gathers are half of every collective the model makes.

    Args:
        rank: This rank's index within the VAE group.
        world_size: Size of the VAE group.
        kernel_size: Kernel size along the patch dimension.

    Returns:
        Tuple (top_halo_width, bottom_halo_width), matching calc_halo_width at stride 1.
    """
    top = 0 if rank == 0 else (kernel_size - 1) // 2
    bottom = 0 if rank == world_size - 1 else kernel_size // 2
    return top, bottom


def correct_end(end, kernel_size, stride):
    """Adjust chunk end so conv output at that boundary aligns with stride.

    Given a nominal chunk end index, returns the smallest end >= that value such
    that the conv output at the boundary aligns with the stride, so chunked conv
    outputs can be concatenated correctly. Used in the chunked conv path.

    Args:
        end: Nominal end index for the chunk.
        kernel_size, stride: Conv parameters along that dimension.

    Returns:
        Aligned end index (input-space).
    """
    return ((end + stride - 1) // stride - 1) * stride + kernel_size


def correct_start(start, stride):
    """Align chunk start to stride so conv output indices line up.

    Returns the start index aligned to the stride grid, for consistent chunk
    boundaries in the chunked conv path.
    """
    return ((start + stride - 1) // stride) * stride


def build_crop_slice(
    patch_dim: int,
    patch_size: int,
    halo_width: tuple,
    out_len: int,
    ndim: int,
    global_start: int = None,
    kernel_size: int = None,
    padding: int = None,
    stride: int = None,
    input_halo_width: tuple = None,
) -> tuple:
    """Build a tuple of slices to crop conv output to the valid patch region.

    When global position information is provided (global_start, etc.),
    computes exact output indices based on global coordinates to ensure alignment
    across ranks (critical for stride > 1). Otherwise falls back to simple halo-based crop.

    Args:
        patch_dim: The spatial dimension that is split across ranks (0-based).
        patch_size: Size of this rank's patch along patch_dim (input space).
        halo_width: (top_halo_width, bottom_halo_width) in input space before conv.
        out_len: Length of the full conv output along patch_dim.
        ndim: Number of dimensions (4 for 2D conv, 5 for 3D).
        global_start: Global start index of this rank's patch (for stride alignment).
        kernel_size: Kernel size along patch_dim (for exact output calculation).
        padding: Padding along patch_dim.
        stride: Stride along patch_dim.
        input_halo_width: (top, bottom) halo in input space.

    Returns:
        Tuple of slices suitable for indexing the conv output tensor.
    """
    # If we have global position info, compute exact output range
    if (global_start is not None and kernel_size is not None and
        padding is not None and stride is not None and
        input_halo_width is not None and stride > 1):

        halo_start = global_start - input_halo_width[0]
        patch_end = global_start + patch_size
        half_k = (kernel_size - 1) // 2

        # Global output indices owned by this rank (kernel-center convention,
        # matching calc_top_halo_width / calc_bottom_halo_width).
        # Output i has its kernel center at  i*stride + half_k - padding  in input space.
        min_i_global = math.ceil((global_start + padding - half_k) / stride)
        max_i_global = math.floor((patch_end - 1 + padding - half_k) / stride)

        # Map global output indices to local output indices. Only rank 0 keeps the
        # left-side padding; all other ranks have it zeroed by adjust_padding_for_patch.
        local_pad_left = padding if global_start == 0 else 0
        # (local_pad_left - padding - halo_start) is always a multiple of stride
        # by construction of input_halo_width[0]; use //.
        shift = (local_pad_left - padding - halo_start) // stride

        min_j = max(0, min_i_global + shift)
        max_j = min(out_len - 1, max_i_global + shift)

        if min_j > max_j:
            patch_slice = slice(0, 0)  # empty
        else:
            patch_slice = slice(min_j, max_j + 1)

    elif out_len == patch_size:
        patch_slice = slice(0, patch_size)
    else:
        # stride == 1: output halo width equals input halo width.
        output_halo_top = halo_width[0] if halo_width else 0
        patch_slice = slice(output_halo_top, output_halo_top + patch_size)

    return (
        (slice(None),) * patch_dim
        + (patch_slice,)
        + (slice(None),) * (ndim - 1 - patch_dim)
    )


def adjust_padding_for_patch(
    padding: Union[int, tuple],
    rank: int,
    world_size: int,
    patch_dim: int,
    ndim: int,
) -> tuple:
    """Zero out padding on the outside edges of the patch dimension.

    For patch-parallel conv we must not pad across rank boundaries: rank 0 zeros
    the right (high-index) padding, last rank zeros the left (low-index), middle
    ranks zero both. ndim 4 => Conv2d padding layout (left_h, right_h, left_w, right_w);
    ndim 5 => Conv3d (left_f, right_f, left_h, right_h, left_w, right_w). patch_dim
    selects which pair (2=first spatial, 3=second, 4=third for 5D).

    Returns:
        Padding tuple with the appropriate sides set to 0.
    """
    if ndim == 4:
        if isinstance(padding, tuple):
            padding = list(padding)
        else:
            padding = [padding] * 4
        right_idx = (3, 1)[patch_dim - 2]
        left_idx = (2, 0)[patch_dim - 2]
    else:
        assert ndim == 5
        if isinstance(padding, tuple):
            padding = list(padding)
        else:
            padding = [padding] * 6
        left_idx, right_idx = {2: (4, 5), 3: (2, 3), 4: (0, 1)}[patch_dim]
    if rank == 0:
        padding[right_idx] = 0
    elif rank == world_size - 1:
        padding[left_idx] = 0
    else:
        padding[left_idx] = 0
        padding[right_idx] = 0
    return tuple(padding)


def exchange_halo(
    input: Tensor,
    patch_dim: int,
    patch_index: list,
    halo_width: tuple,
    prev_bottom_halo_width: int,
    next_top_halo_width: int,
    group_world_size: int,
    rank_in_group: int,
    halo_buffer: dict = None,
    parallel_context: Optional[ParallelContext] = None,
) -> Tensor:
    """Exchange halo regions with previous and next ranks; return extended local tensor.

    Send: bottom halo to next rank (size next_top_halo_width), top halo to prev
    (size prev_bottom_halo_width). Receive: top halo from prev (halo_width[0]),
    bottom halo from next (halo_width[1]). Concatenate [top_halo_recv, input,
    bottom_halo_recv] along patch_dim and return. All four are issued as one
    batch and waited on together.

    Args:
        patch_index: Cumulative patch boundaries, or None when the caller never gathered them.
            They only serve the bounds checks here, which are skipped in that case rather than
            paid for with a collective.
        halo_buffer: Optional dict to cache/reuse comms buffers for better performance
    """
    ndim = input.ndim
    indices_end = [slice(None)] * ndim
    indices_end[patch_dim] = slice(-next_top_halo_width, None)
    indices_start = [slice(None)] * ndim
    indices_start[patch_dim] = slice(0, prev_bottom_halo_width)

    vae_group = (
        parallel_context.group
        if parallel_context is not None
        else DistributedEnv.get_vae_group()
    )
    ops = []
    top_halo_recv = None
    bottom_halo_recv = None
    global_rank_of_next = None
    global_rank_of_prev = None

    def recv_buffer(name: str, width: int) -> Tensor:
        recv_shape = list(input.shape)
        recv_shape[patch_dim] = width
        if halo_buffer is None:
            return torch.empty(recv_shape, dtype=input.dtype, device=input.device)
        key = (name, tuple(recv_shape), input.dtype, input.device)
        if key not in halo_buffer:
            halo_buffer[key] = torch.empty(
                recv_shape, dtype=input.dtype, device=input.device
            )
        return halo_buffer[key]

    if next_top_halo_width > 0:
        global_rank_of_next = (
            parallel_context.global_rank(rank_in_group + 1)
            if parallel_context is not None
            else DistributedEnv.get_global_rank_from_group_rank(rank_in_group + 1)
        )
        bottom_halo_send = input[tuple(indices_end)].contiguous()
        ops.append(dist.P2POp(dist.isend, bottom_halo_send, global_rank_of_next, group=vae_group))
    if halo_width[0] > 0:
        assert patch_index is None or (
            patch_index[rank_in_group] - halo_width[0] >= patch_index[rank_in_group - 1]
        ), "width of top halo region is larger than the input tensor of prev rank"
        top_halo_recv = recv_buffer("top_recv", halo_width[0])
        global_rank_of_prev = (
            parallel_context.global_rank(rank_in_group - 1)
            if parallel_context is not None
            else DistributedEnv.get_global_rank_from_group_rank(rank_in_group - 1)
        )
        ops.append(dist.P2POp(dist.irecv, top_halo_recv, global_rank_of_prev, group=vae_group))
    if prev_bottom_halo_width > 0:
        top_halo_send = input[tuple(indices_start)].contiguous()
        if global_rank_of_prev is None:
            global_rank_of_prev = (
                parallel_context.global_rank(rank_in_group - 1)
                if parallel_context is not None
                else DistributedEnv.get_global_rank_from_group_rank(rank_in_group - 1)
            )
        ops.append(dist.P2POp(dist.isend, top_halo_send, global_rank_of_prev, group=vae_group))
    if halo_width[1] > 0:
        assert patch_index is None or (
            patch_index[rank_in_group + 1] + halo_width[1] <= patch_index[rank_in_group + 2]
        ), "width of bottom halo region is larger than the input tensor of next rank"
        bottom_halo_recv = recv_buffer("bottom_recv", halo_width[1])
        if global_rank_of_next is None:
            global_rank_of_next = (
                parallel_context.global_rank(rank_in_group + 1)
                if parallel_context is not None
                else DistributedEnv.get_global_rank_from_group_rank(rank_in_group + 1)
            )
        ops.append(dist.P2POp(dist.irecv, bottom_halo_recv, global_rank_of_next, group=vae_group))

    # One batch rather than four separate calls. The two directions are independent, so blocking
    # in the receive from the previous rank before even offering the send to the previous rank
    # exposed a round trip that did not have to be exposed; and NCCL builds a fresh two-rank
    # communicator for every unbatched point-to-point op issued on a wider group.
    if ops:
        for work in dist.batch_isend_irecv(ops):
            work.wait()

    if halo_width[0] < 0:
        trim_slice = [slice(None)] * ndim
        trim_slice[patch_dim] = slice(-halo_width[0], None)
        input = input[tuple(trim_slice)]
    if top_halo_recv is not None:
        input = torch.cat([top_halo_recv, input], dim=patch_dim)
    if bottom_halo_recv is not None:
        input = torch.cat([input, bottom_halo_recv], dim=patch_dim)
    return input

