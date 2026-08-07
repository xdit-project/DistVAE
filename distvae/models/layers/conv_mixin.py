"""Mixin for shared patch-conv logic used by PatchConv2d and PatchConv3d.

Provides multi-rank metadata, halo exchange, padding adjustment, and the decision
between direct and chunked conv path.
"""

import torch
import torch.distributed as dist
from torch import Tensor

from distvae.utils import DistributedEnv
from distvae.models.layers.conv_utils import (
    get_world_size_and_rank,
    calc_patch_index,
    calc_halo_width,
    calc_halo_width_unit_stride,
    calc_bottom_halo_width,
    calc_top_halo_width,
    exchange_halo,
    adjust_padding_for_patch,
)


class PatchConvMixin:
    """Mixin providing shared multi-rank metadata, halo exchange, padding adjustment, and direct-path check.

    Subclasses must override _patch_ndim() to return 4 (2D) or 5 (3D).
    Expects self.patch_dim, self.block_size, self.kernel_size, self.padding, self.stride,
    self.padding_mode, self._reversed_padding_repeated_twice from the Conv subclass.

    Methods: _patch_ndim (return 4 or 5); _adjust_padding_for_patch (delegate to conv_utils);
    _use_direct_path (True if single rank or all spatial sizes <= block_size);
    _multi_rank_metadata_and_halo (compute halo, exchange it, return extended input + metadata).
    """

    def _patch_ndim(self) -> int:
        """Return 4 for 2D (Conv2d) or 5 for 3D (Conv3d)."""
        raise NotImplementedError("Subclass must override _patch_ndim() to return 4 or 5")

    def _adjust_padding_for_patch(self, padding, rank, world_size, patch_dim: int = 2):
        """Delegate to conv_utils.adjust_padding_for_patch with ndim from _patch_ndim()."""
        return adjust_padding_for_patch(
            padding, rank, world_size, patch_dim, ndim=self._patch_ndim()
        )

    def _check_padding_mode(self, group_world_size: int) -> None:
        """Refuse a padding mode whose values a halo exchange cannot supply.

        Zeros, replicate and reflect all read from within the patch or from nothing, so a rank
        can produce them once its neighbours' rows have arrived. Circular reads from the far
        edge of the image, which belongs to a rank this one does not border, and would
        otherwise wrap silently within the patch and give an answer no one checked.
        """
        if group_world_size > 1 and self.padding_mode == "circular":
            raise NotImplementedError(
                f"{type(self).__name__} cannot shard a convolution padded circularly: its "
                f"padding wraps to the opposite edge of the image, which is not on a "
                f"neighbouring rank. Use a single rank for this VAE, or tile it instead."
            )

    def _use_direct_path(self, input: Tensor) -> bool:
        """Return True if we can run a single conv and crop (no chunking).

        True when block_size is 0 or every spatial dimension of input is <= block_size.
        Otherwise the chunked path is used.
        """
        ndim = self._patch_ndim()
        spatial_sizes = input.shape[2:ndim]
        block_size = self.block_size
        if block_size == 0:
            return True
        if isinstance(block_size, int):
            return all(s <= block_size for s in spatial_sizes)
        return all(
            spatial_sizes[i] <= block_size[i] for i in range(len(spatial_sizes))
        )

    def _multi_rank_metadata_and_halo(
        self,
        input: Tensor,
        halo_buffer: dict = None
    ):
        """Work out the halo this rank needs, exchange it, and return the extended input.

        A strided conv all-gathers each rank's patch size to build the cumulative boundaries
        its halo widths and its output cropping both turn on. A unit-stride conv derives the
        same widths from its kernel and skips the gather; it also has no use for the
        boundaries, so it reports global_start as None.

        Returns (input, patch_dim, patch_size, halo_width, kernel_size_patch_dim,
        padding_patch_dim, stride_patch_dim, global_start, group_world_size, rank_in_group).
        """
        group_world_size, global_rank, rank_in_group, local_rank = get_world_size_and_rank()
        patch_dim = self.patch_dim if self.patch_dim >= 0 else input.ndim + self.patch_dim
        patch_size = input.shape[patch_dim]
        spatial_idx = patch_dim - 2
        kernel_size_patch_dim = (
            self.kernel_size[spatial_idx]
            if isinstance(self.kernel_size, tuple)
            else self.kernel_size
        )
        padding_patch_dim = (
            self.padding[spatial_idx]
            if isinstance(self.padding, tuple)
            else self.padding
        )
        stride_patch_dim = (
            self.stride[spatial_idx]
            if isinstance(self.stride, tuple)
            else self.stride
        )
        prev_bottom_halo_width: int = 0
        next_top_halo_width: int = 0
        if stride_patch_dim == 1:
            # At unit stride the halo depends on the kernel alone, so no rank has to be told
            # where the others' patches begin and the gather below can be skipped. A rank one
            # along is neither first nor last from this rank's point of view, which is why the
            # widths it wants are the plain kernel halves.
            patch_index = None
            halo_width = calc_halo_width_unit_stride(
                rank_in_group, group_world_size, kernel_size_patch_dim
            )
            if rank_in_group != 0:
                prev_bottom_halo_width = kernel_size_patch_dim // 2
            if rank_in_group != group_world_size - 1:
                next_top_halo_width = (kernel_size_patch_dim - 1) // 2
        else:
            # Patchify cuts bands that differ in size wherever the row count does not divide by
            # the rank count, and a strided conv's halo turns on where in the global stride grid
            # a patch starts, so a rank cannot read this off its own patch and has to be told.
            patch_list = [
                torch.zeros(1, dtype=torch.int64, device=input.device)
                for _ in range(group_world_size)
            ]
            dist.all_gather(
                patch_list,
                torch.tensor(
                    [input.shape[patch_dim]],
                    dtype=torch.int64,
                    device=input.device,
                ),
                group=DistributedEnv.get_vae_group(),
            )
            patch_index = calc_patch_index(patch_list)
            halo_width = calc_halo_width(
                rank_in_group,
                patch_index,
                kernel_size_patch_dim,
                padding_patch_dim,
                stride_patch_dim,
            )
            if rank_in_group != 0:
                prev_bottom_halo_width = calc_bottom_halo_width(
                    rank_in_group - 1,
                    patch_index,
                    kernel_size_patch_dim,
                    padding_patch_dim,
                    stride_patch_dim,
                )
            if rank_in_group != group_world_size - 1:
                next_top_halo_width = calc_top_halo_width(
                    rank_in_group + 1,
                    patch_index,
                    kernel_size_patch_dim,
                    padding_patch_dim,
                    stride_patch_dim,
                )
                next_top_halo_width = max(0, next_top_halo_width)
        if self._patch_ndim() == 4:
            # Backstop, not the guard. Bands differ by a unit, so this can be true on one rank and
            # false on its neighbour, and a rank that stops here stops on its way into the
            # exchange below - leaving the others waiting on rows that will not come. Patchify
            # refuses the same case up front, where every rank works it out from the same numbers
            # and they all refuse together. Anything reaching here came in already split.
            assert halo_width[0] <= patch_size and halo_width[1] <= patch_size, (
                "halo width is larger than the patch dimension of input tensor"
            )

        input = exchange_halo(
            input,
            patch_dim,
            patch_index,
            halo_width,
            prev_bottom_halo_width,
            next_top_halo_width,
            group_world_size,
            rank_in_group,
            halo_buffer,
        )

        # Where this rank's patch begins in the whole image. Only a strided conv needs it, and
        # only a strided conv paid to find it out, so at unit stride there is nothing to report.
        global_start = None if patch_index is None else patch_index[rank_in_group]

        # A block trimming the input back onto the global stride grid used to sit here. It never
        # trimmed anything: the top halo is defined as the distance from the patch start back to
        # the last output step before it, so start - halo + padding is that step's position, which
        # is a whole number of strides by construction and leaves nothing to shift by. What it
        # reported was therefore always zero, and both callers unpacked it and never read it.
        return (
            input,
            patch_dim,
            patch_size,
            halo_width,
            kernel_size_patch_dim,
            padding_patch_dim,
            stride_patch_dim,
            global_start,
            group_world_size,
            rank_in_group,
        )
