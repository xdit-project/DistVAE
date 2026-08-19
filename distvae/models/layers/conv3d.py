"""PatchConv3d: 5D convolution with H/W patch parallelism for distributed VAE.

When world size is 1, behaves as nn.Conv3d. When world size > 1, gathers patch
sizes, exchanges halos along the patch dimension (H or W), then either runs a
single conv and crops (direct path) or splits the padded input into overlapping
chunks, convs each chunk, concatenates, and crops (chunked path). Supports
patch_dim in {-2, -1, 3, 4} for H and W. Dilation is not supported.
"""

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from distvae.models.layers.conv_utils import (
    get_world_size_and_rank,
    chunk_bounds,
    build_crop_slice,
)
from distvae.models.layers.conv_mixin import PatchConvMixin
from distvae.utils import ParallelContext, normalize_patch_dim


Size3 = Union[int, Tuple[int, int, int]]


def _triple(value: Size3) -> Tuple[int, int, int]:
    return value if isinstance(value, tuple) else (value, value, value)


class PatchConv3d(nn.Conv3d, PatchConvMixin):
    """3D convolution with H/W patch parallelism.

    ``patch_dim`` selects height or width for splitting across ranks. ``block_size``
    controls local convolution chunking across all three convolution dimensions:
    an integer applies one limit to F, H, and W, while a tuple is ordered (F, H, W).
    Zero, or all dimensions fitting their limits, selects the direct path. Dilation
    must be 1.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Size3,
        stride: Size3 = 1,
        padding: Union[str, Size3] = 0,
        dilation: Size3 = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',  # TODO: refine this type
        device=None,
        dtype=None,
        block_size: Union[int, Tuple[int, int, int]] = 0,
        parallel_context: ParallelContext = None,
    ) -> None:
        """Initialize H/W sharding and optional local (F, H, W) chunk limits.

        ``patch_dim`` accepts H (-2 or 3) or W (-1 or 4). ``block_size`` is zero
        for the direct path, an integer shared by F/H/W, or an (F, H, W) tuple.
        """
        if isinstance(dilation, int):
            assert dilation == 1, "dilation is not supported in PatchConv3d"
        else:
            for i in dilation:
                assert i == 1, "dilation is not supported in PatchConv3d"
        if not isinstance(parallel_context, ParallelContext):
            raise TypeError("PatchConv3d requires a ParallelContext")
        self.block_size = block_size
        self.parallel_context = parallel_context
        self.patch_dim = normalize_patch_dim(
            parallel_context.patch_dim, 5, spatial_only=True
        )
        self.halo_buffer = {}
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, padding_mode, device, dtype)

    def _patch_ndim(self) -> int:
        """Return 5 for 3D (N, C, F, H, W)."""
        return 5

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        bs, channels, f, h, w = input.shape

        group_world_size, rank_in_group = get_world_size_and_rank(self.parallel_context)

        # Single rank: use standard F.conv3d (with optional padding_mode).
        if (group_world_size == 1):
            if self.padding_mode != 'zeros':
                return F.conv3d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                                weight, bias, self.stride,
                                _triple(0), self.dilation, self.groups)
            return F.conv3d(input, weight, bias, self.stride,
                            self.padding, self.dilation, self.groups)
        # Multi-rank: get extended input and metadata from mixin (halo_width, global_start, etc.), then choose direct or chunked path.
        else:
            self._check_padding_mode(group_world_size)
            (
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
            ) = self._multi_rank_metadata_and_halo(input, self.halo_buffer)
            conv_res: Tensor
            padding = self._adjust_padding_for_patch(
                self._reversed_padding_repeated_twice,
                rank=rank_in_group,
                world_size=group_world_size,
                patch_dim=patch_dim,
            )
            bs, channels, f, h, w = input.shape
            # Direct path: one conv over the extended (halo-padded) input, then crop to this rank's patch output.
            if self._use_direct_path(input):
                if self.padding_mode != 'zeros':
                    conv_res = F.conv3d(F.pad(input, padding, mode=self.padding_mode),
                                        weight, bias, self.stride,
                                        _triple(0), self.dilation, self.groups)
                else:
                    # Fast path: stride 1, padding 1, kernel 3 => no explicit pad, conv then crop.
                    if (
                        stride_patch_dim == 1 and
                        padding_patch_dim == 1 and
                        kernel_size_patch_dim == 3
                    ):
                        conv_res = F.conv3d(input, weight, bias, self.stride,
                                            self.padding, self.dilation, self.groups)
                    else:
                        conv_res = F.conv3d(F.pad(input, padding, "constant", 0.0),
                                            weight, bias, self.stride,
                                            _triple(0), self.dilation, self.groups)

                # Always apply cropping when halos are present to remove halo regions from output
                # This prevents rank boundary artifacts for all convolution configurations.
                # build_crop_slice also recognises the output that is already patch-sized, which
                # is what the branches above that pad only the outer edges produce: there the
                # halo stands in for the padding those branches dropped, so nothing is left over
                # to crop and cropping anyway would eat into the patch itself.
                if halo_width[0] > 0 or halo_width[1] > 0:
                    crop_slice = build_crop_slice(
                        patch_dim, patch_size, halo_width, conv_res.shape[patch_dim], ndim=5,
                        global_start=global_start,
                        kernel_size=kernel_size_patch_dim,
                        padding=padding_patch_dim,
                        stride=stride_patch_dim,
                        input_halo_width=halo_width,
                    )
                    conv_res = conv_res[tuple(crop_slice)].contiguous()

                return conv_res
            # Chunked path: pad input, split into overlapping chunks along F, H, W; conv each chunk with padding=0; concat outputs; crop to this rank's patch.
            else:
                if self.padding_mode != "zeros":
                    input = F.pad(input, padding, mode=self.padding_mode)
                elif self.padding != 0:
                    input = F.pad(input, padding, mode="constant")

                _, _, f, h, w = input.shape
                # nn.Conv3d normalises all three of these to triples in its own __init__, so they
                # are read as triples rather than tested for which they are.
                block_f, block_h, block_w = _triple(self.block_size)
                kernel_f, kernel_h, kernel_w = _triple(self.kernel_size)
                stride_f, stride_h, stride_w = _triple(self.stride)
                frames = chunk_bounds(f, block_f, kernel_f, stride_f)
                rows = chunk_bounds(h, block_h, kernel_h, stride_h)
                columns = chunk_bounds(w, block_w, kernel_w, stride_w)

                outputs = torch.cat([
                    torch.cat([
                        torch.cat([
                            F.conv3d(
                                input[:, :, first:last, top:bottom, left:right],
                                weight,
                                bias,
                                self.stride,
                                0,
                                self.dilation,
                                self.groups,
                            )
                            for left, right in columns
                        ], dim=-1)
                        for top, bottom in rows
                    ], dim=-2)
                    for first, last in frames
                ], dim=-3)
                crop_slice = build_crop_slice(
                    patch_dim, patch_size, halo_width, outputs.shape[patch_dim], ndim=5,
                    global_start=global_start,
                    kernel_size=kernel_size_patch_dim,
                    padding=padding_patch_dim,
                    stride=stride_patch_dim,
                    input_halo_width=halo_width,
                )
                return outputs[tuple(crop_slice)].contiguous()
