"""PatchConv3d: 5D convolution with patch-dim parallelism for distributed VAE.

When world size is 1, behaves as nn.Conv3d. When world size > 1, gathers patch
sizes, exchanges halos along the patch dimension (F, H, or W), then either runs a
single conv and crops (direct path) or splits the padded input into overlapping
chunks, convs each chunk, concatenates, and crops (chunked path). Supports
patch_dim in {-3, -2, -1, 2, 3, 4} for F, H, W. Dilation is not supported.
"""

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from torch.nn.modules.utils import _triple
from torch.nn.common_types import _size_3_t

from distvae.models.layers.conv_utils import (
    get_world_size_and_rank,
    correct_end,
    correct_start,
    build_crop_slice,
)
from distvae.models.layers.conv_mixin import PatchConvMixin


class PatchConv3d(nn.Conv3d, PatchConvMixin):
    """3D convolution with patch-dim parallelism; subclasses nn.Conv3d and PatchConvMixin.

    patch_dim selects which spatial dimension is split across ranks (F=frame, H=height,
    W=width). block_size controls when the chunked path is used: 0 or all spatial
    sizes <= block_size => direct path (one conv + crop); otherwise chunked path.
    Dilation must be 1.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_3_t,
        stride: _size_3_t = 1,
        padding: Union[str, _size_3_t] = 0,
        dilation: _size_3_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',  # TODO: refine this type
        device=None,
        dtype=None,
        block_size: Union[int, Tuple[int, int, int]] = 0,
        patch_dim: int = -2,
        use_uniform_patch: bool = False,
    ) -> None:
        """patch_dim: which spatial dim is split (F=-3/3, H=-2/2, W=-1/4). block_size: 0 => prefer direct path; int or (F,H,W) => chunked when any spatial > block_size."""
        if isinstance(dilation, int):
            assert dilation == 1, "dilation is not supported in PatchConv3d"
        else:
            for i in dilation:
                assert i == 1, "dilation is not supported in PatchConv3d"
        assert patch_dim in (-3, -2, -1, 2, 3, 4), (
            "PatchConv3d patch_dim must be F (-3 or 3) or H (-2 or 2) or W (-1 or 4)"
        )
        self.block_size = block_size
        self.patch_dim = patch_dim
        self.use_uniform_patch = use_uniform_patch
        self.halo_buffer = {}
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, padding_mode, device, dtype)

    def _patch_ndim(self) -> int:
        """Return 5 for 3D (N, C, F, H, W)."""
        return 5

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        bs, channels, f, h, w = input.shape

        group_world_size, global_rank, rank_in_group, local_rank = get_world_size_and_rank()

        # Single rank: use standard F.conv3d (with optional padding_mode).
        if (group_world_size == 1):
            if self.padding_mode != 'zeros':
                return F.conv3d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                                weight, bias, self.stride,
                                _triple(0), self.dilation, self.groups)
            return F.conv3d(input, weight, bias, self.stride,
                            self.padding, self.dilation, self.groups)
        # Multi-rank: get extended input and metadata from mixin (patch_index, halo_width, etc.), then choose direct or chunked path.
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
                patch_index,
                group_world_size,
                rank_in_group,
                stride_shift,
            ) = self._multi_rank_metadata_and_halo(input, self.use_uniform_patch, self.halo_buffer)
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
                        global_start=patch_index[rank_in_group],
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
                if isinstance(self.block_size, int):
                    num_chunks_in_f = (f + self.block_size - 1) // self.block_size
                    num_chunks_in_h = (h + self.block_size - 1) // self.block_size
                    num_chunks_in_w = (w + self.block_size - 1) // self.block_size
                else:
                    num_chunks_in_f = (f + self.block_size[0] - 1) // self.block_size[0]
                    num_chunks_in_h = (h + self.block_size[1] - 1) // self.block_size[1]
                    num_chunks_in_w = (w + self.block_size[2] - 1) // self.block_size[2]
                unit_chunk_size_f = f // num_chunks_in_f
                unit_chunk_size_h = h // num_chunks_in_h
                unit_chunk_size_w = w // num_chunks_in_w
                if isinstance(self.kernel_size, int):
                    kernel_size_f, kernel_size_h, kernel_size_w = self.kernel_size, self.kernel_size, self.kernel_size
                else:
                    kernel_size_f, kernel_size_h, kernel_size_w = self.kernel_size
                if isinstance(self.stride, int):
                    stride_f, stride_h, stride_w = self.stride, self.stride, self.stride
                else:
                    stride_f, stride_h, stride_w = self.stride

                # Chunk boundaries aligned via correct_end/correct_start so conv outputs line up when concatenated.
                outputs = []
                for idx_f in range(num_chunks_in_f):
                    outer_output = []
                    for idx_h in range(num_chunks_in_h):
                        inner_output = []
                        for idx_w in range(num_chunks_in_w):
                            start_f = idx_f * unit_chunk_size_f
                            start_w = idx_w * unit_chunk_size_w
                            start_h = idx_h * unit_chunk_size_h
                            end_f = (idx_f + 1) * unit_chunk_size_f
                            end_w = (idx_w + 1) * unit_chunk_size_w
                            end_h = (idx_h + 1) * unit_chunk_size_h
                            if idx_f + 1 < num_chunks_in_f:
                                end_f = correct_end(end_f, kernel_size_f, stride_f)
                            else:
                                end_f = f
                            if idx_w + 1 < num_chunks_in_w:
                                end_w = correct_end(end_w, kernel_size_w, stride_w)
                            else:
                                end_w = w
                            if idx_h + 1 < num_chunks_in_h:
                                end_h = correct_end(end_h, kernel_size_h, stride_h)
                            else:
                                end_h = h
                            if idx_f > 0:
                                start_f = correct_start(start_f, stride_f)
                            if idx_w > 0:
                                start_w = correct_start(start_w, stride_w)
                            if idx_h > 0:
                                start_h = correct_start(start_h, stride_h)

                            inner_output.append(
                                F.conv3d(
                                    input[:, :, start_f:end_f, start_h:end_h, start_w:end_w],
                                    weight,
                                    bias,
                                    self.stride,
                                    0,
                                    self.dilation,
                                    self.groups,
                                )
                            )
                        outer_output.append(torch.cat(inner_output, dim=-1))
                    outputs.append(torch.cat(outer_output, dim=-2))
                outputs = torch.cat(outputs, dim=-3)
                # Get global position for precise output cropping when stride > 1
                global_start = patch_index[rank_in_group]
                crop_slice = build_crop_slice(
                    patch_dim, patch_size, halo_width, outputs.shape[patch_dim], ndim=5,
                    global_start=global_start,
                    kernel_size=kernel_size_patch_dim,
                    padding=padding_patch_dim,
                    stride=stride_patch_dim,
                    input_halo_width=halo_width,
                )
                return outputs[tuple(crop_slice)].contiguous()
