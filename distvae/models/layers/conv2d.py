from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from torch.nn.modules.utils import _pair
from torch.nn.common_types import _size_2_t

from distvae.models.layers.conv_utils import (
    get_world_size_and_rank,
    chunk_bounds,
    build_crop_slice,
)
from distvae.models.layers.conv_mixin import PatchConvMixin
from distvae.utils import ParallelContext, normalize_patch_dim


class PatchConv2d(nn.Conv2d, PatchConvMixin):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',  # TODO: refine this type
        device=None,
        dtype=None,
        block_size: Union[int, Tuple[int, int]] = 0,
        patch_dim: int = -2,
        parallel_context: ParallelContext = None,
    ) -> None:

        if isinstance(dilation, int):
            assert dilation == 1, "dilation is not supported in PatchConv2d"
        else:
            for i in dilation:
                assert i == 1, "dilation is not supported in PatchConv2d"
        patch_dim = normalize_patch_dim(patch_dim, 4, spatial_only=True)
        self.block_size = block_size
        self.parallel_context = parallel_context
        self.patch_dim = parallel_context.patch_dim if parallel_context is not None else patch_dim
        self.halo_buffer = {}
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, padding_mode, device, dtype)

    def _patch_ndim(self) -> int:
        return 4

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        bs, channels, h, w = input.shape

        group_world_size, global_rank, rank_in_group, local_rank = get_world_size_and_rank(
            self.parallel_context
        )

        if (group_world_size == 1):
            if self.padding_mode != 'zeros':
                return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                                weight, bias, self.stride,
                                _pair(0), self.dilation, self.groups)
            return F.conv2d(input, weight, bias, self.stride,
                            self.padding, self.dilation, self.groups)

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
            bs, channels, h, w = input.shape
            if self._use_direct_path(input):
                if self.padding_mode != 'zeros':
                    conv_res = F.conv2d(F.pad(input, padding, mode=self.padding_mode),
                                    weight, bias, self.stride,
                                    _pair(0), self.dilation, self.groups)
                else:
                    if (
                        stride_patch_dim == 1 and
                        padding_patch_dim == 1 and
                        kernel_size_patch_dim == 3
                    ):
                        conv_res = F.conv2d(input, weight, bias, self.stride,
                                    self.padding, self.dilation, self.groups)
                    else:
                        conv_res = F.conv2d(F.pad(input, padding, "constant", 0.0),
                                        weight, bias, self.stride,
                                        _pair(0), self.dilation, self.groups)

                # Always apply cropping when halos are present to remove halo regions from output
                # This prevents rank boundary artifacts for all convolution configurations.
                # build_crop_slice also recognises the output that is already patch-sized, which
                # is what the branches above that pad only the outer edges produce: there the
                # halo stands in for the padding those branches dropped, so nothing is left over
                # to crop and cropping anyway would eat into the patch itself.
                if halo_width[0] > 0 or halo_width[1] > 0:
                    crop_slice = build_crop_slice(
                        patch_dim, patch_size, halo_width, conv_res.shape[patch_dim], ndim=4,
                        global_start=global_start,
                        kernel_size=kernel_size_patch_dim,
                        padding=padding_patch_dim,
                        stride=stride_patch_dim,
                        input_halo_width=halo_width,
                    )
                    conv_res = conv_res[tuple(crop_slice)].contiguous()

                return conv_res
            else:
                if self.padding_mode != "zeros":
                    input = F.pad(input, padding, mode=self.padding_mode)
                elif self.padding != 0:
                    input = F.pad(input, padding, mode="constant")

                _, _, h, w = input.shape
                # nn.Conv2d normalises all three of these to pairs in its own __init__, so they
                # are read as pairs rather than tested for which they are.
                block_h, block_w = _pair(self.block_size)
                kernel_h, kernel_w = _pair(self.kernel_size)
                stride_h, stride_w = _pair(self.stride)
                rows = chunk_bounds(h, block_h, kernel_h, stride_h)
                columns = chunk_bounds(w, block_w, kernel_w, stride_w)

                outputs = torch.cat([
                    torch.cat([
                        F.conv2d(
                            input[:, :, top:bottom, left:right],
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
                # patch_size here is this rank's own, read before the halo was exchanged.
                crop_slice = build_crop_slice(
                    patch_dim, patch_size, halo_width, outputs.shape[patch_dim], ndim=4,
                    global_start=global_start,
                    kernel_size=kernel_size_patch_dim,
                    padding=padding_patch_dim,
                    stride=stride_patch_dim,
                    input_halo_width=halo_width,
                )
                return outputs[tuple(crop_slice)].contiguous()
