from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from torch.nn.modules.utils import _pair
from torch.nn.common_types import _size_2_t,_size_4_t

from distvae.models.layers.conv_utils import (
    get_world_size_and_rank,
    correct_end,
    correct_start,
)
from distvae.models.layers.conv_mixin import PatchConvMixin


class WanZeroPadConv2d(nn.Conv2d, PatchConvMixin):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t = 3,
        stride: _size_2_t = 2,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        device=None,
        dtype=None,
        reversed_zero_padding: Union[int, _size_4_t] = 0,
        block_size: Union[int, Tuple[int, int, int]] = 0,
        patch_dim: int = -2,
    ) -> None:
        if isinstance(dilation, int):
            assert dilation == 1, "dilation is not supported in WanZeroPadConv2d"
        else:
            for i in dilation:
                assert i == 1, "dilation is not supported in WanZeroPadConv2d"
        assert patch_dim in (-2, -1), (
            "WanZeroPadConv2d patch_dim must be H (-2) or W (-1)"
        )
        if isinstance(reversed_zero_padding, int):
            reversed_zero_padding = (
                reversed_zero_padding, reversed_zero_padding, reversed_zero_padding, reversed_zero_padding
            )
        elif isinstance(reversed_zero_padding, tuple):
            assert len(reversed_zero_padding) == 4, "reversed_zero_padding must be a tuple of 4 integers"
        else:
            raise ValueError(f"Unsupported reversed_zero_padding: {type(reversed_zero_padding)}")
        if (
            reversed_zero_padding[0] != 0 or
            reversed_zero_padding[1] != 1 or
            reversed_zero_padding[2] != 0 or
            reversed_zero_padding[3] != 1
        ):
            raise ValueError(f"Unsupported reversed_zero_padding: {reversed_zero_padding}")
        # Validate kernel_size and stride
        if (
            isinstance(kernel_size, int) and kernel_size != 3 or
            isinstance(kernel_size, tuple) and (kernel_size[0] != 3 or kernel_size[1] != 3)
        ):
            raise ValueError(f"Unsupported kernel_size: {kernel_size}")
        if (
            isinstance(stride, int) and stride != 2 or
            isinstance(stride, tuple) and (stride[0] != 2 or stride[1] != 2)
        ):
            raise ValueError(f"Unsupported stride: {stride}")

        self.reversed_zero_padding = reversed_zero_padding
        self.block_size = block_size
        self.patch_dim = patch_dim
        self.halo_buffer = {}
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            0,
            dilation,
            groups,
            bias,
            "zeros",
            device,
            dtype
        )

    def _patch_ndim(self) -> int:
        """Return 4 for 2D (N, C, H, W)."""
        return 4

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        group_world_size, global_rank, rank_in_group, local_rank = get_world_size_and_rank()

        bs, channels, h, w = input.shape
        reversed_zero_padding = tuple(self.reversed_zero_padding)

        patch_dim = self.patch_dim if self.patch_dim >= 0 else input.ndim + self.patch_dim
        # The pad-then-stride-2 arithmetic below assumes each band halves cleanly. Bands are cut
        # in multiples of what the whole encoder narrows by, so they are still even here.
        assert input.shape[patch_dim] % 2 == 0, "input.shape[patch_dim] must be even"

        # Single rank: use standard F.conv2d
        if group_world_size == 1:
            output = F.conv2d(
                F.pad(
                    input,
                    reversed_zero_padding,
                    mode="constant",
                    value=0
                ),
                weight,
                bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups
            )

            return output
        # Multi-rank: get extended input and metadata from mixin (patch_index, halo_width, etc.), then choose direct or chunked path.
        else:
            # Metadata and halo exchange
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
                _,
            ) = self._multi_rank_metadata_and_halo(input, self.halo_buffer)

            # ZeroPad2d
            if rank_in_group == 0:
                padding = list(reversed_zero_padding)
                padding[2 * (2 - patch_dim + 1) + 1] = 0
            elif rank_in_group == group_world_size - 1:
                padding = list(reversed_zero_padding)
                padding[2 * (2 - patch_dim + 1)] = 0
            else:
                padding = list(reversed_zero_padding)
                padding[2 * (2 - patch_dim + 1)] = 0
                padding[2 * (2 - patch_dim + 1) + 1] = 0
            input = F.pad(input, tuple(padding), mode="constant", value=0)

            # Conv2d
            output: Tensor
            _, channels, h, w = input.shape
            # Direct path: one conv over the extended (halo-padded) input
            if self._use_direct_path(input):
                output = F.conv2d(
                    input,
                    weight,
                    bias,
                    self.stride,
                    _pair(0),
                    self.dilation,
                    self.groups
                )

                return output
            # Chunked path: pad input, split into overlapping chunks along F, H, W; conv each chunk with padding=0; concat outputs
            else:
                _, channels, h, w = input.shape
                if isinstance(self.block_size, int):
                    num_chunks_in_h = (h + self.block_size - 1) // self.block_size
                    num_chunks_in_w = (w + self.block_size - 1) // self.block_size
                else:
                    num_chunks_in_h = (h + self.block_size[0] - 1) // self.block_size[0]
                    num_chunks_in_w = (w + self.block_size[1] - 1) // self.block_size[1]
                unit_chunk_size_h = h // num_chunks_in_h
                unit_chunk_size_w = w // num_chunks_in_w
                if isinstance(self.kernel_size, int):
                    kernel_size_h, kernel_size_w = self.kernel_size, self.kernel_size
                else:
                    kernel_size_h, kernel_size_w = self.kernel_size
                if isinstance(self.stride, int):
                    stride_h, stride_w = self.stride, self.stride
                else:
                    stride_h, stride_w = self.stride

                # Chunk boundaries aligned via correct_end/correct_start so conv outputs line up when concatenated.
                output = []
                for idx_h in range(num_chunks_in_h):
                    inner_output = []
                    for idx_w in range(num_chunks_in_w):
                        start_w = idx_w * unit_chunk_size_w
                        start_h = idx_h * unit_chunk_size_h
                        end_w = (idx_w + 1) * unit_chunk_size_w
                        end_h = (idx_h + 1) * unit_chunk_size_h
                        if idx_w + 1 < num_chunks_in_w:
                            end_w = correct_end(end_w, kernel_size_w, stride_w)
                        else:
                            end_w = w
                        if idx_h + 1 < num_chunks_in_h:
                            end_h = correct_end(end_h, kernel_size_h, stride_h)
                        else:
                            end_h = h
                        if idx_w > 0:
                            start_w = correct_start(start_w, stride_w)
                        if idx_h > 0:
                            start_h = correct_start(start_h, stride_h)

                        inner_output.append(
                            F.conv2d(
                                input[:, :, start_h:end_h, start_w:end_w],
                                weight,
                                bias,
                                self.stride,
                                0,
                                self.dilation,
                                self.groups,
                            )
                        )
                    output.append(torch.cat(inner_output, dim=-1))
                output = torch.cat(output, dim=2)

                return output
