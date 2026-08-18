from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from distvae.models.layers.conv_mixin import PatchConvMixin
from distvae.models.layers.conv_utils import (
    chunk_bounds,
    get_world_size_and_rank,
)
from distvae.utils import ParallelContext, normalize_patch_dim


Size2 = Union[int, Tuple[int, int]]
Size4 = Union[int, Tuple[int, int, int, int]]


class AsymmetricZeroPadConv2d(nn.Conv2d, PatchConvMixin):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Size2 = 3,
        stride: Size2 = 2,
        dilation: Size2 = 1,
        groups: int = 1,
        bias: bool = True,
        device=None,
        dtype=None,
        reversed_zero_padding: Size4 = 0,
        block_size: Union[int, Tuple[int, int]] = 0,
        parallel_context: ParallelContext = None,
    ) -> None:
        if isinstance(dilation, int):
            assert dilation == 1, "dilation is not supported in AsymmetricZeroPadConv2d"
        else:
            for value in dilation:
                assert value == 1, (
                    "dilation is not supported in AsymmetricZeroPadConv2d"
                )
        if not isinstance(parallel_context, ParallelContext):
            raise TypeError("AsymmetricZeroPadConv2d requires a ParallelContext")
        if isinstance(reversed_zero_padding, int):
            reversed_zero_padding = (
                reversed_zero_padding,
                reversed_zero_padding,
                reversed_zero_padding,
                reversed_zero_padding,
            )
        elif isinstance(reversed_zero_padding, tuple):
            assert len(reversed_zero_padding) == 4, (
                "reversed_zero_padding must be a tuple of 4 integers"
            )
        else:
            raise ValueError(
                f"Unsupported reversed_zero_padding: {type(reversed_zero_padding)}"
            )
        if (
            reversed_zero_padding[0] != 0
            or reversed_zero_padding[1] != 1
            or reversed_zero_padding[2] != 0
            or reversed_zero_padding[3] != 1
        ):
            raise ValueError(
                f"Unsupported reversed_zero_padding: {reversed_zero_padding}"
            )
        if (
            isinstance(kernel_size, int)
            and kernel_size != 3
            or isinstance(kernel_size, tuple)
            and (kernel_size[0] != 3 or kernel_size[1] != 3)
        ):
            raise ValueError(f"Unsupported kernel_size: {kernel_size}")
        if (
            isinstance(stride, int)
            and stride != 2
            or isinstance(stride, tuple)
            and (stride[0] != 2 or stride[1] != 2)
        ):
            raise ValueError(f"Unsupported stride: {stride}")

        self.reversed_zero_padding = reversed_zero_padding
        self.block_size = block_size
        self.parallel_context = parallel_context
        self.patch_dim = normalize_patch_dim(
            parallel_context.patch_dim, 4, spatial_only=True
        )
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
            dtype,
        )

    def _patch_ndim(self) -> int:
        """Return 4 for 2D (N, C, H, W)."""
        return 4

    def _conv_forward(
        self, input: Tensor, weight: Tensor, bias: Optional[Tensor]
    ) -> Tensor:
        group_world_size, rank_in_group = get_world_size_and_rank(
            self.parallel_context
        )

        reversed_zero_padding = tuple(self.reversed_zero_padding)
        patch_dim = input.ndim + normalize_patch_dim(self.patch_dim, input.ndim)
        # The pad-then-stride-2 arithmetic below assumes each band halves cleanly. Bands are cut
        # in multiples of what the whole encoder narrows by, so they are still even here.
        assert input.shape[patch_dim] % 2 == 0, (
            "input.shape[patch_dim] must be even"
        )

        if group_world_size == 1:
            return F.conv2d(
                F.pad(
                    input,
                    reversed_zero_padding,
                    mode="constant",
                    value=0,
                ),
                weight,
                bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )

        (
            input,
            patch_dim,
            _patch_size,
            _halo_width,
            _kernel_size_patch_dim,
            _padding_patch_dim,
            _stride_patch_dim,
            _global_start,
            group_world_size,
            rank_in_group,
        ) = self._multi_rank_metadata_and_halo(input, self.halo_buffer)

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

        _, _, height, width = input.shape
        if self._use_direct_path(input):
            return F.conv2d(
                input,
                weight,
                bias,
                self.stride,
                (0, 0),
                self.dilation,
                self.groups,
            )

        block_h, block_w = (
            (self.block_size, self.block_size)
            if isinstance(self.block_size, int)
            else self.block_size
        )
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride
        rows = chunk_bounds(height, block_h, kernel_h, stride_h)
        columns = chunk_bounds(width, block_w, kernel_w, stride_w)

        return torch.cat(
            [
                torch.cat(
                    [
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
                    ],
                    dim=-1,
                )
                for top, bottom in rows
            ],
            dim=-2,
        )
