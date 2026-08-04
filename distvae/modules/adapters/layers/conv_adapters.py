from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.models.autoencoders.autoencoder_kl_wan import WanCausalConv3d
from distvae.models.layers.conv2d import PatchConv2d
from distvae.models.layers.conv3d import PatchConv3d
from distvae.modules.adapters.diffusers_blocks import (
    HUNYUAN_VIDEO,
    HUNYUAN_VIDEO_15,
    LTX2_VIDEO,
    QWEN_IMAGE,
    block,
    require,
    resolved,
)

QwenImageCausalConv3d = block(QWEN_IMAGE, "QwenImageCausalConv3d")
HunyuanVideoCausalConv3d = block(HUNYUAN_VIDEO, "HunyuanVideoCausalConv3d")
HunyuanVideo15CausalConv3d = block(HUNYUAN_VIDEO_15, "HunyuanVideo15CausalConv3d")
LTX2VideoCausalConv3d = block(LTX2_VIDEO, "LTX2VideoCausalConv3d")


class Conv2dAdapter(nn.Module):
    def __init__(
        self, 
        conv2d: nn.Conv2d,
        *,
        block_size = 0,
        patch_dim: int = -2,
        use_uniform_patch: bool = False,
    ):
        super().__init__()
        for i in conv2d.dilation:
            assert i == 1, "dilation is not supported in Conv2dAdapter"
        self.conv2d = PatchConv2d(
            in_channels=conv2d.in_channels,
            out_channels=conv2d.out_channels,
            kernel_size=conv2d.kernel_size,
            stride=conv2d.stride,
            padding=conv2d.padding,
            dilation=conv2d.dilation,
            groups=conv2d.groups,
            bias=conv2d.bias is not None,
            padding_mode=conv2d.padding_mode,
            device=conv2d.weight.device,
            dtype=conv2d.weight.dtype,
            block_size=block_size,
            patch_dim=patch_dim,
            use_uniform_patch=use_uniform_patch,
        )
        self.conv2d.weight.data = conv2d.weight.data
        if conv2d.bias is not None:
            self.conv2d.bias.data = conv2d.bias.data

    def forward(self, x):
        return self.conv2d(x)


class Conv3dAdapter(nn.Module):
    def __init__(
        self, 
        conv3d: nn.Conv3d,
        *,
        block_size = 0,
        patch_dim: int = -2,
        use_uniform_patch: bool = False,
    ):
        super().__init__()
        for i in conv3d.dilation:
            assert i == 1, "dilation is not supported in Conv3dAdapter"
        self.conv3d = PatchConv3d(
            in_channels=conv3d.in_channels,
            out_channels=conv3d.out_channels,
            kernel_size=conv3d.kernel_size,
            stride=conv3d.stride,
            padding=conv3d.padding,
            dilation=conv3d.dilation,
            groups=conv3d.groups,
            bias=conv3d.bias is not None,
            padding_mode=conv3d.padding_mode,
            device=conv3d.weight.device,
            dtype=conv3d.weight.dtype,
            block_size=block_size,
            patch_dim=patch_dim,
            use_uniform_patch=use_uniform_patch,
        )
        self.conv3d.weight.data = conv3d.weight.data
        if conv3d.bias is not None:
            self.conv3d.bias.data = conv3d.bias.data

    def forward(self, x):
        return self.conv3d(x)


class _CausalConv3dAdapter(nn.Module):
    """Shards a causal 3D convolution that subclasses nn.Conv3d and holds its padding in _padding.

    Only the spatial half of that padding reaches PatchConv3d, which exchanges halos so a rank
    pads where the image ends rather than where its own patch happens to. The temporal half is
    applied here instead, before the convolution, because the frame axis is not the one split
    across ranks and its causal padding has to stay one-sided.
    """

    _supported: Tuple[type, ...] = ()
    _requires: str = ""

    def __init__(
        self,
        causal_conv3d: nn.Conv3d,
        *,
        block_size = 0,
        patch_dim: int = -2,
        use_uniform_patch: bool = False,
    ):
        super().__init__()
        adapter = type(self).__name__
        require(self._supported, adapter, self._requires)
        for i in causal_conv3d.dilation:
            assert i == 1, f"dilation is not supported in {adapter}"
        assert isinstance(causal_conv3d, self._supported), (
            f"{adapter} does not support causal_conv3d except {self._requires}"
        )
        self.conv3d = PatchConv3d(
            in_channels=causal_conv3d.in_channels,
            out_channels=causal_conv3d.out_channels,
            kernel_size=causal_conv3d.kernel_size,
            stride=causal_conv3d.stride,
            padding=(0, causal_conv3d._padding[2], causal_conv3d._padding[0]),
            dilation=causal_conv3d.dilation,
            groups=causal_conv3d.groups,
            bias=causal_conv3d.bias is not None,
            padding_mode=causal_conv3d.padding_mode,
            device=causal_conv3d.weight.device,
            dtype=causal_conv3d.weight.dtype,
            block_size=block_size,
            patch_dim=patch_dim,
            use_uniform_patch=use_uniform_patch,
        )
        self.conv3d.weight.data = causal_conv3d.weight.data
        if causal_conv3d.bias is not None:
            self.conv3d.bias.data = causal_conv3d.bias.data
        self._padding = (0, 0, 0, 0, causal_conv3d._padding[4], causal_conv3d._padding[5])

    def forward(self, x, cache_x=None):
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
        x = F.pad(x, padding)
        return self.conv3d(x)


class WanCausalConv3dAdapter(_CausalConv3dAdapter):
    _supported = resolved(WanCausalConv3d)
    _requires = "WanCausalConv3d"


class QwenImageCausalConv3dAdapter(_CausalConv3dAdapter):
    """Qwen-Image's causal convolution, which is WanCausalConv3d under a different name"""

    _supported = resolved(QwenImageCausalConv3d)
    _requires = "QwenImageCausalConv3d"


class _PaddedCausalConv3dAdapter(nn.Module):
    """Shards a causal 3D convolution that holds a plain nn.Conv3d and pads in its own forward.

    Unlike Wan's, these pad by replication rather than with zeros, which left alone would have
    each rank repeat its own top and bottom rows where it ought to be reading its neighbour's.
    Moving the spatial half of the padding into PatchConv3d settles that: it exchanges halos and
    replicates only at the edges of the real image. The temporal half is applied here, because
    the frame axis is not the one being split and its padding has to stay one-sided.
    """

    _supported: Tuple[type, ...] = ()
    _requires: str = ""

    def __init__(
        self,
        causal_conv3d: nn.Module,
        *,
        block_size = 0,
        patch_dim: int = -2,
        use_uniform_patch: bool = False,
    ):
        super().__init__()
        adapter = type(self).__name__
        require(self._supported, adapter, self._requires)
        assert isinstance(causal_conv3d, self._supported), (
            f"{adapter} does not support causal_conv3d except {self._requires}"
        )
        conv = causal_conv3d.conv
        for i in conv.dilation:
            assert i == 1, f"dilation is not supported in {adapter}"
        assert tuple(conv.padding) == (0, 0, 0), (
            f"{adapter} expects all padding to live in time_causal_padding, but the "
            f"convolution also pads by {tuple(conv.padding)}"
        )
        # F.pad orders its argument (W, W, H, H, F, F).
        pad_w, _, pad_h, _, pad_front, pad_back = causal_conv3d.time_causal_padding
        self.conv3d = PatchConv3d(
            in_channels=conv.in_channels,
            out_channels=conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=(0, pad_h, pad_w),
            dilation=conv.dilation,
            groups=conv.groups,
            bias=conv.bias is not None,
            padding_mode=causal_conv3d.pad_mode,
            device=conv.weight.device,
            dtype=conv.weight.dtype,
            block_size=block_size,
            patch_dim=patch_dim,
            use_uniform_patch=use_uniform_patch,
        )
        self.conv3d.weight.data = conv.weight.data
        if conv.bias is not None:
            self.conv3d.bias.data = conv.bias.data
        self.pad_mode = causal_conv3d.pad_mode
        self._padding = (0, 0, 0, 0, pad_front, pad_back)

    def forward(self, hidden_states):
        # Padding one axis and then the other reaches the same place as padding both at once:
        # replication reads a clamped index per axis, and clamping them in turn is the same.
        hidden_states = F.pad(hidden_states, self._padding, mode=self.pad_mode)
        return self.conv3d(hidden_states)


class HunyuanVideoCausalConv3dAdapter(_PaddedCausalConv3dAdapter):
    _supported = resolved(HunyuanVideoCausalConv3d)
    _requires = "HunyuanVideoCausalConv3d"


class HunyuanVideo15CausalConv3dAdapter(_PaddedCausalConv3dAdapter):
    _supported = resolved(HunyuanVideo15CausalConv3d)
    _requires = "HunyuanVideo15CausalConv3d"


class LTX2VideoCausalConv3dAdapter(nn.Module):
    """Shards LTX-2's causal convolution, which needs less rearranging than the others.

    Its spatial padding already sits inside the nn.Conv3d rather than being applied around it,
    so swapping that convolution for a PatchConv3d built from the same arguments is the whole of
    it. The temporal padding repeats the first and last frames along an axis nobody splits, and
    happens in the wrapped module's own forward, which is left to run as it is.
    """

    _supported = resolved(LTX2VideoCausalConv3d)
    _requires = "LTX2VideoCausalConv3d"

    def __init__(
        self,
        causal_conv3d: nn.Module,
        *,
        block_size = 0,
        patch_dim: int = -2,
        use_uniform_patch: bool = False,
    ):
        super().__init__()
        adapter = type(self).__name__
        require(self._supported, adapter, self._requires)
        assert isinstance(causal_conv3d, self._supported), (
            f"{adapter} does not support causal_conv3d except {self._requires}"
        )
        conv = causal_conv3d.conv
        for i in conv.dilation:
            assert i == 1, f"dilation is not supported in {adapter}"
        self.causal_conv3d = causal_conv3d
        sharded = PatchConv3d(
            in_channels=conv.in_channels,
            out_channels=conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=conv.bias is not None,
            padding_mode=conv.padding_mode,
            device=conv.weight.device,
            dtype=conv.weight.dtype,
            block_size=block_size,
            patch_dim=patch_dim,
            use_uniform_patch=use_uniform_patch,
        )
        sharded.weight.data = conv.weight.data
        if conv.bias is not None:
            sharded.bias.data = conv.bias.data
        causal_conv3d.conv = sharded

    def forward(self, hidden_states, causal: bool = True):
        return self.causal_conv3d(hidden_states, causal=causal)
