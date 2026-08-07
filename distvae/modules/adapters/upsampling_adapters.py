from typing import Optional, Tuple

import torch
import torch.nn as nn

from distvae.utils import DistributedEnv, cache_cursor
from distvae.models.upsampling import PatchUpsample2D
from distvae.modules.adapters.diffusers_blocks import (
    HUNYUAN_VIDEO,
    HUNYUAN_VIDEO_15,
    LTX2_VIDEO,
    QWEN_IMAGE,
    block,
    require,
    resolved,
)
from distvae.modules.adapters.layers.conv_adapters import (
    Conv2dAdapter,
    HunyuanVideo15CausalConv3dAdapter,
    HunyuanVideoCausalConv3dAdapter,
    LTX2VideoCausalConv3dAdapter,
    QwenImageCausalConv3dAdapter,
    WanCausalConv3dAdapter,
)
from distvae.modules.adapters.resnet_adapters import (
    HunyuanVideo15ResnetBlockAdapter,
    HunyuanVideoResnetBlockAdapter,
    LTX2VideoResnetBlockAdapter,
    QwenImageResidualBlockAdapter,
    WanResidualBlockAdapter,
)
from diffusers.models.upsampling import Upsample2D
from diffusers.models.autoencoders.autoencoder_kl_wan import WanResample, WanResidualUpBlock, WanUpBlock

QwenImageResample = block(QWEN_IMAGE, "QwenImageResample")
QwenImageUpBlock = block(QWEN_IMAGE, "QwenImageUpBlock")
HunyuanVideoUpsampleCausal3D = block(HUNYUAN_VIDEO, "HunyuanVideoUpsampleCausal3D")
HunyuanVideoUpBlock3D = block(HUNYUAN_VIDEO, "HunyuanVideoUpBlock3D")
HunyuanVideo15Upsample = block(HUNYUAN_VIDEO_15, "HunyuanVideo15Upsample")
HunyuanVideo15UpBlock3D = block(HUNYUAN_VIDEO_15, "HunyuanVideo15UpBlock3D")
LTX2VideoUpsampler3d = block(LTX2_VIDEO, "LTX2VideoUpsampler3d")
LTX2VideoUpBlock3d = block(LTX2_VIDEO, "LTX2VideoUpBlock3d")


class Upsample2DAdapter(nn.Module):
    def __init__(
        self, 
        upsample2d: Upsample2D,
        *,
        conv_block_size = 0,
    ):
        super().__init__()
        assert upsample2d.norm is None, "upsample2dBlock2DAdapter does not support normalization"
        if upsample2d.name == "conv":
            assert not isinstance(upsample2d.conv, nn.ConvTranspose2d), "upsample2dBlock2DAdapter does not support transpose conv"
        else:
            assert not isinstance(upsample2d.Conv2d_0, nn.ConvTranspose2d), "upsample2dBlock2DAdapter does not support transpose conv"
        self.upsample2d = PatchUpsample2D(
            channels=upsample2d.channels,
            use_conv=upsample2d.use_conv,
            use_conv_transpose=upsample2d.use_conv_transpose,
            out_channels=upsample2d.out_channels,
            name=upsample2d.name,
            kernel_size=None,
            padding=1,
            interpolate=upsample2d.interpolate
        )
        if upsample2d.name == "conv":
            self.upsample2d.conv = Conv2dAdapter(upsample2d.conv, block_size=conv_block_size)
        else:
            self.upsample2d.Conv2d_0 = Conv2dAdapter(upsample2d.Conv2d_0, block_size=conv_block_size)
        

    def forward(
        self, hidden_states: torch.FloatTensor, output_size: Optional[int] = None, *args, **kwargs
    ):
        return self.upsample2d(hidden_states, output_size, *args, **kwargs)


class _CausalResampleAdapter(nn.Module):
    """Shards a resample block: the 2D convolution it upsamples with, and its temporal one.

    The interpolation between them is nearest-neighbour, which reads a single input pixel per
    output pixel, so a rank can upsample its own rows without hearing from its neighbours.
    """

    _supported: Tuple[type, ...] = ()
    _requires: str = ""
    _conv_adapter = None

    def __init__(
        self,
        resample: nn.Module,
        conv_block_size = 0,
        patch_dim: int = -2,
    ):
        super().__init__()
        adapter = type(self).__name__
        require(self._supported, adapter, self._requires)
        assert isinstance(resample, self._supported), (
            f"{adapter} does not support resample except {self._requires}"
        )
        if patch_dim == -3:
            raise ValueError(
                f"{adapter} does not support patch_dim F (-3); use H (-2) or W (-1)."
            )
        self.resample = resample
        if hasattr(resample, "time_conv"):
            resample.time_conv = self._conv_adapter(
                resample.time_conv,
                block_size=conv_block_size,
                patch_dim=patch_dim,
            )
        if isinstance(resample.resample, nn.Sequential):
            self.resample.resample = nn.Sequential(*[
                Conv2dAdapter(
                    layer,
                    block_size=conv_block_size,
                    patch_dim=patch_dim,
                ) if isinstance(layer, nn.Conv2d) else layer
                for layer in resample.resample
            ])

    def forward(self, x, feat_cache=None, feat_idx=None):
        return self.resample(x, feat_cache=feat_cache, feat_idx=cache_cursor(feat_idx))


class WanResampleAdapter(_CausalResampleAdapter):
    _supported = resolved(WanResample)
    _requires = "WanResample"
    _conv_adapter = WanCausalConv3dAdapter


class QwenImageResampleAdapter(_CausalResampleAdapter):
    _supported = resolved(QwenImageResample)
    _requires = "QwenImageResample"
    _conv_adapter = QwenImageCausalConv3dAdapter


class _CausalUpBlockAdapter(nn.Module):
    """Shards an up block: its residual blocks, and whichever resample it upsamples with"""

    _supported: Tuple[type, ...] = ()
    _requires: str = ""
    _resnet_adapter = None
    _resample_adapter = None
    _resample_types: Tuple[type, ...] = ()
    # Which attribute the wrapped block is kept under, since a decoder reaches back through it.
    _attr = "up_block"
    # Wan threads first_chunk through its up blocks to tell the temporal cache it is starting
    # over. The families forked from it dropped that argument.
    _takes_first_chunk = True

    def __init__(
        self,
        up_block: nn.Module,
        conv_block_size = 0,
        patch_dim: int = -2,
    ):
        super().__init__()
        adapter = type(self).__name__
        require(self._supported, adapter, self._requires)
        assert isinstance(up_block, self._supported), (
            f"{adapter} does not support up block except {self._requires}"
        )
        options = dict(
            conv_block_size=conv_block_size,
            patch_dim=patch_dim,
        )
        up_block.resnets = nn.ModuleList(
            [self._resnet_adapter(resnet, **options) for resnet in up_block.resnets]
        )
        if hasattr(up_block, "upsamplers"):
            if up_block.upsamplers is not None:
                up_block.upsamplers = nn.ModuleList([
                    self._resample_adapter(upsampler, **options)
                    if isinstance(upsampler, self._resample_types) else upsampler
                    for upsampler in up_block.upsamplers
                ])
        elif hasattr(up_block, "upsampler"):
            if isinstance(up_block.upsampler, self._resample_types):
                up_block.upsampler = self._resample_adapter(up_block.upsampler, **options)
        setattr(self, self._attr, up_block)

    def forward(self, x, feat_cache=None, feat_idx=None, first_chunk=False):
        up_block = getattr(self, self._attr)
        feat_idx = cache_cursor(feat_idx)
        if self._takes_first_chunk:
            return up_block(
                x, feat_cache=feat_cache, feat_idx=feat_idx, first_chunk=first_chunk
            )
        return up_block(x, feat_cache=feat_cache, feat_idx=feat_idx)


class WanResidualUpBlockAdapter(_CausalUpBlockAdapter):
    _supported = resolved(WanResidualUpBlock)
    _requires = "WanResidualUpBlock"
    _resnet_adapter = WanResidualBlockAdapter
    _resample_adapter = WanResampleAdapter
    _resample_types = resolved(WanResample)
    _attr = "residual_up_block"


class WanUpBlockAdapter(_CausalUpBlockAdapter):
    _supported = resolved(WanUpBlock)
    _requires = "WanUpBlock"
    _resnet_adapter = WanResidualBlockAdapter
    _resample_adapter = WanResampleAdapter
    _resample_types = resolved(WanResample)


class QwenImageUpBlockAdapter(_CausalUpBlockAdapter):
    _supported = resolved(QwenImageUpBlock)
    _requires = "QwenImageUpBlock"
    _resnet_adapter = QwenImageResidualBlockAdapter
    _resample_adapter = QwenImageResampleAdapter
    _resample_types = resolved(QwenImageResample)
    _takes_first_chunk = False


class _PaddedCausalUpsampleAdapter(nn.Module):
    """Shards a HunyuanVideo upsampler, which is only its convolution

    What surrounds that convolution is nearest-neighbour interpolation in one family and a
    channel-to-space shuffle in the other. Both read a single input position per output one, so
    a rank can upsample its own rows knowing nothing about anyone else's.
    """

    _supported: Tuple[type, ...] = ()
    _requires: str = ""
    _conv_adapter = None

    def __init__(
        self,
        upsampler: nn.Module,
        conv_block_size = 0,
        patch_dim: int = -2,
    ):
        super().__init__()
        adapter = type(self).__name__
        require(self._supported, adapter, self._requires)
        assert isinstance(upsampler, self._supported), (
            f"{adapter} does not support upsampler except {self._requires}"
        )
        self.upsampler = upsampler
        upsampler.conv = self._conv_adapter(
            upsampler.conv,
            block_size=conv_block_size,
            patch_dim=patch_dim,
        )

    def forward(self, hidden_states):
        return self.upsampler(hidden_states)


class HunyuanVideoUpsampleAdapter(_PaddedCausalUpsampleAdapter):
    _supported = resolved(HunyuanVideoUpsampleCausal3D)
    _requires = "HunyuanVideoUpsampleCausal3D"
    _conv_adapter = HunyuanVideoCausalConv3dAdapter


class HunyuanVideo15UpsampleAdapter(_PaddedCausalUpsampleAdapter):
    _supported = resolved(HunyuanVideo15Upsample)
    _requires = "HunyuanVideo15Upsample"
    _conv_adapter = HunyuanVideo15CausalConv3dAdapter


class _PaddedCausalUpBlockAdapter(nn.Module):
    """Shards a HunyuanVideo up block: its residual blocks and its upsampler"""

    _supported: Tuple[type, ...] = ()
    _requires: str = ""
    _resnet_adapter = None
    _upsample_adapter = None

    def __init__(
        self,
        up_block: nn.Module,
        conv_block_size = 0,
        patch_dim: int = -2,
    ):
        super().__init__()
        adapter = type(self).__name__
        require(self._supported, adapter, self._requires)
        assert isinstance(up_block, self._supported), (
            f"{adapter} does not support up block except {self._requires}"
        )
        options = dict(
            conv_block_size=conv_block_size,
            patch_dim=patch_dim,
        )
        self.up_block = up_block
        up_block.resnets = nn.ModuleList(
            [self._resnet_adapter(resnet, **options) for resnet in up_block.resnets]
        )
        if up_block.upsamplers is not None:
            up_block.upsamplers = nn.ModuleList(
                [self._upsample_adapter(up, **options) for up in up_block.upsamplers]
            )

    def forward(self, hidden_states):
        return self.up_block(hidden_states)


class HunyuanVideoUpBlockAdapter(_PaddedCausalUpBlockAdapter):
    _supported = resolved(HunyuanVideoUpBlock3D)
    _requires = "HunyuanVideoUpBlock3D"
    _resnet_adapter = HunyuanVideoResnetBlockAdapter
    _upsample_adapter = HunyuanVideoUpsampleAdapter


class HunyuanVideo15UpBlockAdapter(_PaddedCausalUpBlockAdapter):
    _supported = resolved(HunyuanVideo15UpBlock3D)
    _requires = "HunyuanVideo15UpBlock3D"
    _resnet_adapter = HunyuanVideo15ResnetBlockAdapter
    _upsample_adapter = HunyuanVideo15UpsampleAdapter


class LTX2VideoUpsamplerAdapter(nn.Module):
    """Shards an LTX-2 upsampler, which is its convolution

    What follows the convolution moves channels into space, reading one input position per
    output one, so a rank can do it to its own rows alone.
    """

    _supported = resolved(LTX2VideoUpsampler3d)
    _requires = "LTX2VideoUpsampler3d"

    def __init__(
        self,
        upsampler: nn.Module,
        conv_block_size = 0,
        patch_dim: int = -2,
    ):
        super().__init__()
        adapter = type(self).__name__
        require(self._supported, adapter, self._requires)
        assert isinstance(upsampler, self._supported), (
            f"{adapter} does not support upsampler except {self._requires}"
        )
        self.upsampler = upsampler
        upsampler.conv = LTX2VideoCausalConv3dAdapter(
            upsampler.conv,
            block_size=conv_block_size,
            patch_dim=patch_dim,
        )

    def forward(self, hidden_states, causal: bool = True):
        return self.upsampler(hidden_states, causal=causal)


class LTX2VideoUpBlockAdapter(nn.Module):
    """Shards an LTX-2 up block: an optional leading residual block, an upsampler, and resnets

    Unlike the other families the upsampler comes before the residual blocks rather than after,
    which changes nothing about what has to be sharded, only the order it runs in.
    """

    _supported = resolved(LTX2VideoUpBlock3d)
    _requires = "LTX2VideoUpBlock3d"

    def __init__(
        self,
        up_block: nn.Module,
        conv_block_size = 0,
        patch_dim: int = -2,
    ):
        super().__init__()
        adapter = type(self).__name__
        require(self._supported, adapter, self._requires)
        assert isinstance(up_block, self._supported), (
            f"{adapter} does not support up block except {self._requires}"
        )
        options = dict(
            conv_block_size=conv_block_size,
            patch_dim=patch_dim,
        )
        self.up_block = up_block
        if up_block.conv_in is not None:
            up_block.conv_in = LTX2VideoResnetBlockAdapter(up_block.conv_in, **options)
        if up_block.upsamplers is not None:
            up_block.upsamplers = nn.ModuleList(
                [LTX2VideoUpsamplerAdapter(up, **options) for up in up_block.upsamplers]
            )
        up_block.resnets = nn.ModuleList(
            [LTX2VideoResnetBlockAdapter(resnet, **options) for resnet in up_block.resnets]
        )

    def forward(self, hidden_states, temb=None, generator=None, causal: bool = True):
        return self.up_block(hidden_states, temb, generator, causal=causal)
