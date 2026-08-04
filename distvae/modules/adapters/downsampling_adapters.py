from typing import Tuple

import torch.nn as nn

from distvae.models.layers.wan.zeropadconv2d import WanZeroPadConv2d
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
    WanResidualBlockAdapter,
)
from diffusers.models.autoencoders.autoencoder_kl_wan import WanResample, WanResidualDownBlock
from diffusers.models.downsampling import Downsample2D

QwenImageResample = block(QWEN_IMAGE, "QwenImageResample")
HunyuanVideoDownsampleCausal3D = block(HUNYUAN_VIDEO, "HunyuanVideoDownsampleCausal3D")
HunyuanVideoDownBlock3D = block(HUNYUAN_VIDEO, "HunyuanVideoDownBlock3D")
HunyuanVideo15Downsample = block(HUNYUAN_VIDEO_15, "HunyuanVideo15Downsample")
HunyuanVideo15DownBlock3D = block(HUNYUAN_VIDEO_15, "HunyuanVideo15DownBlock3D")
LTX2VideoCausalConv3d = block(LTX2_VIDEO, "LTX2VideoCausalConv3d")
LTX2VideoDownsampler3d = block(LTX2_VIDEO, "LTX2VideoDownsampler3d")
LTX2VideoDownBlock3D = block(LTX2_VIDEO, "LTX2VideoDownBlock3D")


def _zero_pad_strided_conv(conv, conv_block_size, patch_dim, use_uniform_patch):
    """A sharded stand-in for a (0, 1, 0, 1) zero pad followed by a stride-2 convolution

    The pair cannot be split as written, because a rank's bottom row is padding only if it is the
    bottom row of the whole image. One module that pads the outside edges and exchanges halos on
    the inside ones settles it. Named for Wan, whose resample was the first to need it, but the
    shape is just as much the one diffusers' own Downsample2D takes when told to pad by hand.
    """
    padding = conv.padding
    if (isinstance(padding, int) and padding != 0) or (
        isinstance(padding, tuple) and sum(padding) != 0
    ):
        raise ValueError(f"Unsupported padding: {padding}")
    sharded = WanZeroPadConv2d(
        in_channels=conv.in_channels,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=conv.bias is not None,
        device=conv.weight.device,
        dtype=conv.weight.dtype,
        reversed_zero_padding=(0, 1, 0, 1),
        block_size=conv_block_size,
        patch_dim=patch_dim,
        use_uniform_patch=use_uniform_patch,
    )
    sharded.weight.data = conv.weight.data
    if conv.bias is not None:
        sharded.bias.data = conv.bias.data
    return sharded


class Downsample2DAdapter(nn.Module):
    """Shards the 2D downsampler AutoencoderKL and Flux.2 use, of which the convolution is the
    only part that reaches across the split

    Told to pad by hand, as the encoders here tell it, it pads (0, 1, 0, 1) and then strides over
    the result with no padding of its own, which is the pair replaced above. Because the pad is
    written into the downsampler's own forward rather than the convolution, that case runs the
    pieces here instead of delegating, so the pad is not applied twice. Told to pad inside the
    convolution it is an ordinary strided one. Told not to convolve at all it averages each 2x2,
    which reads one input position per output one so long as a rank holds whole pairs of rows, and
    the bands Patchify cuts do. Its norm, where it has one, reduces over channels, so it is left
    alone either way.
    """

    def __init__(
        self,
        downsampler: Downsample2D,
        conv_block_size = 0,
        patch_dim: int = -2,
        use_uniform_patch: bool = False,
    ):
        super().__init__()
        assert isinstance(downsampler, Downsample2D), (
            "Downsample2DAdapter does not support downsampler except Downsample2D"
        )
        self.downsampler = downsampler
        self.pads_by_hand = downsampler.use_conv and downsampler.padding == 0
        if not downsampler.use_conv:
            return
        conv = downsampler.conv
        if self.pads_by_hand:
            sharded = _zero_pad_strided_conv(
                conv, conv_block_size, patch_dim, use_uniform_patch
            )
        else:
            sharded = Conv2dAdapter(
                conv,
                block_size=conv_block_size,
                patch_dim=patch_dim,
                use_uniform_patch=use_uniform_patch,
            )
        downsampler.conv = sharded
        # Some configurations name the same convolution twice. Both have to move, or the original
        # stays alive holding a second copy of the weights.
        if getattr(downsampler, "Conv2d_0", None) is conv:
            downsampler.Conv2d_0 = sharded

    def forward(self, hidden_states, *args, **kwargs):
        if not self.pads_by_hand:
            return self.downsampler(hidden_states, *args, **kwargs)
        if self.downsampler.norm is not None:
            hidden_states = self.downsampler.norm(
                hidden_states.permute(0, 2, 3, 1)
            ).permute(0, 3, 1, 2)
        return self.downsampler.conv(hidden_states)


class _CausalResampleDownAdapter(nn.Module):
    """Shards a resample used to downsample: a temporal convolution and a strided spatial one

    The spatial half is a zero pad of (0, 1, 0, 1) followed by a stride-2 convolution with no
    padding of its own. Splitting that needs the pad and the convolution taken together, since a
    rank's bottom row is padding only if it is the bottom row of the whole image, so the pair is
    replaced by one module that pads the outside edges and exchanges halos on the inside ones.
    """

    _supported: Tuple[type, ...] = ()
    _requires: str = ""
    _conv_adapter = None

    def __init__(
        self,
        resample: nn.Module,
        conv_block_size = 0,
        patch_dim: int = -2,
        use_uniform_patch: bool = True,
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

        if getattr(resample, "time_conv", None) is not None:
            resample.time_conv = self._conv_adapter(
                resample.time_conv,
                block_size=conv_block_size,
                patch_dim=patch_dim,
                use_uniform_patch=use_uniform_patch,
            )

        if isinstance(resample.resample, nn.Sequential):
            layers = list(resample.resample)
            convs = [layer for layer in layers if isinstance(layer, nn.Conv2d)]
            pads = [layer for layer in layers if isinstance(layer, nn.ZeroPad2d)]
            if len(layers) != 2 or len(convs) != 1 or len(pads) != 1:
                raise ValueError(
                    f"{adapter} expects a zero pad and one convolution, got "
                    f"{[type(layer).__name__ for layer in layers]}"
                )
            resample.resample = _zero_pad_strided_conv(
                convs[0], conv_block_size, patch_dim, use_uniform_patch
            )
        elif isinstance(resample.resample, nn.Conv2d):
            resample.resample = Conv2dAdapter(
                resample.resample,
                block_size=conv_block_size,
                patch_dim=patch_dim,
                use_uniform_patch=use_uniform_patch,
            )

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        return self.resample(x, feat_cache=feat_cache, feat_idx=feat_idx)


class WanResampleDownAdapter(_CausalResampleDownAdapter):
    _supported = resolved(WanResample)
    _requires = "WanResample"
    _conv_adapter = WanCausalConv3dAdapter


class QwenImageResampleDownAdapter(_CausalResampleDownAdapter):
    _supported = resolved(QwenImageResample)
    _requires = "QwenImageResample"
    _conv_adapter = QwenImageCausalConv3dAdapter


class _PaddedCausalDownsampleAdapter(nn.Module):
    """Shards a HunyuanVideo downsampler, of which the convolution is the only sharded part

    Whatever the downsampler does after that convolution reads one input position per output
    one: HunyuanVideo strides, and 1.5 folds each pair of rows and columns into channels. Either
    way a rank can do it to its own rows, provided it holds whole pairs of them, which the bands
    Patchify cuts guarantee by being whole multiples of what the encoder narrows by.
    """

    _supported: Tuple[type, ...] = ()
    _requires: str = ""
    _conv_adapter = None

    def __init__(
        self,
        downsampler: nn.Module,
        conv_block_size = 0,
        patch_dim: int = -2,
        use_uniform_patch: bool = False,
    ):
        super().__init__()
        adapter = type(self).__name__
        require(self._supported, adapter, self._requires)
        assert isinstance(downsampler, self._supported), (
            f"{adapter} does not support downsampler except {self._requires}"
        )
        self.downsampler = downsampler
        downsampler.conv = self._conv_adapter(
            downsampler.conv,
            block_size=conv_block_size,
            patch_dim=patch_dim,
            use_uniform_patch=use_uniform_patch,
        )

    def forward(self, hidden_states):
        return self.downsampler(hidden_states)


class HunyuanVideoDownsampleAdapter(_PaddedCausalDownsampleAdapter):
    _supported = resolved(HunyuanVideoDownsampleCausal3D)
    _requires = "HunyuanVideoDownsampleCausal3D"
    _conv_adapter = HunyuanVideoCausalConv3dAdapter


class HunyuanVideo15DownsampleAdapter(_PaddedCausalDownsampleAdapter):
    _supported = resolved(HunyuanVideo15Downsample)
    _requires = "HunyuanVideo15Downsample"
    _conv_adapter = HunyuanVideo15CausalConv3dAdapter


class _PaddedCausalDownBlockAdapter(nn.Module):
    """Shards a HunyuanVideo down block: its residual blocks and its downsampler"""

    _supported: Tuple[type, ...] = ()
    _requires: str = ""
    _resnet_adapter = None
    _downsample_adapter = None

    def __init__(
        self,
        down_block: nn.Module,
        conv_block_size = 0,
        patch_dim: int = -2,
        use_uniform_patch: bool = False,
    ):
        super().__init__()
        adapter = type(self).__name__
        require(self._supported, adapter, self._requires)
        assert isinstance(down_block, self._supported), (
            f"{adapter} does not support down block except {self._requires}"
        )
        options = dict(
            conv_block_size=conv_block_size,
            patch_dim=patch_dim,
            use_uniform_patch=use_uniform_patch,
        )
        self.down_block = down_block
        down_block.resnets = nn.ModuleList(
            [self._resnet_adapter(resnet, **options) for resnet in down_block.resnets]
        )
        if down_block.downsamplers is not None:
            down_block.downsamplers = nn.ModuleList(
                [self._downsample_adapter(down, **options) for down in down_block.downsamplers]
            )

    def forward(self, hidden_states):
        return self.down_block(hidden_states)


class HunyuanVideoDownBlockAdapter(_PaddedCausalDownBlockAdapter):
    _supported = resolved(HunyuanVideoDownBlock3D)
    _requires = "HunyuanVideoDownBlock3D"
    _resnet_adapter = HunyuanVideoResnetBlockAdapter
    _downsample_adapter = HunyuanVideoDownsampleAdapter


class HunyuanVideo15DownBlockAdapter(_PaddedCausalDownBlockAdapter):
    _supported = resolved(HunyuanVideo15DownBlock3D)
    _requires = "HunyuanVideo15DownBlock3D"
    _resnet_adapter = HunyuanVideo15ResnetBlockAdapter
    _downsample_adapter = HunyuanVideo15DownsampleAdapter


class LTX2VideoDownsamplerAdapter(nn.Module):
    """Shards an LTX-2 downsampler, which is its convolution

    What follows the convolution moves space into channels and averages the input the same way
    for the residual, reading one input position per output one, so a rank can do it to its own
    rows alone.
    """

    _supported = resolved(LTX2VideoDownsampler3d)
    _requires = "LTX2VideoDownsampler3d"

    def __init__(
        self,
        downsampler: nn.Module,
        conv_block_size = 0,
        patch_dim: int = -2,
        use_uniform_patch: bool = False,
    ):
        super().__init__()
        adapter = type(self).__name__
        require(self._supported, adapter, self._requires)
        assert isinstance(downsampler, self._supported), (
            f"{adapter} does not support downsampler except {self._requires}"
        )
        self.downsampler = downsampler
        downsampler.conv = LTX2VideoCausalConv3dAdapter(
            downsampler.conv,
            block_size=conv_block_size,
            patch_dim=patch_dim,
            use_uniform_patch=use_uniform_patch,
        )

    def forward(self, hidden_states, causal: bool = True):
        return self.downsampler(hidden_states, causal=causal)


class LTX2VideoDownBlockAdapter(nn.Module):
    """Shards an LTX-2 down block: its residual blocks and its downsampler

    Which downsampler that is depends on how the stage was configured: a strided causal
    convolution where it downsamples by striding, or the space-to-channel downsampler where it
    does so by folding. Both are handled because a checkpoint may hold either.
    """

    _supported = resolved(LTX2VideoDownBlock3D)
    _requires = "LTX2VideoDownBlock3D"

    def __init__(
        self,
        down_block: nn.Module,
        conv_block_size = 0,
        patch_dim: int = -2,
        use_uniform_patch: bool = False,
    ):
        super().__init__()
        adapter = type(self).__name__
        require(self._supported, adapter, self._requires)
        assert isinstance(down_block, self._supported), (
            f"{adapter} does not support down block except {self._requires}"
        )
        options = dict(
            conv_block_size=conv_block_size,
            patch_dim=patch_dim,
            use_uniform_patch=use_uniform_patch,
        )
        self.down_block = down_block
        down_block.resnets = nn.ModuleList(
            [LTX2VideoResnetBlockAdapter(resnet, **options) for resnet in down_block.resnets]
        )
        if down_block.downsamplers is not None:
            down_block.downsamplers = nn.ModuleList(
                [self._adapt_downsampler(down, adapter, conv_block_size, patch_dim,
                                         use_uniform_patch)
                 for down in down_block.downsamplers]
            )

    @staticmethod
    def _adapt_downsampler(downsampler, adapter, conv_block_size, patch_dim, use_uniform_patch):
        if LTX2VideoDownsampler3d is not None and isinstance(downsampler, LTX2VideoDownsampler3d):
            return LTX2VideoDownsamplerAdapter(
                downsampler,
                conv_block_size=conv_block_size,
                patch_dim=patch_dim,
                use_uniform_patch=use_uniform_patch,
            )
        if LTX2VideoCausalConv3d is not None and isinstance(downsampler, LTX2VideoCausalConv3d):
            return LTX2VideoCausalConv3dAdapter(
                downsampler,
                block_size=conv_block_size,
                patch_dim=patch_dim,
                use_uniform_patch=use_uniform_patch,
            )
        raise TypeError(
            f"{adapter} cannot shard a downsampler of type {type(downsampler).__name__}. It "
            f"handles LTX2VideoDownsampler3d and LTX2VideoCausalConv3d."
        )

    def forward(self, hidden_states, temb=None, generator=None, causal: bool = True):
        return self.down_block(hidden_states, temb, generator, causal=causal)


class WanResidualDownBlockAdapter(nn.Module):
    """
    Adapter for WanResidualDownBlock used in the encoder (Wan2.2).
    Patches residual blocks and downsampler with distributed processing support.
    """
    def __init__(
        self,
        wan_residual_down_block: WanResidualDownBlock,
        conv_block_size = 0,
        patch_dim: int = -2,
        use_uniform_patch: bool = True,
    ):
        super().__init__()
        assert isinstance(wan_residual_down_block, WanResidualDownBlock), (
            "WanResidualDownBlockAdapter only supports WanResidualDownBlock"
        )
        if patch_dim == -3:
            raise ValueError("WanResidualDownBlockAdapter does not support patch_dim F (-3); use H (-2) or W (-1).")

        self.down_block = wan_residual_down_block
        if hasattr(wan_residual_down_block, "resnets"):
            adapted_resnets = []
            for resnet in wan_residual_down_block.resnets:
                adapted_resnets.append(
                    WanResidualBlockAdapter(
                        resnet,
                        conv_block_size=conv_block_size,
                        patch_dim=patch_dim,
                        use_uniform_patch=use_uniform_patch
                    )
                )
            self.down_block.resnets = nn.ModuleList(adapted_resnets)
        if hasattr(wan_residual_down_block, "downsampler") and wan_residual_down_block.downsampler is not None:
            if isinstance(wan_residual_down_block.downsampler, WanResample):
                self.down_block.downsampler = WanResampleDownAdapter(
                    wan_residual_down_block.downsampler,
                    conv_block_size=conv_block_size,
                    patch_dim=patch_dim,
                    use_uniform_patch=use_uniform_patch
                )

    def forward(self, hidden_states, feat_cache=None, feat_idx=[0]):
        return self.down_block(hidden_states, feat_cache=feat_cache, feat_idx=feat_idx)
