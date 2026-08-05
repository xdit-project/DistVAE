from typing import Tuple

import torch
import torch.nn as nn

from distvae.models.resnet import PatchResnetBlock2D
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
from distvae.modules.adapters.layers.norm_adapters import GroupNormAdapter
from diffusers.models.resnet import ResnetBlock2D
from diffusers.models.autoencoders.autoencoder_kl_wan import WanCausalConv3d, WanResidualBlock

QwenImageResidualBlock = block(QWEN_IMAGE, "QwenImageResidualBlock")
HunyuanVideoResnetBlockCausal3D = block(HUNYUAN_VIDEO, "HunyuanVideoResnetBlockCausal3D")
HunyuanVideo15ResnetBlock = block(HUNYUAN_VIDEO_15, "HunyuanVideo15ResnetBlock")
LTX2VideoResnetBlock3d = block(LTX2_VIDEO, "LTX2VideoResnetBlock3d")


class ResnetBlock2DAdapter(nn.Module):
    def __init__(
        self, 
        resnet: ResnetBlock2D, 
        *, 
        conv_block_size = 0,
    ):
        super().__init__()
        assert resnet.time_emb_proj is None, "temb_channels is not supported in ResnetBlock2DAdapter currently"
        assert resnet.up is False, "up sample is not supported in ResnetBlock2DAdapter currently"
        assert resnet.down is False, "ResnetBlock2DAdapter does not support down sample currently"
        self.resnet = PatchResnetBlock2D(
            in_channels=resnet.in_channels,
            out_channels=resnet.out_channels,
            conv_shortcut=resnet.use_conv_shortcut,
            dropout=0,
            temb_channels=None,
            groups=1,
            groups_out=None,
            pre_norm=resnet.pre_norm,
            skip_time_act=resnet.skip_time_act,
            time_embedding_norm=resnet.time_embedding_norm,
            output_scale_factor=resnet.output_scale_factor,
            use_in_shortcut=resnet.use_in_shortcut,
            up=resnet.up,
            down=resnet.down,
        )
        self.resnet.use_in_shortcut = resnet.use_in_shortcut
        self.resnet.conv1 = Conv2dAdapter(resnet.conv1, block_size=conv_block_size)
        self.resnet.norm1 = GroupNormAdapter(resnet.norm1)
        self.resnet.conv2 = Conv2dAdapter(resnet.conv2, block_size=conv_block_size)
        self.resnet.norm2 = GroupNormAdapter(resnet.norm2)
        self.resnet.dropout = resnet.dropout
        self.resnet.nonlinearity = resnet.nonlinearity
        self.resnet.conv_shortcut = Conv2dAdapter(resnet.conv_shortcut, block_size=conv_block_size) if resnet.conv_shortcut is not None else None
        

    def forward(self, x, temb: torch.FloatTensor = None, *args, **kwargs):
        return self.resnet(x, temb, *args, **kwargs)


class _CausalResidualBlockAdapter(nn.Module):
    """Shards a residual block built from two causal 3D convolutions and an optional shortcut.

    The norms either side of them are RMS, which reduces over channels and so needs nothing from
    the other ranks; only the convolutions reach across the split.
    """

    _supported: Tuple[type, ...] = ()
    _requires: str = ""
    _conv_adapter = None

    def __init__(
        self,
        residual_block: nn.Module,
        conv_block_size = 0,
        patch_dim: int = -2,
    ):
        super().__init__()
        adapter = type(self).__name__
        require(self._supported, adapter, self._requires)
        assert isinstance(residual_block, self._supported), (
            f"{adapter} does not support resnet except {self._requires}"
        )
        self.residual_block = residual_block
        for name in ("conv1", "conv2"):
            setattr(
                self.residual_block,
                name,
                self._conv_adapter(
                    getattr(residual_block, name),
                    block_size=conv_block_size,
                    patch_dim=patch_dim,
                ),
            )
        # Adapt conv_shortcut if it's not nn.Identity
        if not isinstance(residual_block.conv_shortcut, nn.Identity):
            self.residual_block.conv_shortcut = self._conv_adapter(
                residual_block.conv_shortcut,
                block_size=conv_block_size,
                patch_dim=patch_dim,
            )

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        return self.residual_block(x, feat_cache=feat_cache, feat_idx=feat_idx)


class WanResidualBlockAdapter(_CausalResidualBlockAdapter):
    _supported = resolved(WanResidualBlock)
    _requires = "WanResidualBlock"
    _conv_adapter = WanCausalConv3dAdapter


class QwenImageResidualBlockAdapter(_CausalResidualBlockAdapter):
    _supported = resolved(QwenImageResidualBlock)
    _requires = "QwenImageResidualBlock"
    _conv_adapter = QwenImageCausalConv3dAdapter


class _PaddedCausalResnetBlockAdapter(nn.Module):
    """Shards a HunyuanVideo residual block: its two causal convolutions, and any GroupNorms

    HunyuanVideo normalises with GroupNorm, whose statistics span the axis being split and so
    have to be summed across ranks. HunyuanVideo 1.5 replaced those with RMS, which reduces over
    channels and needs nothing from anyone; the isinstance check below is what tells them apart.
    """

    _supported: Tuple[type, ...] = ()
    _requires: str = ""
    _conv_adapter = None

    def __init__(
        self,
        resnet: nn.Module,
        conv_block_size = 0,
        patch_dim: int = -2,
    ):
        super().__init__()
        adapter = type(self).__name__
        require(self._supported, adapter, self._requires)
        assert isinstance(resnet, self._supported), (
            f"{adapter} does not support resnet except {self._requires}"
        )
        self.resnet = resnet
        for name in ("conv1", "conv2"):
            setattr(
                resnet,
                name,
                self._conv_adapter(
                    getattr(resnet, name),
                    block_size=conv_block_size,
                    patch_dim=patch_dim,
                ),
            )
        for name in ("norm1", "norm2"):
            norm = getattr(resnet, name)
            if isinstance(norm, nn.GroupNorm):
                setattr(resnet, name, GroupNormAdapter(norm))
        # Where the shortcut is a causal convolution it needs the same treatment; where it is a
        # bare 1x1x1 it reads one position per output and is already right on a patch.
        if isinstance(resnet.conv_shortcut, self._conv_adapter._supported):
            resnet.conv_shortcut = self._conv_adapter(
                resnet.conv_shortcut,
                block_size=conv_block_size,
                patch_dim=patch_dim,
            )

    def forward(self, hidden_states):
        return self.resnet(hidden_states)


class HunyuanVideoResnetBlockAdapter(_PaddedCausalResnetBlockAdapter):
    _supported = resolved(HunyuanVideoResnetBlockCausal3D)
    _requires = "HunyuanVideoResnetBlockCausal3D"
    _conv_adapter = HunyuanVideoCausalConv3dAdapter


class HunyuanVideo15ResnetBlockAdapter(_PaddedCausalResnetBlockAdapter):
    _supported = resolved(HunyuanVideo15ResnetBlock)
    _requires = "HunyuanVideo15ResnetBlock"
    _conv_adapter = HunyuanVideo15CausalConv3dAdapter


class LTX2VideoResnetBlockAdapter(nn.Module):
    """Shards an LTX-2 residual block, which is its two convolutions and nothing else.

    Both its norms reduce over channels, its shortcut is a 1x1x1 convolution reading one
    position per output, and its timestep conditioning arrives shaped to broadcast over space.
    """

    _supported = resolved(LTX2VideoResnetBlock3d)
    _requires = "LTX2VideoResnetBlock3d"

    def __init__(
        self,
        resnet: nn.Module,
        conv_block_size = 0,
        patch_dim: int = -2,
    ):
        super().__init__()
        adapter = type(self).__name__
        require(self._supported, adapter, self._requires)
        assert isinstance(resnet, self._supported), (
            f"{adapter} does not support resnet except {self._requires}"
        )
        if resnet.per_channel_scale1 is not None or resnet.per_channel_scale2 is not None:
            # Each rank would draw its own noise for its own rows, and the ranks together would
            # not reconstruct the field a single one draws, so a sharded decode could not match
            # an unsharded one at all. No shipped LTX-2 or LTX-2.3 config turns this on.
            raise NotImplementedError(
                f"{adapter} cannot shard a residual block with inject_noise enabled: the noise "
                f"is drawn per rank and would not add up to the noise one rank draws. Decode "
                f"this VAE on a single rank, or tile it instead."
            )
        self.resnet = resnet
        for name in ("conv1", "conv2"):
            setattr(
                resnet,
                name,
                LTX2VideoCausalConv3dAdapter(
                    getattr(resnet, name),
                    block_size=conv_block_size,
                    patch_dim=patch_dim,
                ),
            )

    def forward(self, inputs, temb=None, generator=None, causal: bool = True):
        return self.resnet(inputs, temb, generator, causal=causal)
