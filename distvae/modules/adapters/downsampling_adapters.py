from typing import Tuple

import torch.nn as nn

from distvae.models.layers.wan.zeropadconv2d import WanZeroPadConv2d
from distvae.modules.adapters.diffusers_blocks import (
    QWEN_IMAGE,
    block,
    require,
    resolved,
)
from distvae.modules.adapters.layers.conv_adapters import (
    Conv2dAdapter,
    QwenImageCausalConv3dAdapter,
    WanCausalConv3dAdapter,
)
from distvae.modules.adapters.resnet_adapters import WanResidualBlockAdapter
from diffusers.models.autoencoders.autoencoder_kl_wan import WanResample, WanResidualDownBlock

QwenImageResample = block(QWEN_IMAGE, "QwenImageResample")


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
            conv = convs[0]
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
            resample.resample = sharded
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
