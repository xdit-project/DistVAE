from typing import Tuple

import torch
import torch.nn as nn

from distvae.models.resnet import PatchResnetBlock2D
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
from distvae.modules.adapters.layers.norm_adapters import GroupNormAdapter
from diffusers.models.resnet import ResnetBlock2D
from diffusers.models.autoencoders.autoencoder_kl_wan import WanCausalConv3d, WanResidualBlock

QwenImageResidualBlock = block(QWEN_IMAGE, "QwenImageResidualBlock")


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
        use_uniform_patch: bool = False,
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
                    use_uniform_patch=use_uniform_patch,
                ),
            )
        # Adapt conv_shortcut if it's not nn.Identity
        if not isinstance(residual_block.conv_shortcut, nn.Identity):
            self.residual_block.conv_shortcut = self._conv_adapter(
                residual_block.conv_shortcut,
                block_size=conv_block_size,
                patch_dim=patch_dim,
                use_uniform_patch=use_uniform_patch,
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
