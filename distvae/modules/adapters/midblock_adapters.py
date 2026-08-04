from typing import Tuple

import torch.nn as nn
from diffusers.models.autoencoders.autoencoder_kl_wan import WanMidBlock

from distvae.modules.adapters.diffusers_blocks import (
    QWEN_IMAGE,
    block,
    require,
    resolved,
)
from distvae.modules.adapters.layers.attn_adapters import GatheredAttentionAdapter
from distvae.modules.adapters.resnet_adapters import (
    QwenImageResidualBlockAdapter,
    WanResidualBlockAdapter,
)

QwenImageMidBlock = block(QWEN_IMAGE, "QwenImageMidBlock")


class _CausalMidBlockAdapter(nn.Module):
    """Shards a mid block: its residual blocks stay local, its attentions have to gather"""

    _supported: Tuple[type, ...] = ()
    _requires: str = ""
    _resnet_adapter = None

    def __init__(
        self,
        mid_block: nn.Module,
        conv_block_size = 0,
        patch_dim: int = -2,
        use_uniform_patch: bool = False,
    ):
        super().__init__()

        adapter = type(self).__name__
        require(self._supported, adapter, self._requires)
        assert isinstance(mid_block, self._supported), (
            f"{adapter} does not support mid block except {self._requires}"
        )
        self.mid_block = mid_block
        self.mid_block.resnets = nn.ModuleList([
            self._resnet_adapter(
                resnet,
                conv_block_size=conv_block_size,
                patch_dim=patch_dim,
                use_uniform_patch=use_uniform_patch,
            ) for resnet in mid_block.resnets
        ])
        self.mid_block.attentions = nn.ModuleList([
            GatheredAttentionAdapter(attn, patch_dim=patch_dim) if attn is not None else attn
            for attn in mid_block.attentions
        ])

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        return self.mid_block(x, feat_cache=feat_cache, feat_idx=feat_idx)


class WanMidBlockAdapter(_CausalMidBlockAdapter):
    _supported = resolved(WanMidBlock)
    _requires = "WanMidBlock"
    _resnet_adapter = WanResidualBlockAdapter


class QwenImageMidBlockAdapter(_CausalMidBlockAdapter):
    _supported = resolved(QwenImageMidBlock)
    _requires = "QwenImageMidBlock"
    _resnet_adapter = QwenImageResidualBlockAdapter
