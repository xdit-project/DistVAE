from typing import Tuple

import torch.nn as nn
from diffusers.models.autoencoders.autoencoder_kl_wan import WanMidBlock

from distvae.modules.adapters.diffusers_blocks import (
    HUNYUAN_VIDEO,
    HUNYUAN_VIDEO_15,
    LTX2_VIDEO,
    QWEN_IMAGE,
    block,
    require,
    resolved,
)
from distvae.modules.adapters.layers.attn_adapters import GatheredAttentionAdapter
from distvae.utils import ParallelContext, cache_cursor
from distvae.modules.adapters.resnet_adapters import (
    HunyuanVideo15ResnetBlockAdapter,
    HunyuanVideoResnetBlockAdapter,
    LTX2VideoResnetBlockAdapter,
    QwenImageResidualBlockAdapter,
    WanResidualBlockAdapter,
)

QwenImageMidBlock = block(QWEN_IMAGE, "QwenImageMidBlock")
HunyuanVideoMidBlock3D = block(HUNYUAN_VIDEO, "HunyuanVideoMidBlock3D")
HunyuanVideo15MidBlock = block(HUNYUAN_VIDEO_15, "HunyuanVideo15MidBlock")
LTX2VideoMidBlock3d = block(LTX2_VIDEO, "LTX2VideoMidBlock3d")


class _CausalMidBlockAdapter(nn.Module):
    """Shards a mid block: its residual blocks stay local, its attentions have to gather"""

    _supported: Tuple[type, ...] = ()
    _requires: str = ""
    _resnet_adapter = None

    def __init__(
        self,
        mid_block: nn.Module,
        conv_block_size = 0,
        parallel_context: ParallelContext = None,
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
                parallel_context=parallel_context,
            ) for resnet in mid_block.resnets
        ])
        self.mid_block.attentions = nn.ModuleList([
            GatheredAttentionAdapter(attn, parallel_context=parallel_context)
            if attn is not None else attn
            for attn in mid_block.attentions
        ])

    def forward(self, x, feat_cache=None, feat_idx=None):
        return self.mid_block(x, feat_cache=feat_cache, feat_idx=cache_cursor(feat_idx))


class WanMidBlockAdapter(_CausalMidBlockAdapter):
    _supported = resolved(WanMidBlock)
    _requires = "WanMidBlock"
    _resnet_adapter = WanResidualBlockAdapter


class QwenImageMidBlockAdapter(_CausalMidBlockAdapter):
    _supported = resolved(QwenImageMidBlock)
    _requires = "QwenImageMidBlock"
    _resnet_adapter = QwenImageResidualBlockAdapter


class HunyuanVideo15MidBlockAdapter(_CausalMidBlockAdapter):
    """Shards HunyuanVideo 1.5's mid block: residual blocks stay local, attentions gather"""

    _supported = resolved(HunyuanVideo15MidBlock)
    _requires = "HunyuanVideo15MidBlock"
    _resnet_adapter = HunyuanVideo15ResnetBlockAdapter

    def forward(self, hidden_states):
        return self.mid_block(hidden_states)


class HunyuanVideoMidBlockAdapter(nn.Module):
    """Shards HunyuanVideo's mid block, or gathers around the whole of it when it has attention.

    Its attention cannot be wrapped on its own the way every other family's can. The mid block
    flattens (F, H, W) into a sequence and builds the causal mask itself, both from the height
    it can see, so an attention handed a patch would also be handed a mask cut for a patch and
    would quietly attend over the wrong span. Gathering around the entire block avoids
    reimplementing that forward, and costs little: the mid block runs at the latent resolution,
    which is the cheapest point in the decoder, and it is the up blocks after it that hold the
    activations worth splitting.
    """

    def __init__(
        self,
        mid_block: nn.Module,
        conv_block_size = 0,
        parallel_context: ParallelContext = None,
    ):
        super().__init__()
        adapter = type(self).__name__
        supported = resolved(HunyuanVideoMidBlock3D)
        require(supported, adapter, "HunyuanVideoMidBlock3D")
        assert isinstance(mid_block, supported), (
            f"{adapter} does not support mid block except HunyuanVideoMidBlock3D"
        )
        if any(attn is not None for attn in mid_block.attentions):
            self.mid_block = GatheredAttentionAdapter(
                mid_block, parallel_context=parallel_context
            )
        else:
            mid_block.resnets = nn.ModuleList([
                HunyuanVideoResnetBlockAdapter(
                    resnet,
                    conv_block_size=conv_block_size,
                    parallel_context=parallel_context,
                ) for resnet in mid_block.resnets
            ])
            self.mid_block = mid_block

    def forward(self, hidden_states):
        return self.mid_block(hidden_states)


class LTX2VideoMidBlockAdapter(nn.Module):
    """Shards an LTX-2 mid block, which is only residual blocks

    Alone among these families LTX-2 puts no attention in its mid block, so nothing here needs
    to see the whole image and every rank can stay on its own patch throughout.
    """

    def __init__(
        self,
        mid_block: nn.Module,
        conv_block_size = 0,
        parallel_context: ParallelContext = None,
    ):
        super().__init__()
        adapter = type(self).__name__
        supported = resolved(LTX2VideoMidBlock3d)
        require(supported, adapter, "LTX2VideoMidBlock3d")
        assert isinstance(mid_block, supported), (
            f"{adapter} does not support mid block except LTX2VideoMidBlock3d"
        )
        self.mid_block = mid_block
        mid_block.resnets = nn.ModuleList([
            LTX2VideoResnetBlockAdapter(
                resnet,
                conv_block_size=conv_block_size,
                parallel_context=parallel_context,
            ) for resnet in mid_block.resnets
        ])

    def forward(self, hidden_states, temb=None, generator=None, causal: bool = True):
        return self.mid_block(hidden_states, temb, generator, causal=causal)
