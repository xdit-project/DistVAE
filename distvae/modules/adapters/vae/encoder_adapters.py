from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.distributed import ProcessGroup

from distvae.modules.adapters.diffusers_blocks import (
    HUNYUAN_VIDEO,
    HUNYUAN_VIDEO_15,
    LTX2_VIDEO,
    QWEN_IMAGE,
    block,
)
from distvae.modules.adapters.downsampling_adapters import (
    HunyuanVideo15DownBlockAdapter,
    HunyuanVideoDownBlockAdapter,
    LTX2VideoDownBlockAdapter,
    QwenImageResampleDownAdapter,
    WanResampleDownAdapter,
    WanResidualDownBlockAdapter,
)
from distvae.modules.adapters.layers.attn_adapters import GatheredAttentionAdapter
from distvae.modules.adapters.layers.conv_adapters import (
    HunyuanVideo15CausalConv3dAdapter,
    HunyuanVideoCausalConv3dAdapter,
    LTX2VideoCausalConv3dAdapter,
    QwenImageCausalConv3dAdapter,
    WanCausalConv3dAdapter,
)
from distvae.modules.adapters.layers.norm_adapters import GroupNormAdapter
from distvae.modules.adapters.midblock_adapters import (
    HunyuanVideo15MidBlockAdapter,
    HunyuanVideoMidBlockAdapter,
    LTX2VideoMidBlockAdapter,
    QwenImageMidBlockAdapter,
    WanMidBlockAdapter,
)
from distvae.modules.adapters.resnet_adapters import (
    QwenImageResidualBlockAdapter,
    WanResidualBlockAdapter,
)
from distvae.modules.patch_utils import Patchify, DePatchify
from distvae.utils import DistributedEnv

from diffusers.models.autoencoders.autoencoder_kl_wan import (
    WanAttentionBlock,
    WanResample,
    WanResidualBlock,
    WanResidualDownBlock,
)

QwenImageAttentionBlock = block(QWEN_IMAGE, "QwenImageAttentionBlock")
QwenImageResample = block(QWEN_IMAGE, "QwenImageResample")
QwenImageResidualBlock = block(QWEN_IMAGE, "QwenImageResidualBlock")
HunyuanVideoDownBlock3D = block(HUNYUAN_VIDEO, "HunyuanVideoDownBlock3D")
HunyuanVideo15DownBlock3D = block(HUNYUAN_VIDEO_15, "HunyuanVideo15DownBlock3D")
LTX2VideoDownBlock3D = block(LTX2_VIDEO, "LTX2VideoDownBlock3D")


def _gathered(attention: nn.Module, **options) -> nn.Module:
    """Adapt an attention block, which needs the whole image rather than a patch of it

    Written as a function so it can sit in a down block table beside the adapters that shard a
    convolution, none of whose sizing options a gather has any use for.
    """
    return GatheredAttentionAdapter(attention, patch_dim=options["patch_dim"])


class _CausalEncoderAdapter(nn.Module):
    """Shards a causal 3D video encoder across ranks along one spatial axis.

    The mirror of _CausalDecoderAdapter, over the same skeleton read the other way: a causal
    convolution in, a run of down blocks, a mid block, a normalisation, a causal convolution
    out. What differs is the arithmetic at the end. An encoder narrows what it is handed, so the
    rows a rank owns are divided by the VAE's spatial ratio where a decoder multiplies them by
    what it upsampled, and the input is padded up to a multiple of that ratio times the rank
    count so the division lands whole.
    """

    _label = "Encoder"
    _conv_adapter = None
    _mid_adapter = None
    # Which adapter fits which down block class. A family whose blocks are not in the installed
    # diffusers leaves None in the type slot, which no block can match.
    _down_block_adapters: Tuple[Tuple[Optional[type], object], ...] = ()
    # Wan and the family forked from it thread a temporal cache through every forward. The
    # HunyuanVideo and LTX-2 encoders take a tensor and nothing else.
    _takes_feature_cache = True

    def __init__(
        self,
        encoder: nn.Module,
        vae_group: ProcessGroup = None,
        *,
        vae_scale_factor: int = 8,
        conv_block_size = 0,
        patch_dim: int = -2,
    ):
        super().__init__()
        adapter = type(self).__name__
        if patch_dim == -3:
            raise ValueError(
                f"{adapter} does not support patch_dim F (-3); use H (-2) or W (-1)."
            )
        DistributedEnv.initialize(vae_group)
        self.patch_dim = patch_dim
        DistributedEnv.set_patch_dim(patch_dim)
        self.vae_scale_factor = vae_scale_factor
        # Bands differ in size where the rows do not divide by the rank count, so every
        # convolution has to read the sizes rather than assume its neighbours match it.
        options = dict(patch_dim=patch_dim, use_uniform_patch=False)
        self.encoder = encoder
        self.encoder.conv_in = self._conv_adapter(
            encoder.conv_in, block_size=conv_block_size, **options
        )
        self.encoder.down_blocks = nn.ModuleList([
            self._adapt_down_block(down_block, adapter, conv_block_size, options)
            for down_block in encoder.down_blocks
        ])
        self.encoder.mid_block = self._mid_adapter(
            encoder.mid_block, conv_block_size=conv_block_size, **options
        )
        self.encoder.conv_out = self._conv_adapter(
            encoder.conv_out, block_size=conv_block_size, **options
        )
        # HunyuanVideo ends on a GroupNorm, whose statistics span the axis being split. The RMS
        # norms the other families end on do not, and are left as they are.
        if isinstance(getattr(encoder, "conv_norm_out", None), nn.GroupNorm):
            self.encoder.conv_norm_out = GroupNormAdapter(encoder.conv_norm_out)
        # Each band is a whole multiple of what the encoder narrows by, so it starts on the grid
        # the strided convolutions step along and the latent rows it produces are its own.
        self.patchify = Patchify(patch_dim=patch_dim, scale_factor=vae_scale_factor)
        self.depatchify = DePatchify(patch_dim=patch_dim)
        self.vae_group = vae_group

    @classmethod
    def _adapt_down_block(cls, down_block, adapter, conv_block_size, options):
        for block_type, block_adapter in cls._down_block_adapters:
            if block_type is not None and isinstance(down_block, block_type):
                return block_adapter(down_block, conv_block_size=conv_block_size, **options)
        handled = ", ".join(t.__name__ for t, _ in cls._down_block_adapters if t is not None)
        raise TypeError(
            f"{adapter} cannot shard a down block of type {type(down_block).__name__}. "
            f"It handles {handled or 'no down block type the installed diffusers provides'}."
        )

    def _run_encoder(self, sample, feat_cache, feat_idx):
        if not self._takes_feature_cache:
            return self.encoder(sample)
        return self.encoder(sample, feat_cache=feat_cache, feat_idx=feat_idx)

    def _sharded_encode(self, sample: torch.FloatTensor, patchify: bool, run):
        """Split the sample across ranks, encode this rank's share, and reassemble

        Kept apart from forward because the families do not agree on what an encoder call looks
        like: some thread a temporal cache through it, LTX-2 takes a causal flag. Splitting and
        reassembling is the same either way.
        """
        if patchify:
            sample = self.patchify(sample)
        return self.depatchify(run(sample))

    def forward(
        self,
        sample: torch.FloatTensor,
        feat_cache: Optional[torch.FloatTensor] = None,
        feat_idx: Optional[int] = 0,
        patchify: bool = True,
    ):
        return self._sharded_encode(
            sample, patchify, lambda x: self._run_encoder(x, feat_cache, feat_idx)
        )


class WanEncoderAdapter(_CausalEncoderAdapter):
    """Wan's encoder, whose down blocks come either grouped or one layer at a time

    Wan 2.2 wraps each stage in a WanResidualDownBlock; Wan 2.1 lays the same residual blocks,
    attentions and resamples out flat in one list. Both ship, and the encoder class alone does
    not say which, so both shapes are handled.
    """

    _label = "WanEncoder"
    _conv_adapter = WanCausalConv3dAdapter
    _mid_adapter = WanMidBlockAdapter
    _down_block_adapters = (
        (WanResidualDownBlock, WanResidualDownBlockAdapter),
        (WanResidualBlock, WanResidualBlockAdapter),
        (WanResample, WanResampleDownAdapter),
        (WanAttentionBlock, _gathered),
    )


class QwenImageEncoderAdapter(_CausalEncoderAdapter):
    """Qwen-Image's encoder, which is Wan 2.1's laid out flat and renamed

    Its resample carries the same zero-pad-then-strided-convolution downsample as Wan's, so the
    one thing it does not inherit outright is the residual down block Wan 2.2 groups its stages
    into, which Qwen-Image has no equivalent of.
    """

    _label = "QwenImageEncoder"
    _conv_adapter = QwenImageCausalConv3dAdapter
    _mid_adapter = QwenImageMidBlockAdapter
    _down_block_adapters = (
        (QwenImageResidualBlock, QwenImageResidualBlockAdapter),
        (QwenImageResample, QwenImageResampleDownAdapter),
        (QwenImageAttentionBlock, _gathered),
    )


class HunyuanVideoEncoderAdapter(_CausalEncoderAdapter):
    """HunyuanVideo's encoder, which groups its stages and ends on a GroupNorm

    That norm reduces over the axis being split, so the base wraps it. Its mid block holds
    diffusers' own attention, flattened over frames and rows and columns together, which the mid
    block adapter gathers around rather than trying to shard.
    """

    _label = "HunyuanVideoEncoder"
    _conv_adapter = HunyuanVideoCausalConv3dAdapter
    _mid_adapter = HunyuanVideoMidBlockAdapter
    _down_block_adapters = ((HunyuanVideoDownBlock3D, HunyuanVideoDownBlockAdapter),)
    _takes_feature_cache = False


class HunyuanVideo15EncoderAdapter(_CausalEncoderAdapter):
    """HunyuanVideo 1.5's encoder, which downsamples by folding space into channels

    It ends on an RMS norm, which reduces over channels and so needs no sharding. Its downsampler
    packs each pair of rows and columns into channels, which reads one input position per output
    one so long as a rank holds whole pairs of rows, and the bands Patchify cuts do.
    """

    _label = "HunyuanVideo15Encoder"
    _conv_adapter = HunyuanVideo15CausalConv3dAdapter
    _mid_adapter = HunyuanVideo15MidBlockAdapter
    _down_block_adapters = ((HunyuanVideo15DownBlock3D, HunyuanVideo15DownBlockAdapter),)
    _takes_feature_cache = False


class LTX2VideoEncoderAdapter(_CausalEncoderAdapter):
    """LTX-2's encoder, which takes a causal flag where the others take a temporal cache

    Its mid block holds no attention, so nothing here has to be gathered: every layer is a
    convolution or a norm that reduces over channels.
    """

    _label = "LTX2VideoEncoder"
    _conv_adapter = LTX2VideoCausalConv3dAdapter
    _mid_adapter = LTX2VideoMidBlockAdapter
    _down_block_adapters = ((LTX2VideoDownBlock3D, LTX2VideoDownBlockAdapter),)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        causal: Optional[bool] = None,
        patchify: bool = True,
    ):
        return self._sharded_encode(hidden_states, patchify, lambda x: self.encoder(x, causal))
