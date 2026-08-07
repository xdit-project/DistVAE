from typing import List, Optional, Tuple

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
    Conv2dAdapter,
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
from distvae.modules.adapters.unets.unet_2d_blocks_adapters import DownEncoderBlock2DAdapter
from distvae.modules.patch_utils import Patchify, DePatchify, narrowing, widest_halo
from distvae.utils import DistributedEnv, cache_cursor

from diffusers.models.autoencoders.vae import Encoder
from diffusers.models.unets.unet_2d_blocks import DownEncoderBlock2D
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


class EncoderAdapter(nn.Module):
    """Shards the 2D encoder AutoencoderKL and Flux.2 use, over its down blocks alone.

    The mirror of the 2D decoder adapter, which splits after its mid block rather than before.
    Here the split is undone before the mid block, so the attention in it and the GroupNorm after
    it see the whole feature map and need no sharding of their own. What that costs is running the
    narrowest part of the encoder on every rank, and what it buys is that the down blocks, which
    carry the image at full size and are the reason to encode in parallel at all, are the part
    that gets split.
    """

    def __init__(
        self,
        encoder: Encoder,
        vae_group: ProcessGroup = None,
        *,
        vae_scale_factor: int = 8,
        conv_block_size = 0,
        patch_dim: int = -2,
    ):
        super().__init__()
        adapter = type(self).__name__
        if patch_dim != -2:
            # The resnet adapter this reaches through splits H and says nothing about which axis.
            raise ValueError(f"{adapter} only supports patch_dim H (-2).")
        for down_block in encoder.down_blocks:
            assert isinstance(down_block, DownEncoderBlock2D), (
                f"{adapter} does not support down block except DownEncoderBlock2D"
            )
        # A band has to be a whole multiple of what the encoder narrows by, and here that can be
        # counted rather than taken on trust: one halving per stage that carries a downsampler.
        # A caller working from a config default rather than from the blocks would otherwise cut
        # bands that a later stage halves into a row it does not own.
        counted = 2 ** sum(
            1 for down_block in encoder.down_blocks if down_block.downsamplers
        )
        if vae_scale_factor != counted:
            raise ValueError(
                f"{adapter} was told this encoder narrows by {vae_scale_factor}, but its "
                f"down blocks narrow by {counted}."
            )
        DistributedEnv.initialize(vae_group)
        self.patch_dim = patch_dim
        DistributedEnv.set_patch_dim(patch_dim)
        self.encoder = encoder
        encoder.conv_in = Conv2dAdapter(encoder.conv_in, block_size=conv_block_size)
        encoder.down_blocks = nn.ModuleList([
            DownEncoderBlock2DAdapter(
                down_block, conv_block_size=conv_block_size, patch_dim=patch_dim
            )
            for down_block in encoder.down_blocks
        ])
        # Read after the whole stack is adapted, so it sees every convolution that will exchange.
        self.patchify = Patchify(
            patch_dim=patch_dim,
            scale_factor=vae_scale_factor,
            halo=widest_halo(self.encoder),
        )
        self.depatchify = DePatchify(patch_dim=patch_dim)
        self.vae_group = vae_group

    def forward(self, sample: torch.FloatTensor):
        sample = self.encoder.conv_in(self.patchify(sample))
        for down_block in self.encoder.down_blocks:
            sample = down_block(sample)
        sample = self.encoder.mid_block(self.depatchify(sample))
        sample = self.encoder.conv_act(self.encoder.conv_norm_out(sample))
        return self.encoder.conv_out(sample)


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
    out. What differs is what a band has to be a multiple of. An encoder narrows what it is
    handed, so a band is cut in whole multiples of the VAE's spatial ratio and the latent rows it
    produces are its own, where a decoder cuts latent rows and multiplies.
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
        options = dict(patch_dim=patch_dim)
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
            self.encoder.conv_norm_out = GroupNormAdapter(
                encoder.conv_norm_out, patch_dim=patch_dim
            )
        # Checked against the adapted stack rather than taken on trust, as the 2D adapter has
        # always done by counting its down blocks. A caller reading a ratio off a config can be
        # told a number the convolutions disagree with - a VAE stating its ratio under a name the
        # caller does not know falls back to a default of 8 for an encoder that narrows by 16 -
        # and the bands are then cut in eights for a stack that halves four times. That does not
        # fail here; it fails several stages down as an odd band, on whichever ranks drew one.
        counted = narrowing(self.encoder, patch_dim)
        if counted != vae_scale_factor:
            raise ValueError(
                f"{adapter} was told this encoder narrows by {vae_scale_factor}, but its "
                f"convolutions narrow the split axis by {counted}."
            )
        # Each band is a whole multiple of what the encoder narrows by, so it starts on the grid
        # the strided convolutions step along and the latent rows it produces are its own. Read
        # the halo after the whole stack is adapted, so it sees every convolution that exchanges.
        self.patchify = Patchify(
            patch_dim=patch_dim,
            scale_factor=vae_scale_factor,
            halo=widest_halo(self.encoder),
        )
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
        feat_idx: Optional[List[int]] = None,
        patchify: bool = True,
    ):
        # A one-element list the causal blocks advance in place; see the decoder's forward for
        # why it is neither a mutable default nor the bare 0 this used to take.
        feat_idx = cache_cursor(feat_idx)
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
