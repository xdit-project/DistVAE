from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch.distributed import ProcessGroup
from diffusers.models.autoencoders.vae import Decoder
from diffusers.models.unets.unet_2d_blocks import UpDecoderBlock2D
from diffusers.models.autoencoders.autoencoder_kl_wan import (
    WanUpBlock,
    WanResidualUpBlock,
)

from distvae.modules.adapters.diffusers_blocks import (
    HUNYUAN_VIDEO,
    HUNYUAN_VIDEO_15,
    LTX2_VIDEO,
    QWEN_IMAGE,
    block,
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
from distvae.modules.adapters.unets.unet_2d_blocks_adapters import UpDecoderBlock2DAdapter
from distvae.modules.adapters.vae.causal_setup import CausalVAEAdapterSetup
from distvae.modules.adapters.upsampling_adapters import (
    HunyuanVideo15UpBlockAdapter,
    HunyuanVideoUpBlockAdapter,
    LTX2VideoUpBlockAdapter,
    QwenImageUpBlockAdapter,
    WanResidualUpBlockAdapter,
    WanUpBlockAdapter,
)
from distvae.modules.adapters.midblock_adapters import (
    HunyuanVideo15MidBlockAdapter,
    HunyuanVideoMidBlockAdapter,
    LTX2VideoMidBlockAdapter,
    QwenImageMidBlockAdapter,
    WanMidBlockAdapter,
)
from distvae.modules.patch_utils import Patchify, DePatchify
from distvae.utils import (
    cache_cursor,
    normalize_patch_dim,
    parallel_context,
)

QwenImageUpBlock = block(QWEN_IMAGE, "QwenImageUpBlock")
HunyuanVideoUpBlock3D = block(HUNYUAN_VIDEO, "HunyuanVideoUpBlock3D")
HunyuanVideo15UpBlock3D = block(HUNYUAN_VIDEO_15, "HunyuanVideo15UpBlock3D")
LTX2VideoUpBlock3d = block(LTX2_VIDEO, "LTX2VideoUpBlock3d")


def _reject_benchmark_options(use_profiler: bool, verbose: bool):
    if use_profiler or verbose:
        raise ValueError(
            "Decoder adapter profiling and verbose timing moved to the bench harness; "
            "run bench/distvae_bench.py for benchmark instrumentation."
        )


class DecoderAdapter(nn.Module):
    def __init__(
        self, 
        decoder: Decoder, 
        vae_group: ProcessGroup = None,
        *,
        use_profiler: bool = False,
        verbose: bool = False,
        conv_block_size = 0,
        patch_dim: int = -2,
    ):
        super().__init__()
        _reject_benchmark_options(use_profiler, verbose)
        assert isinstance(decoder.conv_norm_out, nn.GroupNorm), "DecoderAdapter does not support normalization method except GroupNorm"
        for up_block in decoder.up_blocks:
            assert isinstance(up_block, UpDecoderBlock2D), "DecoderAdapter does not support up block except UpDecoderBlock2D"
        patch_dim = normalize_patch_dim(patch_dim, 4, spatial_only=True)
        self.patch_dim = patch_dim
        self.parallel_context = parallel_context(vae_group, patch_dim, ndim=4)
        options = dict(parallel_context=self.parallel_context)
        self.decoder = decoder
        self.decoder.up_blocks = nn.ModuleList([
            UpDecoderBlock2DAdapter(
                up_block, conv_block_size=conv_block_size, **options
            ) for up_block in decoder.up_blocks
        ])
        self.decoder.conv_norm_out = GroupNormAdapter(decoder.conv_norm_out, **options)
        self.decoder.conv_act = decoder.conv_act
        self.decoder.conv_out = Conv2dAdapter(
            decoder.conv_out, block_size=conv_block_size, **options
        )
        self.patch = Patchify(**options)
        self.depatch = DePatchify(**options)
        self.vae_group = vae_group
        self.train(decoder.training)

    def forward(
        self,
        sample: torch.FloatTensor,
        latent_embeds: Optional[torch.FloatTensor] = None,
    ):
        if torch.is_grad_enabled():
            raise RuntimeError(
                "DecoderAdapter is inference-only; use torch.no_grad() or inference mode "
                "(torch.inference_mode())."
            )

        decoder = self.decoder
        sample = decoder.conv_in(sample)
        upscale_dtype = next(iter(decoder.up_blocks.parameters())).dtype

        sample = decoder.mid_block(sample, latent_embeds)
        sample = sample.to(upscale_dtype)
        sample = self.patch(sample)
        for up_block in decoder.up_blocks:
            sample = up_block(sample, latent_embeds)

        if latent_embeds is None:
            sample = decoder.conv_norm_out(sample)
        else:
            sample = decoder.conv_norm_out(sample, latent_embeds)
        sample = decoder.conv_act(sample)
        sample = decoder.conv_out(sample)
        return self.depatch(sample)


class _CausalDecoderAdapter(nn.Module):
    """Shards a causal 3D video decoder across ranks along one spatial axis.

    These decoders share a skeleton: a causal convolution in, a mid block, a run of up blocks, a
    normalisation, and a causal convolution out. Where that norm is RMS it needs no sharding,
    reducing over channels rather than over the axis being split; where it is a GroupNorm it
    does, and gets wrapped below. What else differs between the families is which classes fill
    the other slots, and how much of the temporal caching their forwards thread through.
    """

    _label = "Decoder"
    _conv_adapter = None
    _mid_adapter = None
    _up_block_adapters: Tuple[Tuple[Optional[type], type], ...] = ()
    # Wan and the families forked from it thread a temporal cache through every forward so a
    # decode can be split into chunks of frames. The HunyuanVideo decoders take a tensor and
    # nothing else.
    _takes_feature_cache = True
    # Of those that do, Wan alone also passes first_chunk, to tell the cache it is starting over.
    _takes_first_chunk = True
    _setup_type = CausalVAEAdapterSetup

    def __init__(
        self,
        decoder: nn.Module,
        vae_group: ProcessGroup = None,
        *,
        use_profiler: bool = False,
        verbose: bool = False,
        conv_block_size = 0,
        patch_dim: int = -2,
    ):
        super().__init__()
        _reject_benchmark_options(use_profiler, verbose)
        setup = self._setup_type.create(
            adapter=type(self).__name__,
            conv_adapter=self._conv_adapter,
            block_adapters=self._up_block_adapters,
            conv_block_size=conv_block_size,
            patch_dim=patch_dim,
            vae_group=vae_group,
        )
        self._setup = setup
        self.patch_dim = setup.patch_dim
        self.parallel_context = setup.parallel_context
        self.decoder = decoder
        self.decoder.conv_in = setup.adapt_convolution(decoder.conv_in)
        self.decoder.mid_block = self._mid_adapter(
            decoder.mid_block, conv_block_size=conv_block_size, **setup.options
        )
        self.decoder.up_blocks = setup.adapt_blocks(decoder.up_blocks, "up")
        self.decoder.conv_out = setup.adapt_convolution(decoder.conv_out)
        # HunyuanVideo ends on a GroupNorm, whose statistics span the axis being split. The RMS
        # norms the other families end on do not, and are left as they are.
        if hasattr(decoder, "conv_norm_out"):
            self.decoder.conv_norm_out = setup.adapt_group_norm(decoder.conv_norm_out)
        # Read after the whole stack is adapted, so it sees every convolution that will exchange.
        self.patchify, self.depatchify = setup.patchers(self.decoder)
        self.vae_group = vae_group

    def _run_decoder(self, sample, feat_cache, feat_idx, first_chunk):
        if not self._takes_feature_cache:
            return self.decoder(sample)
        feat_idx = cache_cursor(feat_idx)
        if self._takes_first_chunk:
            return self.decoder(
                sample, feat_cache=feat_cache, feat_idx=feat_idx, first_chunk=first_chunk
            )
        return self.decoder(sample, feat_cache=feat_cache, feat_idx=feat_idx)

    def _sharded_decode(self, sample: torch.FloatTensor, patchify: bool, run):
        """Split the sample across ranks, run the decoder on this rank's share, and reassemble

        Kept apart from forward because the families do not agree on what a decoder call looks
        like: some thread a temporal cache through it, LTX-2 a timestep embedding. Splitting and
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
        first_chunk: bool = False,
        patchify: bool = True,
    ):
        return self._sharded_decode(
            sample,
            patchify,
            lambda x: self._run_decoder(x, feat_cache, feat_idx, first_chunk),
        )


class WanDecoderAdapter(_CausalDecoderAdapter):
    _label = "WanDecoder"
    _conv_adapter = WanCausalConv3dAdapter
    _mid_adapter = WanMidBlockAdapter
    _up_block_adapters = (
        (WanUpBlock, WanUpBlockAdapter),
        (WanResidualUpBlock, WanResidualUpBlockAdapter),
    )


class QwenImageDecoderAdapter(_CausalDecoderAdapter):
    """Qwen-Image's decoder, which is Wan's without the first_chunk argument"""

    _label = "QwenImageDecoder"
    _conv_adapter = QwenImageCausalConv3dAdapter
    _mid_adapter = QwenImageMidBlockAdapter
    _up_block_adapters = ((QwenImageUpBlock, QwenImageUpBlockAdapter),)
    _takes_first_chunk = False


class HunyuanVideoDecoderAdapter(_CausalDecoderAdapter):
    _label = "HunyuanVideoDecoder"
    _conv_adapter = HunyuanVideoCausalConv3dAdapter
    _mid_adapter = HunyuanVideoMidBlockAdapter
    _up_block_adapters = ((HunyuanVideoUpBlock3D, HunyuanVideoUpBlockAdapter),)
    _takes_feature_cache = False


class HunyuanVideo15DecoderAdapter(_CausalDecoderAdapter):
    _label = "HunyuanVideo15Decoder"
    _conv_adapter = HunyuanVideo15CausalConv3dAdapter
    _mid_adapter = HunyuanVideo15MidBlockAdapter
    _up_block_adapters = ((HunyuanVideo15UpBlock3D, HunyuanVideo15UpBlockAdapter),)
    _takes_feature_cache = False


class LTX2VideoDecoderAdapter(_CausalDecoderAdapter):
    """LTX-2's decoder, which takes a timestep embedding where the others take a temporal cache

    The embedding arrives shaped to broadcast over space, so it needs no sharding of its own,
    and neither does the channel-to-space shuffle this decoder ends on.
    """

    _label = "LTX2VideoDecoder"
    _conv_adapter = LTX2VideoCausalConv3dAdapter
    _mid_adapter = LTX2VideoMidBlockAdapter
    _up_block_adapters = ((LTX2VideoUpBlock3d, LTX2VideoUpBlockAdapter),)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        temb: Optional[torch.FloatTensor] = None,
        causal: Optional[bool] = None,
        patchify: bool = True,
    ):
        return self._sharded_decode(
            hidden_states, patchify, lambda x: self.decoder(x, temb, causal)
        )
