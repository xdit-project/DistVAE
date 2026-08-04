import time
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.distributed import ProcessGroup
from torch.profiler import profile, ProfilerActivity
from diffusers.models.autoencoders.vae import Decoder
from diffusers.models.unets.unet_2d_blocks import UpDecoderBlock2D
from diffusers.models.autoencoders.autoencoder_kl_wan import (
    WanUpBlock,
    WanResidualUpBlock,
)

from distvae.models.vae import PatchDecoder
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
from distvae.utils import DistributedEnv

try:
    import torch_musa
except ModuleNotFoundError:
    pass

QwenImageUpBlock = block(QWEN_IMAGE, "QwenImageUpBlock")
HunyuanVideoUpBlock3D = block(HUNYUAN_VIDEO, "HunyuanVideoUpBlock3D")
HunyuanVideo15UpBlock3D = block(HUNYUAN_VIDEO_15, "HunyuanVideo15UpBlock3D")
LTX2VideoUpBlock3d = block(LTX2_VIDEO, "LTX2VideoUpBlock3d")


def _decode(run, label: str, *, use_profiler: bool, verbose: bool):
    """Run a decode, optionally under the torch profiler, and report what it cost"""
    rank = DistributedEnv.get_global_rank()
    device_type = DistributedEnv.get_device_type()
    start_time = time.time()
    if use_profiler:
        if device_type == "musa":
            torch.musa.memory._record_memory_history(enabled=None)
            activities = [ProfilerActivity.CPU, ProfilerActivity.MUSA]
        else:
            torch.cuda.memory._record_memory_history(enabled=None)
            activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]

        with profile(
            activities=activities,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                f"./profile/patch_vae_{rank}"
            ),
            profile_memory=True,
            with_stack=True,
            record_shapes=True,
        ) as prof:
            output = run()
        prof.export_memory_timeline(f"patch_vae_profiler_mem_{rank}.html")
    else:
        output = run()

    elapsed_time = time.time() - start_time
    peak_memory = DistributedEnv.get_peak_memory(device_type)

    if verbose and rank == 0:
        print(
            f"{label}: [elapsed_time: {elapsed_time:.2f} sec, "
            f"peak_memory: {peak_memory/1e9} GB]"
        )
    return output


class DecoderAdapter(nn.Module):
    def __init__(
        self, 
        decoder: Decoder, 
        vae_group: ProcessGroup = None,
        *,
        use_profiler: bool = False,
        verbose: bool = False,
        conv_block_size = 0,
    ):
        super().__init__()
        assert isinstance(decoder.conv_norm_out, nn.GroupNorm), "DecoderAdapter does not support normalization method except GroupNorm"
        for up_block in decoder.up_blocks:
            assert isinstance(up_block, UpDecoderBlock2D), "DecoderAdapter does not support up block except UpDecoderBlock2D"
        DistributedEnv.initialize(vae_group)
        self.decoder = PatchDecoder()
        self.decoder.layers_per_block = decoder.layers_per_block
        self.decoder.conv_in = decoder.conv_in
        self.decoder.mid_block = decoder.mid_block
        self.decoder.up_blocks = nn.ModuleList([
            UpDecoderBlock2DAdapter(up_block, conv_block_size=conv_block_size) for up_block in decoder.up_blocks
        ])
        self.decoder.conv_norm_out = GroupNormAdapter(decoder.conv_norm_out)
        self.decoder.conv_act = decoder.conv_act
        self.decoder.conv_out = Conv2dAdapter(decoder.conv_out, block_size=conv_block_size)
        self.use_profiler = use_profiler
        self.verbose = verbose
        self.vae_group = vae_group

    def forward(
        self,
        sample: torch.FloatTensor,
        latent_embeds: Optional[torch.FloatTensor] = None,
    ):
        return _decode(
            lambda: self.decoder(sample, latent_embeds),
            "Decoder",
            use_profiler=self.use_profiler,
            verbose=self.verbose,
        )


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

    def __init__(
        self,
        decoder: nn.Module,
        vae_group: ProcessGroup = None,
        *,
        use_uniform_patch: bool = True,
        use_profiler: bool = False,
        verbose: bool = False,
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
        options = dict(patch_dim=patch_dim, use_uniform_patch=use_uniform_patch)
        self.decoder = decoder
        self.decoder.conv_in = self._conv_adapter(
            decoder.conv_in, block_size=conv_block_size, **options
        )
        self.decoder.mid_block = self._mid_adapter(
            decoder.mid_block, conv_block_size=conv_block_size, **options
        )
        self.decoder.up_blocks = nn.ModuleList([
            self._adapt_up_block(up_block, adapter, conv_block_size, options)
            for up_block in decoder.up_blocks
        ])
        self.decoder.conv_out = self._conv_adapter(
            decoder.conv_out, block_size=conv_block_size, **options
        )
        # HunyuanVideo ends on a GroupNorm, whose statistics span the axis being split. The RMS
        # norms the other families end on do not, and are left as they are.
        if isinstance(getattr(decoder, "conv_norm_out", None), nn.GroupNorm):
            self.decoder.conv_norm_out = GroupNormAdapter(decoder.conv_norm_out)
        self.patchify = Patchify(patch_dim=patch_dim, use_uniform_patch=use_uniform_patch)
        self.depatchify = DePatchify(patch_dim=patch_dim, use_uniform_patch=use_uniform_patch)
        self.use_uniform_patch = use_uniform_patch
        self.use_profiler = use_profiler
        self.verbose = verbose
        self.vae_group = vae_group

    @classmethod
    def _adapt_up_block(cls, up_block, adapter, conv_block_size, options):
        for block_type, block_adapter in cls._up_block_adapters:
            if block_type is not None and isinstance(up_block, block_type):
                return block_adapter(up_block, conv_block_size=conv_block_size, **options)
        handled = ", ".join(t.__name__ for t, _ in cls._up_block_adapters if t is not None)
        raise TypeError(
            f"{adapter} cannot shard an up block of type {type(up_block).__name__}. "
            f"It handles {handled or 'no up block type the installed diffusers provides'}."
        )

    def _run_decoder(self, sample, feat_cache, feat_idx, first_chunk):
        if not self._takes_feature_cache:
            return self.decoder(sample)
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
        adapter = type(self).__name__
        if self.use_uniform_patch and not patchify:
            raise ValueError(
                f"{adapter} does not support use_uniform_patch for already patchified inputs."
            )

        if self.use_uniform_patch:
            patch_dim = self.patch_dim if self.patch_dim >= 0 else sample.ndim + self.patch_dim
            patch_dim_size = sample.shape[patch_dim]

        if patchify:
            sample = self.patchify(sample)
        output = self.depatchify(run(sample))

        if self.use_uniform_patch:
            group_world_size = DistributedEnv.get_group_world_size()
            upsampling_factor = output.shape[patch_dim] // (sample.shape[patch_dim] * group_world_size)
            output = output.narrow(patch_dim, 0, patch_dim_size * upsampling_factor)

        return output

    def forward(
        self,
        sample: torch.FloatTensor,
        feat_cache: Optional[torch.FloatTensor] = None,
        feat_idx: Optional[int] = 0,
        first_chunk: bool = False,
        patchify: bool = True,
    ):
        return _decode(
            lambda: self._sharded_decode(
                sample,
                patchify,
                lambda x: self._run_decoder(x, feat_cache, feat_idx, first_chunk),
            ),
            self._label,
            use_profiler=self.use_profiler,
            verbose=self.verbose,
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
        return _decode(
            lambda: self._sharded_decode(
                hidden_states, patchify, lambda x: self.decoder(x, temb, causal)
            ),
            self._label,
            use_profiler=self.use_profiler,
            verbose=self.verbose,
        )
