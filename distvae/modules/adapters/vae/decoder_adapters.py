from typing import Optional
import time

import torch
import torch.nn as nn
from distvae.models.vae import PatchDecoder
from distvae.modules.adapters.unets.unet_2d_blocks_adapters import UpDecoderBlock2DAdapter
from distvae.modules.adapters.layers.norm_adapters import GroupNormAdapter
from distvae.modules.adapters.layers.conv_adapters import Conv2dAdapter

from diffusers.models.autoencoders.vae import Decoder
from diffusers.models.unets.unet_2d_blocks import UpDecoderBlock2D

from torch.profiler import profile, record_function, ProfilerActivity


class DecoderAdapter(nn.Module):
    def __init__(
        self, 
        decoder: Decoder, 
        *,
        use_profiler: bool = False,
        conv_block_size = 0,
    ):
        super().__init__()
        assert isinstance(decoder.conv_norm_out, nn.GroupNorm), "DecoderAdapter dose not support normalization method except GroupNorm"
        for up_block in decoder.up_blocks:
            assert isinstance(up_block, UpDecoderBlock2D), "DecoderAdapter dose not support up block except UpDecoderBlock2D"
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


    def forward(
        self,
        sample: torch.FloatTensor,
        latent_embeds: Optional[torch.FloatTensor] = None,
    ):
        rank = torch.distributed.get_rank()
        start_time = time.time()
        elapsed_time = 0
        if self.use_profiler:
            torch.cuda.memory._record_memory_history(enabled=None)
            with profile(
                activities=[
                    ProfilerActivity.CPU,
                    ProfilerActivity.CUDA
                ],
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    f"./profile/patch_vae_{rank}"
                ),
                profile_memory=True,
                with_stack=True,
                record_shapes=True,
            ) as prof:
                output = self.decoder(sample, latent_embeds)
            prof.export_memory_timeline(f"patch_vae_profiler_mem_{rank}.html")
        else:
            output =  self.decoder(sample, latent_embeds)

        end_time = time.time()
        elapsed_time = end_time - start_time
        peak_memory = torch.cuda.max_memory_allocated(device="cuda")

        if rank == 0:
            print(f"Patch vae: [elapsed_time: {elapsed_time:.2f} sec, peak_memory: {peak_memory/1e9} GB]")
        return output