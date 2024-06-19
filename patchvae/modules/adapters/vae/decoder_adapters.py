from typing import Optional

import torch
import torch.nn as nn
from patchvae.models.vae import PatchDecoder
from patchvae.modules.adapters.unets.unet_2d_blocks_adapters import UpDecoderBlock2DAdapter
from patchvae.modules.adapters.layers.norm_adapters import GroupNormAdapter
from patchvae.modules.adapters.layers.conv_adapters import Conv2dAdapter

from diffusers.models.autoencoders.vae import Decoder
from diffusers.models.unets.unet_2d_blocks import UpDecoderBlock2D

class DecoderAdapter(nn.Module):
    def __init__(self, decoder: Decoder):
        super().__init__()
        assert isinstance(decoder.conv_norm_out, nn.GroupNorm), "DecoderAdapter dose not support normalization method except GroupNorm"
        for up_block in decoder.up_blocks:
            assert isinstance(up_block, UpDecoderBlock2D), "DecoderAdapter dose not support up block except UpDecoderBlock2D"
        self.decoder = PatchDecoder()
        self.decoder.layers_per_block = decoder.layers_per_block
        self.decoder.conv_in = decoder.conv_in
        self.decoder.mid_block = decoder.mid_block
        self.decoder.up_blocks = nn.ModuleList([
            UpDecoderBlock2DAdapter(up_block) for up_block in decoder.up_blocks
        ])
        self.decoder.conv_norm_out = GroupNormAdapter(decoder.conv_norm_out)
        self.decoder.conv_act = decoder.conv_act
        self.decoder.conv_out = Conv2dAdapter(decoder.conv_out)


    def forward(
        self,
        sample: torch.FloatTensor,
        latent_embeds: Optional[torch.FloatTensor] = None,
    ):
        return self.decoder(sample, latent_embeds)