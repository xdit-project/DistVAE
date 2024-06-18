from typing import Optional
import torch
import torch.nn as nn

from patchvae.modules.adapters.resnet_adapters import ResnetBlock2DAdapter
from patchvae.modules.adapters.upsampling_adapters import Upsample2DAdapter

from patchvae.models.unets.unet_2d_blocks import PatchUpDecoderBlock2D
from diffusers.models.unets.unet_2d_blocks import UpDecoderBlock2D
from diffusers.models.resnet import ResnetBlock2D
from diffusers.models.upsampling import Upsample2D


class UpDecoderBlock2DAdapter(nn.Module):
    def __init__(self, up_block: UpDecoderBlock2D):
        super().__init__()
        self.up_block = PatchUpDecoderBlock2D(
            in_channels=32,
            out_channels=32,
        )
        self.up_block.resolution_idx = up_block.resolution_idx
        self.up_block.resnets = nn.ModuleList([
            ResnetBlock2DAdapter(resnet) for resnet in up_block.resnets if isinstance(resnet, ResnetBlock2D)
        ])
        if up_block.upsamplers is not None:
            self.up_block.upsamplers = nn.ModuleList([
                Upsample2DAdapter(upsampler) for upsampler in up_block.upsamplers if isinstance(upsampler, Upsample2D)
            ])
            assert len(self.up_block.upsamplers) == len(up_block.upsamplers), "Number of upsamplers in the adapter must match the number of upsamplers in the original block"

        assert len(self.up_block.resnets) == len(up_block.resnets), "Number of resnets in the adapter must match the number of resnets in the original block"

    def forward(self, hidden_states: torch.FloatTensor, temb: Optional[torch.FloatTensor] = None):
        return self.up_block(hidden_states, temb)