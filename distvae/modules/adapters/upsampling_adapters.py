from typing import Optional

import torch
import torch.nn as nn
from distvae.models.upsampling import PatchUpsample2D
from distvae.modules.adapters.layers.conv_adapters import Conv2dAdapter

from diffusers.models.upsampling import Upsample2D

class Upsample2DAdapter(nn.Module):
    def __init__(
        self, 
        upsample2d: Upsample2D,
        *,
        conv_block_size = 0,
    ):
        super().__init__()
        assert upsample2d.norm is None, "upsample2dBlock2DAdapter dose not support normalization"
        if upsample2d.name == "conv":
            assert not isinstance(upsample2d.conv, nn.ConvTranspose2d), "upsample2dBlock2DAdapter dose not support transpose conv"
        else:
            assert not isinstance(upsample2d.Conv2d_0, nn.ConvTranspose2d), "upsample2dBlock2DAdapter dose not support transpose conv"
        self.upsample2d = PatchUpsample2D(
            channels=upsample2d.channels,
            use_conv=upsample2d.use_conv,
            use_conv_transpose=upsample2d.use_conv_transpose,
            out_channels=upsample2d.out_channels,
            name=upsample2d.name,
            kernel_size=None,
            padding=1,
            interpolate=upsample2d.interpolate
        )
        if upsample2d.name == "conv":
            self.upsample2d.conv = Conv2dAdapter(upsample2d.conv, block_size=conv_block_size)
        else:
            self.upsample2d.Conv2d_0 = Conv2dAdapter(upsample2d.Conv2d_0, block_size=conv_block_size)
        

    def forward(
        self, hidden_states: torch.FloatTensor, output_size: Optional[int] = None, *args, **kwargs
    ):
        return self.upsample2d(hidden_states, output_size, *args, **kwargs)