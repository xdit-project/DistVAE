from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.utils import deprecate
from diffusers.models.upsampling import Upsample2D

from distvae.models.layers.normalization import RMSNorm
from distvae.modules.adapters.layers.conv_adapters import Conv2dAdapter


class PatchUpsample2D(Upsample2D):
    """A 2D upsampling layer with an optional convolution.

    Parameters:
        channels (`int`):
            number of channels in the inputs and outputs.
        use_conv (`bool`, default `False`):
            option to use a convolution.
        use_conv_transpose (`bool`, default `False`):
            option to use a convolution transpose.
        out_channels (`int`, optional):
            number of output channels. Defaults to `channels`.
        name (`str`, default `conv`):
            name of the upsampling 2D layer.
    """

    def __init__(
        self,
        channels: int,
        use_conv: bool = False,
        use_conv_transpose: bool = False,
        out_channels: Optional[int] = None,
        name: str = "conv",
        kernel_size: Optional[int] = None,
        padding=1,
        norm_type=None,
        eps=None,
        elementwise_affine=None,
        bias=True,
        interpolate=True,
        conv_block_size = 0,
    ):
        assert norm_type is None, "norm_type has not been supported for PatchUpsample2D yat."
        assert use_conv_transpose is False, "use_conv_transpose has not been supported for PatchUpsample2D yet."
        super().__init__(channels, use_conv, use_conv_transpose, out_channels, name, 
                         kernel_size, padding, norm_type, eps, elementwise_affine, 
                         bias, interpolate)
        if name == "conv":
            self.conv = Conv2dAdapter(self.conv, block_size=conv_block_size)
        else:
            self.Conv2d_0 = Conv2dAdapter(self.Conv2d_0, block_size=conv_block_size)