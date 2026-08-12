from typing import Optional
import torch
import torch.nn as nn

from distvae.modules.adapters.downsampling_adapters import Downsample2DAdapter
from distvae.modules.adapters.resnet_adapters import ResnetBlock2DAdapter
from distvae.modules.adapters.upsampling_adapters import Upsample2DAdapter

from diffusers.models.unets.unet_2d_blocks import DownEncoderBlock2D, UpDecoderBlock2D
from distvae.utils import ParallelContext


class UpDecoderBlock2DAdapter(nn.Module):
    def __init__(
        self, 
        up_block: UpDecoderBlock2D,
        *,
        conv_block_size = 0,
        parallel_context: ParallelContext = None,
    ):
        super().__init__()
        assert up_block is not None and isinstance(up_block, UpDecoderBlock2D), "up_block must be a UpDecoderBlock2D instance"
        self.up_block = up_block
        up_block.resnets = nn.ModuleList([
            ResnetBlock2DAdapter(
                resnet,
                conv_block_size=conv_block_size,
                parallel_context=parallel_context,
            ) for resnet in up_block.resnets
        ])
        if up_block.upsamplers is not None:
            up_block.upsamplers = nn.ModuleList([
                Upsample2DAdapter(
                    upsampler,
                    conv_block_size=conv_block_size,
                    parallel_context=parallel_context,
                ) for upsampler in up_block.upsamplers
            ])

    def forward(self, hidden_states: torch.FloatTensor, temb: Optional[torch.FloatTensor] = None):
        return self.up_block(hidden_states, temb)


class DownEncoderBlock2DAdapter(nn.Module):
    """Shards the 2D down block AutoencoderKL and Flux.2 encode with: its resnets and downsampler

    Unlike the up block, this block is wrapped in place rather than rebuilt. Its forward method
    runs both components in order without additional patch metadata.
    """

    def __init__(
        self,
        down_block: DownEncoderBlock2D,
        *,
        conv_block_size = 0,
        parallel_context: ParallelContext = None,
    ):
        super().__init__()
        assert isinstance(down_block, DownEncoderBlock2D), (
            "down_block must be a DownEncoderBlock2D instance"
        )
        self.down_block = down_block
        down_block.resnets = nn.ModuleList([
            ResnetBlock2DAdapter(
                resnet,
                conv_block_size=conv_block_size,
                parallel_context=parallel_context,
            )
            for resnet in down_block.resnets
        ])
        if down_block.downsamplers is not None:
            down_block.downsamplers = nn.ModuleList([
                Downsample2DAdapter(
                    downsampler,
                    conv_block_size=conv_block_size,
                    parallel_context=parallel_context,
                )
                for downsampler in down_block.downsamplers
            ])

    def forward(self, hidden_states: torch.FloatTensor, *args, **kwargs):
        return self.down_block(hidden_states, *args, **kwargs)