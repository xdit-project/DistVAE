import torch
import torch.nn as nn
from distvae.models.resnet import PatchResnetBlock2D
from distvae.modules.adapters.layers.conv_adapters import Conv2dAdapter
from distvae.modules.adapters.layers.norm_adapters import GroupNormAdapter

from diffusers.models.resnet import ResnetBlock2D

class ResnetBlock2DAdapter(nn.Module):
    def __init__(
        self, 
        resnet: ResnetBlock2D, 
        *, 
        conv_block_size = 0,
    ):
        super().__init__()
        assert resnet.time_emb_proj is None, "temb_channels is not supported in ResnetBlock2DAdapter currently"
        assert resnet.up is False, "up sample is not supported in ResnetBlock2DAdapter currently"
        assert resnet.down is False, "ResnetBlock2DAdapter dose not support down sample is not supported in  currently"
        self.resnet = PatchResnetBlock2D(
            in_channels=resnet.in_channels,
            out_channels=resnet.out_channels,
            conv_shortcut=resnet.use_conv_shortcut,
            dropout=0,
            temb_channels=None,
            groups=1,
            groups_out=None,
            pre_norm=resnet.pre_norm,
            skip_time_act=resnet.skip_time_act,
            time_embedding_norm=resnet.time_embedding_norm,
            output_scale_factor=resnet.output_scale_factor,
            use_in_shortcut=resnet.use_in_shortcut,
            up=resnet.up,
            down=resnet.down,
        )
        self.resnet.use_in_shortcut = resnet.use_in_shortcut
        self.resnet.conv1 = Conv2dAdapter(resnet.conv1, block_size=conv_block_size)
        self.resnet.norm1 = GroupNormAdapter(resnet.norm1)
        self.resnet.conv2 = Conv2dAdapter(resnet.conv2, block_size=conv_block_size)
        self.resnet.norm2 = GroupNormAdapter(resnet.norm2)
        self.resnet.dropout = resnet.dropout
        self.resnet.nonlinearity = resnet.nonlinearity
        self.resnet.conv_shortcut = Conv2dAdapter(resnet.conv_shortcut, block_size=conv_block_size) if resnet.conv_shortcut is not None else None
        

    def forward(self, x, temb: torch.FloatTensor = None, *args, **kwargs):
        return self.resnet(x, temb, *args, **kwargs)