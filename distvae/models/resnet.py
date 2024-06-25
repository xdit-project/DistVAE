from typing import Optional

import torch
import torch.distributed
import torch.nn as nn

from diffusers.utils import deprecate 
from diffusers.models.resnet import ResnetBlock2D
from diffusers.models.activations import get_activation
from diffusers.models.normalization import AdaGroupNorm
from diffusers.models.attention_processor import SpatialNorm
from diffusers.models.downsampling import Downsample2D
from diffusers.models.upsampling import Upsample2D

from distvae.modules.adapters.layers.norm_adapters import GroupNormAdapter
from distvae.modules.adapters.layers.conv_adapters import Conv2dAdapter

# class ResnetBlockCondNorm2D(nn.Module):
#     r"""
#     A Resnet block that use normalization layer that incorporate conditioning information.

#     Parameters:
#         in_channels (`int`): The number of channels in the input.
#         out_channels (`int`, *optional*, default to be `None`):
#             The number of output channels for the first conv2d layer. If None, same as `in_channels`.
#         dropout (`float`, *optional*, defaults to `0.0`): The dropout probability to use.
#         temb_channels (`int`, *optional*, default to `512`): the number of channels in timestep embedding.
#         groups (`int`, *optional*, default to `32`): The number of groups to use for the first normalization layer.
#         groups_out (`int`, *optional*, default to None):
#             The number of groups to use for the second normalization layer. if set to None, same as `groups`.
#         eps (`float`, *optional*, defaults to `1e-6`): The epsilon to use for the normalization.
#         non_linearity (`str`, *optional*, default to `"swish"`): the activation function to use.
#         time_embedding_norm (`str`, *optional*, default to `"ada_group"` ):
#             The normalization layer for time embedding `temb`. Currently only support "ada_group" or "spatial".
#         kernel (`torch.FloatTensor`, optional, default to None): FIR filter, see
#             [`~models.resnet.FirUpsample2D`] and [`~models.resnet.FirDownsample2D`].
#         output_scale_factor (`float`, *optional*, default to be `1.0`): the scale factor to use for the output.
#         use_in_shortcut (`bool`, *optional*, default to `True`):
#             If `True`, add a 1x1 nn.conv2d layer for skip-connection.
#         up (`bool`, *optional*, default to `False`): If `True`, add an upsample layer.
#         down (`bool`, *optional*, default to `False`): If `True`, add a downsample layer.
#         conv_shortcut_bias (`bool`, *optional*, default to `True`):  If `True`, adds a learnable bias to the
#             `conv_shortcut` output.
#         conv_2d_out_channels (`int`, *optional*, default to `None`): the number of channels in the output.
#             If None, same as `out_channels`.
#     """

#     def __init__(
#         self,
#         *,
#         in_channels: int,
#         out_channels: Optional[int] = None,
#         conv_shortcut: bool = False,
#         dropout: float = 0.0,
#         temb_channels: int = 512,
#         groups: int = 32,
#         groups_out: Optional[int] = None,
#         eps: float = 1e-6,
#         non_linearity: str = "swish",
#         time_embedding_norm: str = "ada_group",  # ada_group, spatial
#         output_scale_factor: float = 1.0,
#         use_in_shortcut: Optional[bool] = None,
#         up: bool = False,
#         down: bool = False,
#         conv_shortcut_bias: bool = True,
#         conv_2d_out_channels: Optional[int] = None,
#     ):
#         super().__init__()
#         self.in_channels = in_channels
#         out_channels = in_channels if out_channels is None else out_channels
#         self.out_channels = out_channels
#         self.use_conv_shortcut = conv_shortcut
#         self.up = up
#         self.down = down
#         self.output_scale_factor = output_scale_factor
#         self.time_embedding_norm = time_embedding_norm

#         conv_cls = nn.Conv2d

#         if groups_out is None:
#             groups_out = groups

#         if self.time_embedding_norm == "ada_group":  # ada_group
#             self.norm1 = AdaGroupNorm(temb_channels, in_channels, groups, eps=eps)
#         elif self.time_embedding_norm == "spatial":
#             self.norm1 = SpatialNorm(in_channels, temb_channels)
#         else:
#             raise ValueError(f" unsupported time_embedding_norm: {self.time_embedding_norm}")

#         self.conv1 = conv_cls(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

#         if self.time_embedding_norm == "ada_group":  # ada_group
#             self.norm2 = AdaGroupNorm(temb_channels, out_channels, groups_out, eps=eps)
#         elif self.time_embedding_norm == "spatial":  # spatial
#             self.norm2 = SpatialNorm(out_channels, temb_channels)
#         else:
#             raise ValueError(f" unsupported time_embedding_norm: {self.time_embedding_norm}")

#         self.dropout = torch.nn.Dropout(dropout)

#         conv_2d_out_channels = conv_2d_out_channels or out_channels
#         self.conv2 = conv_cls(out_channels, conv_2d_out_channels, kernel_size=3, stride=1, padding=1)

#         self.nonlinearity = get_activation(non_linearity)

#         self.upsample = self.downsample = None
#         if self.up:
#             self.upsample = Upsample2D(in_channels, use_conv=False)
#         elif self.down:
#             self.downsample = Downsample2D(in_channels, use_conv=False, padding=1, name="op")

#         self.use_in_shortcut = self.in_channels != conv_2d_out_channels if use_in_shortcut is None else use_in_shortcut

#         self.conv_shortcut = None
#         if self.use_in_shortcut:
#             self.conv_shortcut = conv_cls(
#                 in_channels,
#                 conv_2d_out_channels,
#                 kernel_size=1,
#                 stride=1,
#                 padding=0,
#                 bias=conv_shortcut_bias,
#             )

#     def forward(self, input_tensor: torch.FloatTensor, temb: torch.FloatTensor, *args, **kwargs) -> torch.FloatTensor:
#         if len(args) > 0 or kwargs.get("scale", None) is not None:
#             deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
#             deprecate("scale", "1.0.0", deprecation_message)

#         hidden_states = input_tensor

#         hidden_states = self.norm1(hidden_states, temb)

#         hidden_states = self.nonlinearity(hidden_states)

#         if self.upsample is not None:
#             # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
#             if hidden_states.shape[0] >= 64:
#                 input_tensor = input_tensor.contiguous()
#                 hidden_states = hidden_states.contiguous()
#             input_tensor = self.upsample(input_tensor)
#             hidden_states = self.upsample(hidden_states)

#         elif self.downsample is not None:
#             input_tensor = self.downsample(input_tensor)
#             hidden_states = self.downsample(hidden_states)

#         hidden_states = self.conv1(hidden_states)

#         hidden_states = self.norm2(hidden_states, temb)

#         hidden_states = self.nonlinearity(hidden_states)

#         hidden_states = self.dropout(hidden_states)
#         hidden_states = self.conv2(hidden_states)

#         if self.conv_shortcut is not None:
#             input_tensor = self.conv_shortcut(input_tensor)

#         output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

#         return output_tensor


class PatchResnetBlock2D(nn.Module):
    r"""
    A Resnet block.

    Parameters:
        in_channels (`int`): The number of channels in the input.
        out_channels (`int`, *optional*, default to be `None`):
            The number of output channels for the first conv2d layer. If None, same as `in_channels`.
        dropout (`float`, *optional*, defaults to `0.0`): The dropout probability to use.
        temb_channels (`int`, *optional*, default to `512`): the number of channels in timestep embedding.
        groups (`int`, *optional*, default to `32`): The number of groups to use for the first normalization layer.
        groups_out (`int`, *optional*, default to None):
            The number of groups to use for the second normalization layer. if set to None, same as `groups`.
        eps (`float`, *optional*, defaults to `1e-6`): The epsilon to use for the normalization.
        non_linearity (`str`, *optional*, default to `"swish"`): the activation function to use.
        time_embedding_norm (`str`, *optional*, default to `"default"` ): Time scale shift config.
            By default, apply timestep embedding conditioning with a simple shift mechanism. Choose "scale_shift"
            for a stronger conditioning with scale and shift.
        kernel (`torch.FloatTensor`, optional, default to None): FIR filter, see
            [`~models.resnet.FirUpsample2D`] and [`~models.resnet.FirDownsample2D`].
        output_scale_factor (`float`, *optional*, default to be `1.0`): the scale factor to use for the output.
        use_in_shortcut (`bool`, *optional*, default to `True`):
            If `True`, add a 1x1 nn.conv2d layer for skip-connection.
        up (`bool`, *optional*, default to `False`): If `True`, add an upsample layer.
        down (`bool`, *optional*, default to `False`): If `True`, add a downsample layer.
        conv_shortcut_bias (`bool`, *optional*, default to `True`):  If `True`, adds a learnable bias to the
            `conv_shortcut` output.
        conv_2d_out_channels (`int`, *optional*, default to `None`): the number of channels in the output.
            If None, same as `out_channels`.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: Optional[int] = None,
        conv_shortcut: bool = False,
        dropout: float = 0.0,
        temb_channels: int = 512,
        groups: int = 32,
        groups_out: Optional[int] = None,
        pre_norm: bool = True,
        eps: float = 1e-6,
        non_linearity: str = "swish",
        skip_time_act: bool = False,
        time_embedding_norm: str = "default",  # default, scale_shift,
        kernel: Optional[torch.FloatTensor] = None,
        output_scale_factor: float = 1.0,
        use_in_shortcut: Optional[bool] = None,
        up: bool = False,
        down: bool = False,
        conv_shortcut_bias: bool = True,
        conv_2d_out_channels: Optional[int] = None,
        conv_block_size = 0,
    ):
        assert temb_channels is None, "temb_channels is not supported currently."
        assert up is False, "Upsampling is not supported currently."
        assert down is False, "Downsampling is not supported currently."

        super().__init__()
        if time_embedding_norm == "ada_group":
            raise ValueError(
                "This class cannot be used with `time_embedding_norm==ada_group`, please use `ResnetBlockCondNorm2D` instead",
            )
        if time_embedding_norm == "spatial":
            raise ValueError(
                "This class cannot be used with `time_embedding_norm==spatial`, please use `ResnetBlockCondNorm2D` instead",
            )

        self.pre_norm = True
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.up = up
        self.down = down
        self.output_scale_factor = output_scale_factor
        self.time_embedding_norm = time_embedding_norm
        self.skip_time_act = skip_time_act

        linear_cls = nn.Linear
        conv_cls = nn.Conv2d

        if groups_out is None:
            groups_out = groups

        self.norm1 = GroupNormAdapter(torch.nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True))

        self.conv1 = Conv2dAdapter(conv_cls(in_channels, out_channels, kernel_size=3, stride=1, padding=1), block_size=conv_block_size)

        #TODO: Add support for temb_channels
        assert temb_channels is None, "temb_channels is not supported currently."
        self.time_emb_proj = None
        # if temb_channels is not None:
        #     if self.time_embedding_norm == "default":
        #         self.time_emb_proj = linear_cls(temb_channels, out_channels)
        #     elif self.time_embedding_norm == "scale_shift":
        #         self.time_emb_proj = linear_cls(temb_channels, 2 * out_channels)
        #     else:
        #         raise ValueError(f"unknown time_embedding_norm : {self.time_embedding_norm} ")
        # else:
        #     self.time_emb_proj = None

        self.norm2 = GroupNormAdapter(torch.nn.GroupNorm(num_groups=groups_out, num_channels=out_channels, eps=eps, affine=True))

        self.dropout = torch.nn.Dropout(dropout)
        conv_2d_out_channels = conv_2d_out_channels or out_channels
        self.conv2 = Conv2dAdapter(conv_cls(out_channels, conv_2d_out_channels, kernel_size=3, stride=1, padding=1), block_size=conv_block_size)

        self.nonlinearity = get_activation(non_linearity)

        self.upsample = self.downsample = None

        #TODO: Add support for upsample and downsample
        assert self.up is False, "Upsampling is not supported currently."
        assert self.down is False, "Downsampling is not supported currently."
        # if self.up:
        #     if kernel == "fir":
        #         fir_kernel = (1, 3, 3, 1)
        #         self.upsample = lambda x: upsample_2d(x, kernel=fir_kernel)
        #     elif kernel == "sde_vp":
        #         self.upsample = partial(F.interpolate, scale_factor=2.0, mode="nearest")
        #     else:
        #         self.upsample = Upsample2D(in_channels, use_conv=False)
        # elif self.down:
        #     if kernel == "fir":
        #         fir_kernel = (1, 3, 3, 1)
        #         self.downsample = lambda x: downsample_2d(x, kernel=fir_kernel)
        #     elif kernel == "sde_vp":
        #         self.downsample = partial(F.avg_pool2d, kernel_size=2, stride=2)
        #     else:
        #         self.downsample = Downsample2D(in_channels, use_conv=False, padding=1, name="op")

        self.use_in_shortcut = self.in_channels != conv_2d_out_channels if use_in_shortcut is None else use_in_shortcut

        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = Conv2dAdapter(
                conv_cls(
                    in_channels,
                    conv_2d_out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=conv_shortcut_bias,
                ),
                block_size=conv_block_size
            )

    def forward(self, input_tensor: torch.FloatTensor, temb: torch.FloatTensor, *args, **kwargs) -> torch.FloatTensor:
        #TODO: Add support for temb
        assert temb is None, "temb is not supported currently."

        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        hidden_states = input_tensor

        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)

        if self.upsample is not None:
            # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
            if hidden_states.shape[0] >= 64:
                input_tensor = input_tensor.contiguous()
                hidden_states = hidden_states.contiguous()
            input_tensor = self.upsample(input_tensor)
            hidden_states = self.upsample(hidden_states)
        elif self.downsample is not None:
            input_tensor = self.downsample(input_tensor)
            hidden_states = self.downsample(hidden_states)

        hidden_states = self.conv1(hidden_states)

        if self.time_emb_proj is not None:
            if not self.skip_time_act:
                temb = self.nonlinearity(temb)
            temb = self.time_emb_proj(temb)[:, :, None, None]

        if self.time_embedding_norm == "default":
            if temb is not None:
                hidden_states = hidden_states + temb
            hidden_states = self.norm2(hidden_states)
        elif self.time_embedding_norm == "scale_shift":
            if temb is None:
                raise ValueError(
                    f" `temb` should not be None when `time_embedding_norm` is {self.time_embedding_norm}"
                )
            time_scale, time_shift = torch.chunk(temb, 2, dim=1)
            hidden_states = self.norm2(hidden_states)
            hidden_states = hidden_states * (1 + time_scale) + time_shift
        else:
            hidden_states = self.norm2(hidden_states)

        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

        return output_tensor