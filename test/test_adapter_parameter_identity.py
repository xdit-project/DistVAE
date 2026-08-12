import pytest
import torch
import torch.nn as nn
from diffusers.models.autoencoders.autoencoder_kl_wan import WanCausalConv3d
from diffusers.models.unets.unet_2d_blocks import UpDecoderBlock2D
from diffusers.models.upsampling import Upsample2D

from distvae.modules.adapters.downsampling_adapters import _zero_pad_strided_conv
from distvae.modules.adapters.layers.conv_adapters import (
    Conv2dAdapter,
    Conv3dAdapter,
    WanCausalConv3dAdapter,
)
from distvae.modules.adapters.unets.unet_2d_blocks_adapters import (
    UpDecoderBlock2DAdapter,
)
from distvae.modules.adapters.upsampling_adapters import Upsample2DAdapter
from distributed_harness import make_parallel_context


def _assert_reuses_parameters_and_gradients(
    original, replacement, optimizer, input_shape
):
    weight = original.weight
    bias = original.bias

    assert replacement.weight is weight
    assert replacement.bias is bias
    assert optimizer.param_groups[0]["params"][0] is replacement.weight

    replacement(torch.randn(input_shape)).sum().backward()

    assert weight.grad is not None
    if bias is not None:
        assert bias.grad is not None


@pytest.mark.parametrize("bias", [True, False])
def test_conv2d_adapter_reuses_original_parameters(bias):
    conv = nn.Conv2d(2, 3, 3, padding=1, bias=bias)
    optimizer = torch.optim.SGD(conv.parameters(), lr=0.1)
    adapted = Conv2dAdapter(
        conv, parallel_context=make_parallel_context()
    ).conv2d

    _assert_reuses_parameters_and_gradients(
        conv, adapted, optimizer, (1, 2, 5, 5)
    )


@pytest.mark.parametrize("bias", [True, False])
def test_conv3d_adapter_reuses_original_parameters(bias):
    conv = nn.Conv3d(2, 3, 3, padding=1, bias=bias)
    optimizer = torch.optim.SGD(conv.parameters(), lr=0.1)
    adapted = Conv3dAdapter(
        conv, parallel_context=make_parallel_context()
    ).conv3d

    _assert_reuses_parameters_and_gradients(
        conv, adapted, optimizer, (1, 2, 4, 5, 5)
    )


@pytest.mark.parametrize("bias", [True, False])
def test_wan_causal_conv3d_adapter_reuses_original_parameters(bias):
    conv = WanCausalConv3d(2, 3, 3, padding=1)
    if not bias:
        conv.register_parameter("bias", None)
    optimizer = torch.optim.SGD(conv.parameters(), lr=0.1)
    adapted = WanCausalConv3dAdapter(
        conv, parallel_context=make_parallel_context()
    ).conv3d

    _assert_reuses_parameters_and_gradients(
        conv, adapted, optimizer, (1, 2, 4, 5, 5)
    )


@pytest.mark.parametrize("bias", [True, False])
def test_zero_pad_strided_conv_reuses_original_parameters(bias):
    conv = nn.Conv2d(2, 3, 3, stride=2, padding=0, bias=bias)
    optimizer = torch.optim.SGD(conv.parameters(), lr=0.1)
    adapted = _zero_pad_strided_conv(
        conv, conv_block_size=0, parallel_context=make_parallel_context()
    )

    _assert_reuses_parameters_and_gradients(
        conv, adapted, optimizer, (1, 2, 6, 6)
    )


def test_upsample_adapter_wraps_the_original_module_in_place():
    upsample = Upsample2D(channels=2, use_conv=True)
    adapted = Upsample2DAdapter(
        upsample, parallel_context=make_parallel_context()
    )

    assert adapted.upsample2d is upsample


def test_up_decoder_adapter_wraps_the_original_block_in_place():
    up_block = UpDecoderBlock2D(
        in_channels=2,
        out_channels=2,
        num_layers=1,
        resnet_groups=1,
        add_upsample=True,
    )
    adapted = UpDecoderBlock2DAdapter(
        up_block, parallel_context=make_parallel_context()
    )

    assert adapted.up_block is up_block
