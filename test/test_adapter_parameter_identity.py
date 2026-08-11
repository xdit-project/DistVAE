import pytest
import torch
import torch.nn as nn
from diffusers.models.autoencoders.autoencoder_kl_wan import WanCausalConv3d

from distvae.modules.adapters.downsampling_adapters import _zero_pad_strided_conv
from distvae.modules.adapters.layers.conv_adapters import (
    Conv2dAdapter,
    Conv3dAdapter,
    WanCausalConv3dAdapter,
)
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
