import torch.nn as nn

from diffusers.models.resnet import ResnetBlock2D

from distvae.modules.adapters.layers.conv_adapters import Conv2dAdapter
from distvae.modules.adapters.layers.norm_adapters import GroupNormAdapter
from distvae.modules.adapters.resnet_adapters import ResnetBlock2DAdapter
from distvae.utils import ParallelContext
import distvae.models.resnet as patch_resnet


def test_patch_resnet_constructor_receives_the_adapters_parallel_settings(monkeypatch):
    context = ParallelContext(group=None, rank=0, world_size=1, patch_dim=-1)
    received_norms = []
    received_convs = []

    def recording_group_norm(norm, patch_dim=None, parallel_context=None):
        received_norms.append((patch_dim, parallel_context))
        return GroupNormAdapter(
            norm,
            patch_dim=-2 if patch_dim is None else patch_dim,
            parallel_context=parallel_context,
        )

    def recording_conv(conv, *, block_size=0, patch_dim=None, parallel_context=None):
        received_convs.append((patch_dim, parallel_context))
        return Conv2dAdapter(
            conv,
            block_size=block_size,
            patch_dim=-2 if patch_dim is None else patch_dim,
            parallel_context=parallel_context,
        )

    monkeypatch.setattr(patch_resnet, "GroupNormAdapter", recording_group_norm)
    monkeypatch.setattr(patch_resnet, "Conv2dAdapter", recording_conv)
    source = ResnetBlock2D(
        in_channels=4,
        out_channels=8,
        temb_channels=None,
        groups=1,
        dropout=0.0,
    )

    ResnetBlock2DAdapter(source, patch_dim=3, parallel_context=context)

    assert received_norms == [(3, context), (3, context)]
    assert received_convs == [(3, context), (3, context), (3, context)]
