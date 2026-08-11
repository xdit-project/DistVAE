import pytest

from distvae.models.unets.unet_2d_blocks import (
    PatchUpDecoderBlock2D,
    get_up_block,
)


def test_patch_up_decoder_block_requires_a_parallel_context():
    with pytest.raises(TypeError, match="parallel_context must be provided"):
        PatchUpDecoderBlock2D(
            in_channels=8,
            out_channels=8,
            num_layers=1,
            resnet_groups=8,
        )


def test_up_decoder_block_factory_rejects_a_missing_parallel_context():
    with pytest.raises(TypeError, match="parallel_context must be provided"):
        get_up_block(
            "UpDecoderBlock2D",
            num_layers=1,
            in_channels=8,
            out_channels=8,
            prev_output_channel=8,
            temb_channels=None,
            add_upsample=False,
            resnet_eps=1e-6,
            resnet_act_fn="swish",
            resnet_groups=8,
        )
