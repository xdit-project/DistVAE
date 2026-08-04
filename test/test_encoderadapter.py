"""EncoderAdapter against the encoder it shards, over gloo on CPU.

This is the encoder every AutoencoderKL model encodes through, and Flux.2's as well, which builds
the same one. It is what an image-to-image or inpainting pipeline runs over a full-sized image,
so it is the encode worth splitting.

Run from repo root:
  pytest test/test_encoderadapter.py -v
"""

import argparse
import os
import sys

import pytest
import torch
import torch.distributed as dist

from distvae.modules.adapters.vae.encoder_adapters import EncoderAdapter

from distributed_harness import assert_matches_reference, init_gloo, run_distributed

diffusers = pytest.importorskip("diffusers")

# Four stages, three of which downsample, so the encoder narrows by 8 as the shipped ones do.
CONFIG = dict(
    block_out_channels=[8, 8, 16, 16],
    layers_per_block=1,
    latent_channels=4,
    norm_num_groups=8,
    sample_size=256,
    down_block_types=["DownEncoderBlock2D"] * 4,
    up_block_types=["UpDecoderBlock2D"] * 4,
)
SCALE_FACTOR = 8
IN_CHANNELS = 3


def build_encoder(mid_block_add_attention=True):
    vae = diffusers.AutoencoderKL(
        **CONFIG, mid_block_add_attention=mid_block_add_attention
    )
    return vae.eval().encoder


def worker(
    rank, world_size, height, width, add_attention, conv_block_size, seed, master_port
):
    init_gloo(rank, world_size, master_port)
    try:
        torch.manual_seed(seed)
        encoder = build_encoder(add_attention)
        # Taken before the adapter runs, which rebuilds the encoder in place.
        weights = encoder.state_dict()

        pixels = torch.randn(1, IN_CHANNELS, height, width)

        with torch.no_grad():
            expected = None
            if rank == 0:
                reference = build_encoder(add_attention)
                reference.load_state_dict(weights)
                expected = reference(pixels)

            adapter = EncoderAdapter(
                encoder,
                vae_group=None,
                vae_scale_factor=SCALE_FACTOR,
                conv_block_size=conv_block_size,
            ).eval()
            actual = adapter(pixels)

        # The sharded GroupNorms inside the down blocks sum their statistics across ranks in
        # float32 before dividing, which lands a little away from one rank reducing the same
        # values in one pass.
        assert_matches_reference(rank, actual, expected, "EncoderAdapter", atol=1e-4)
    finally:
        dist.destroy_process_group()


@pytest.mark.gloo
@pytest.mark.parametrize("world_size", [1, 2, 4])
def test_a_sharded_encode_matches_a_single_rank_one(world_size, master_port, seed=42):
    run_distributed(worker, world_size, (64, 64, True, 0, seed), master_port)


@pytest.mark.gloo
@pytest.mark.parametrize("world_size", [1, 2])
def test_a_mid_block_without_attention_encodes_the_same(world_size, master_port, seed=42):
    # Flux.2 can be configured either way, and the mid block runs whole on every rank regardless,
    # so this checks the split is undone before it either way.
    run_distributed(worker, world_size, (64, 64, False, 0, seed), master_port)


@pytest.mark.gloo
def test_an_image_taller_than_it_is_wide_still_encodes(master_port, seed=42):
    run_distributed(worker, 2, (96, 64, True, 0, seed), master_port)


@pytest.mark.gloo
def test_rows_that_do_not_divide_by_the_rank_count_still_encode(master_port, seed=42):
    # 80 rows at a ratio of 8 is ten bands, which over 3 ranks leaves them different sizes: the
    # downsamplers have to read their neighbours' sizes rather than assume they match.
    run_distributed(worker, 3, (80, 64, True, 0, seed), master_port)


@pytest.mark.gloo
def test_the_chunked_convolution_path_encodes_the_same(master_port, seed=42):
    # A conv_block_size under the feature map size sends PatchConv2d down its chunked path, which
    # splits and reassembles each convolution on top of the sharding.
    run_distributed(worker, 2, (64, 64, True, 32, seed), master_port)


def test_a_ratio_that_the_down_blocks_do_not_agree_with_is_refused():
    # Sizing bands by a ratio the stages do not actually narrow by would leave a later stage
    # halving a band into a row belonging to the next rank, quietly, so it is counted and checked.
    encoder = build_encoder()
    with pytest.raises(ValueError, match="narrow by 8"):
        EncoderAdapter(encoder, vae_group=None, vae_scale_factor=16)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EncoderAdapter GLOO multi-rank tests")
    parser.add_argument("--world_size", type=int, default=None)
    args, remainder = parser.parse_known_args()
    pytest_args = [os.path.abspath(__file__), "-v"] + remainder
    if args.world_size is not None:
        pytest_args.extend(["-k", f"[{args.world_size}]"])
    sys.exit(pytest.main(pytest_args))
