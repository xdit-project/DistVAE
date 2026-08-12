"""HunyuanVideoEncoderAdapter against the encoder it shards, over gloo on CPU.

Two things here are unlike the Wan-derived families. The causal convolutions pad by replication,
so a rank left alone would repeat its own edge rows rather than read its neighbour's, and the
encoder ends on a GroupNorm, whose statistics span the axis being split.

Run from repo root:
  pytest test/test_hunyuanvideoencoderadapter.py -v
"""

import argparse
import os
import sys

import pytest
import torch
import torch.distributed as dist

from distvae.modules.adapters.vae.encoder_adapters import HunyuanVideoEncoderAdapter

from distributed_harness import assert_matches_reference, init_gloo, run_distributed

diffusers = pytest.importorskip("diffusers")
if not hasattr(diffusers, "AutoencoderKLHunyuanVideo"):
    pytest.skip("installed diffusers has no AutoencoderKLHunyuanVideo", allow_module_level=True)

# Four channel stages exercise every encoder downsampling transition.
CONFIG = dict(
    block_out_channels=(8, 8, 16, 16),
    layers_per_block=1,
    latent_channels=4,
    norm_num_groups=8,
)
# Three stages downsample, so the encoder narrows the rows by this much.
SCALE_FACTOR = 8
IN_CHANNELS = 3


def build_encoder(mid_block_add_attention=True):
    vae = diffusers.AutoencoderKLHunyuanVideo(
        **CONFIG, mid_block_add_attention=mid_block_add_attention
    )
    return vae.eval().encoder


def worker(
    rank, world_size, frames, height, width, add_attention, conv_block_size, seed, master_port
):
    init_gloo(rank, world_size, master_port)
    try:
        torch.manual_seed(seed)
        encoder = build_encoder(add_attention)
        # Taken before the adapter runs, which rebuilds the encoder in place.
        weights = encoder.state_dict()

        pixels = torch.randn(1, IN_CHANNELS, frames, height, width)

        with torch.no_grad():
            expected = None
            if rank == 0:
                reference = build_encoder(add_attention)
                reference.load_state_dict(weights)
                expected = reference(pixels)

            adapter = HunyuanVideoEncoderAdapter(
                encoder,
                vae_group=None,
                vae_scale_factor=SCALE_FACTOR,
                conv_block_size=conv_block_size,
            ).eval()
            actual = adapter(pixels)

        assert_matches_reference(rank, actual, expected, "HunyuanVideoEncoderAdapter")
    finally:
        dist.destroy_process_group()


@pytest.mark.gloo
def test_sharded_hunyuan_encode_matches_unsharded_encode(master_port, seed=42):
    run_distributed(worker, 2, (5, 64, 64, True, 0, seed), master_port)


@pytest.mark.gloo
def test_an_encoder_whose_mid_block_has_no_attention(master_port, seed=42):
    # Without attention the mid block is convolutions alone, so it stays sharded rather than
    # being gathered around, which is a different path through the mid block adapter.
    run_distributed(worker, 2, (5, 64, 64, False, 0, seed), master_port)


@pytest.mark.gloo
def test_an_image_taller_than_it_is_wide_still_encodes(master_port, seed=42):
    run_distributed(worker, 2, (5, 96, 64, True, 0, seed), master_port)


@pytest.mark.gloo
def test_rows_that_do_not_divide_by_the_rank_count_still_encode(master_port, seed=42):
    # 64 rows at a ratio of 8 is 8 bands over 3 ranks, so the bands come out 3, 3 and 2 long.
    run_distributed(worker, 3, (5, 64, 64, True, 0, seed), master_port)


@pytest.mark.gloo
def test_the_chunked_convolution_path_encodes_the_same(master_port, seed=42):
    run_distributed(worker, 2, (5, 64, 64, True, 32, seed), master_port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HunyuanVideoEncoderAdapter GLOO tests")
    parser.add_argument("--world_size", type=int, default=None)
    args, remainder = parser.parse_known_args()
    pytest_args = [os.path.abspath(__file__), "-v"] + remainder
    if args.world_size is not None:
        pytest_args.extend(["-k", f"[{args.world_size}]"])
    sys.exit(pytest.main(pytest_args))
