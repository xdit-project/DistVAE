"""HunyuanVideo15EncoderAdapter against the encoder it shards, over gloo on CPU.

HunyuanVideo 1.5 pads by replication like HunyuanVideo does, but normalises with RMS rather than
GroupNorm, and downsamples by folding each pair of rows and columns into channels instead of by
striding. That fold reads one input position per output one, so what it needs is for a rank to
hold whole pairs of rows, which is what cutting bands in multiples of the ratio is for.

Run from repo root:
  pytest test/test_hunyuanvideo15encoderadapter.py -v
"""

import argparse
import os
import sys

import pytest
import torch
import torch.distributed as dist

from distvae.modules.adapters.vae.encoder_adapters import HunyuanVideo15EncoderAdapter

from distributed_harness import assert_matches_reference, init_gloo, run_distributed

diffusers = pytest.importorskip("diffusers")
if not hasattr(diffusers, "AutoencoderKLHunyuanVideo15"):
    pytest.skip(
        "installed diffusers has no AutoencoderKLHunyuanVideo15", allow_module_level=True
    )

# The tiny stand-in xDiT builds this class from, small enough to encode on CPU.
CONFIG = dict(
    block_out_channels=(8, 8, 16, 16, 16),
    layers_per_block=1,
    latent_channels=4,
)
# Four stages downsample, so the encoder narrows the rows by this much.
SCALE_FACTOR = 16
IN_CHANNELS = 3


def build_encoder():
    return diffusers.AutoencoderKLHunyuanVideo15(**CONFIG).eval().encoder


def worker(rank, world_size, frames, height, width, conv_block_size, seed, master_port):
    init_gloo(rank, world_size, master_port)
    try:
        torch.manual_seed(seed)
        encoder = build_encoder()
        # Taken before the adapter runs, which rebuilds the encoder in place.
        weights = encoder.state_dict()

        pixels = torch.randn(1, IN_CHANNELS, frames, height, width)

        with torch.no_grad():
            expected = None
            if rank == 0:
                reference = build_encoder()
                reference.load_state_dict(weights)
                expected = reference(pixels)

            adapter = HunyuanVideo15EncoderAdapter(
                encoder,
                vae_group=None,
                vae_scale_factor=SCALE_FACTOR,
                conv_block_size=conv_block_size,
            ).eval()
            actual = adapter(pixels)

        assert_matches_reference(rank, actual, expected, "HunyuanVideo15EncoderAdapter")
    finally:
        dist.destroy_process_group()


@pytest.mark.gloo
@pytest.mark.parametrize("world_size", [1, 2, 4])
def test_a_sharded_hunyuan15_encode_matches_a_single_rank_one(world_size, master_port, seed=42):
    run_distributed(worker, world_size, (5, 64, 64, 0, seed), master_port)


@pytest.mark.gloo
def test_an_image_taller_than_it_is_wide_still_encodes(master_port, seed=42):
    run_distributed(worker, 2, (5, 96, 64, 0, seed), master_port)


@pytest.mark.gloo
def test_rows_that_do_not_divide_by_the_rank_count_still_encode(master_port, seed=42):
    # 48 rows at a ratio of 16 is 3 bands over 2 ranks, so one rank takes two and the other one.
    run_distributed(worker, 2, (5, 48, 64, 0, seed), master_port)


@pytest.mark.gloo
def test_the_chunked_convolution_path_encodes_the_same(master_port, seed=42):
    run_distributed(worker, 2, (5, 64, 64, 32, seed), master_port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HunyuanVideo15EncoderAdapter GLOO tests")
    parser.add_argument("--world_size", type=int, default=None)
    args, remainder = parser.parse_known_args()
    pytest_args = [os.path.abspath(__file__), "-v"] + remainder
    if args.world_size is not None:
        pytest_args.extend(["-k", f"[{args.world_size}]"])
    sys.exit(pytest.main(pytest_args))
