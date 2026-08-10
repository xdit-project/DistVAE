"""LTX2VideoEncoderAdapter against the encoder it shards, over gloo on CPU.

LTX-2 is the easiest of these to shard: its mid block has no attention, so nothing needs the
whole image, and its spatial padding already lives inside the convolution rather than being
applied around it. Its down block is the one place a checkpoint has a real choice, holding either
a space-to-channel downsampler or a plain strided convolution, and both are covered here.

Run from repo root:
  pytest test/test_ltx2videoencoderadapter.py -v
"""

import argparse
import os
import sys

import pytest
import torch
import torch.distributed as dist

from distvae.modules.adapters.vae.encoder_adapters import LTX2VideoEncoderAdapter

from distributed_harness import assert_matches_reference, init_gloo, run_distributed

diffusers = pytest.importorskip("diffusers")
if not hasattr(diffusers, "AutoencoderKLLTX2Video"):
    pytest.skip("installed diffusers has no AutoencoderKLLTX2Video", allow_module_level=True)

# Four channel stages exercise every downsampling mode. The compression ratio matches their
# space-to-channel factors.
CONFIG = dict(
    block_out_channels=(8, 16, 32, 32),
    latent_channels=8,
    layers_per_block=(1, 1, 1, 1, 1),
    spatial_compression_ratio=32,
)
SCALE_FACTOR = 32
IN_CHANNELS = 3
# The shipped stages, every one of which downsamples by folding space into channels. Only "conv"
# leaves a bare strided convolution behind, and only a stage that does not widen can hold one.
FOLDING = ("spatial", "temporal", "spatiotemporal", "spatiotemporal")
STRIDING = ("spatial", "temporal", "spatiotemporal", "conv")


def build_encoder(downsample_type=FOLDING, padding_mode="reflect"):
    vae = diffusers.AutoencoderKLLTX2Video(
        **CONFIG,
        downsample_type=downsample_type,
        encoder_spatial_padding_mode=padding_mode,
    )
    return vae.eval().encoder


def worker(
    rank, world_size, frames, height, width, downsample_type, padding_mode, conv_block_size,
    seed, master_port,
):
    init_gloo(rank, world_size, master_port)
    try:
        torch.manual_seed(seed)
        encoder = build_encoder(downsample_type, padding_mode)
        # Taken before the adapter runs, which rebuilds the encoder in place.
        weights = encoder.state_dict()

        pixels = torch.randn(1, IN_CHANNELS, frames, height, width)

        with torch.no_grad():
            expected = None
            if rank == 0:
                reference = build_encoder(downsample_type, padding_mode)
                reference.load_state_dict(weights)
                expected = reference(pixels)

            adapter = LTX2VideoEncoderAdapter(
                encoder,
                vae_group=None,
                vae_scale_factor=SCALE_FACTOR,
                conv_block_size=conv_block_size,
            ).eval()
            actual = adapter(pixels)

        assert_matches_reference(rank, actual, expected, "LTX2VideoEncoderAdapter")
    finally:
        dist.destroy_process_group()


@pytest.mark.gloo
def test_a_sharded_ltx2_encode_matches_a_single_rank_one(master_port, seed=42):
    run_distributed(worker, 2, (9, 64, 64, FOLDING, "reflect", 0, seed), master_port)


@pytest.mark.gloo
def test_a_stage_that_downsamples_by_striding_instead_of_folding(master_port, seed=42):
    # A bare strided convolution rather than the space-to-channel downsampler, which reaches the
    # sharded convolution by a different route through the down block adapter.
    run_distributed(worker, 2, (9, 64, 64, STRIDING, "reflect", 0, seed), master_port)


@pytest.mark.gloo
def test_the_zero_padding_ltx23_uses_encodes_the_same(master_port, seed=42):
    run_distributed(worker, 2, (9, 64, 64, FOLDING, "zeros", 0, seed), master_port)


@pytest.mark.gloo
def test_rows_that_do_not_divide_by_the_rank_count_still_encode(master_port, seed=42):
    # 96 rows at a ratio of 32 is 3 bands over 2 ranks, so one takes two and the other one. The
    # image is also taller than it is wide, which is where a mis-split shows up as a wrong shape.
    run_distributed(worker, 2, (9, 96, 64, FOLDING, "reflect", 0, seed), master_port)


@pytest.mark.gloo
def test_the_chunked_convolution_path_encodes_the_same(master_port, seed=42):
    run_distributed(worker, 2, (9, 64, 64, FOLDING, "reflect", 32, seed), master_port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LTX2VideoEncoderAdapter GLOO tests")
    parser.add_argument("--world_size", type=int, default=None)
    args, remainder = parser.parse_known_args()
    pytest_args = [os.path.abspath(__file__), "-v"] + remainder
    if args.world_size is not None:
        pytest_args.extend(["-k", f"[{args.world_size}]"])
    sys.exit(pytest.main(pytest_args))
