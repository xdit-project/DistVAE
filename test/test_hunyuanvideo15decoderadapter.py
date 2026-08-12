"""HunyuanVideo15DecoderAdapter against the decoder it shards, over gloo on CPU.

HunyuanVideo 1.5 pads by replication like HunyuanVideo does, but normalises with RMS rather than
GroupNorm, and its attention block takes a whole 5D tensor and builds its own mask, so unlike
HunyuanVideo's it can be wrapped on its own and the mid block around it stays sharded.

Run from repo root:
  pytest test/test_hunyuanvideo15decoderadapter.py -v
"""

import argparse
import os
import sys

import pytest
import torch
import torch.distributed as dist

from distvae.modules.adapters.vae.decoder_adapters import HunyuanVideo15DecoderAdapter

from distributed_harness import assert_matches_reference, init_gloo, run_distributed

diffusers = pytest.importorskip("diffusers")
if not hasattr(diffusers, "AutoencoderKLHunyuanVideo15"):
    pytest.skip(
        "installed diffusers has no AutoencoderKLHunyuanVideo15", allow_module_level=True
    )

# Five channel stages exercise every decoder upsampling transition.
CONFIG = dict(
    block_out_channels=(8, 8, 16, 16, 16),
    layers_per_block=1,
    latent_channels=4,
)
LATENT_CHANNELS = 4


def build_decoder():
    return diffusers.AutoencoderKLHunyuanVideo15(**CONFIG).eval().decoder


def worker(rank, world_size, frames, height, width, conv_block_size, seed, master_port):
    init_gloo(rank, world_size, master_port)
    try:
        torch.manual_seed(seed)
        decoder = build_decoder()
        # Taken before the adapter runs, which rebuilds the decoder in place.
        weights = decoder.state_dict()

        latents = torch.randn(1, LATENT_CHANNELS, frames, height, width)

        with torch.no_grad():
            expected = None
            if rank == 0:
                reference = build_decoder()
                reference.load_state_dict(weights)
                expected = reference(latents)

            adapter = HunyuanVideo15DecoderAdapter(
                decoder, vae_group=None, conv_block_size=conv_block_size
            ).eval()
            actual = adapter(latents)

        assert_matches_reference(rank, actual, expected, "HunyuanVideo15DecoderAdapter")
    finally:
        dist.destroy_process_group()


@pytest.mark.gloo
def test_sharded_hunyuan15_decode_matches_unsharded_decode(master_port, seed=42):
    run_distributed(worker, 2, (1, 16, 16, 0, seed), master_port)


@pytest.mark.gloo
def test_a_latent_taller_than_it_is_wide_still_decodes(master_port, seed=42):
    run_distributed(worker, 2, (1, 24, 16, 0, seed), master_port)


@pytest.mark.gloo
def test_more_than_one_frame_still_decodes(master_port, seed=42):
    # Its upsampler treats the first frame differently from the rest, so a single frame would
    # never reach the branch that shuffles channels into time as well as space.
    run_distributed(worker, 2, (5, 16, 16, 0, seed), master_port)


@pytest.mark.gloo
def test_the_chunked_convolution_path_decodes_the_same(master_port, seed=42):
    run_distributed(worker, 2, (1, 16, 16, 32, seed), master_port)


@pytest.mark.gloo
def test_latent_rows_that_do_not_divide_by_the_rank_count(master_port, seed=42):
    # Uneven bands must preserve all 16 rows without padding the decoder input.
    run_distributed(worker, 3, (1, 16, 16, 0, seed), master_port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HunyuanVideo15DecoderAdapter GLOO tests")
    parser.add_argument("--world_size", type=int, default=None)
    args, remainder = parser.parse_known_args()
    pytest_args = [os.path.abspath(__file__), "-v"] + remainder
    if args.world_size is not None:
        pytest_args.extend(["-k", f"[{args.world_size}]"])
    sys.exit(pytest.main(pytest_args))
