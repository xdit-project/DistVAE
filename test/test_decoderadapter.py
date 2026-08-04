"""DecoderAdapter against the decoder it shards, over gloo on CPU.

The equivalent check exists in test_vae_decoder.py, but only as a torchrun script needing NCCL
and a GPU, so nothing exercised this adapter in a plain test run. It is the adapter every
AutoencoderKL model decodes through, xDiT's SD3 and Z-Image included.

Run from repo root:
  pytest test/test_decoderadapter.py -v
"""

import argparse
import os
import sys

import pytest
import torch
import torch.distributed as dist

from distvae.modules.adapters.vae.decoder_adapters import DecoderAdapter

from distributed_harness import assert_matches_reference, init_gloo, run_distributed

diffusers = pytest.importorskip("diffusers")

CONFIG = dict(
    block_out_channels=[8, 8, 16, 16],
    layers_per_block=1,
    latent_channels=4,
    norm_num_groups=8,
    sample_size=256,
    down_block_types=["DownEncoderBlock2D"] * 4,
    up_block_types=["UpDecoderBlock2D"] * 4,
)
LATENT_CHANNELS = 4


def build_decoder():
    return diffusers.AutoencoderKL(**CONFIG).eval().decoder


def worker(rank, world_size, height, width, conv_block_size, seed, master_port):
    init_gloo(rank, world_size, master_port)
    try:
        torch.manual_seed(seed)
        decoder = build_decoder()
        weights = decoder.state_dict()

        latents = torch.randn(1, LATENT_CHANNELS, height, width)

        with torch.no_grad():
            expected = None
            if rank == 0:
                reference = build_decoder()
                reference.load_state_dict(weights)
                expected = reference(latents)

            adapter = DecoderAdapter(
                decoder, vae_group=None, conv_block_size=conv_block_size
            ).eval()
            actual = adapter(latents)

        # The sharded GroupNorm sums its statistics across ranks in float32 before dividing, so
        # it lands a little away from a single-rank reduction over the same values.
        assert_matches_reference(rank, actual, expected, "DecoderAdapter", atol=1e-4)
    finally:
        dist.destroy_process_group()


@pytest.mark.gloo
@pytest.mark.parametrize("world_size", [1, 2, 4])
def test_a_sharded_decode_matches_a_single_rank_one(world_size, master_port, seed=42):
    run_distributed(worker, world_size, (16, 16, 0, seed), master_port)


@pytest.mark.gloo
def test_the_chunked_convolution_path_decodes_the_same(master_port, seed=42):
    # A conv_block_size under the feature map size sends PatchConv2d down its chunked path,
    # which splits and reassembles each convolution on top of the sharding.
    run_distributed(worker, 2, (16, 16, 32, seed), master_port)


@pytest.mark.gloo
def test_latent_rows_that_do_not_divide_by_the_rank_count(master_port, seed=42):
    # This adapter was never exposed to the pad-and-crop the causal ones used, because
    # PatchDecoder splits after its mid block rather than before. Pinned so it stays that way.
    run_distributed(worker, 3, (16, 16, 0, seed), master_port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DecoderAdapter GLOO multi-rank tests")
    parser.add_argument("--world_size", type=int, default=None)
    args, remainder = parser.parse_known_args()
    pytest_args = [os.path.abspath(__file__), "-v"] + remainder
    if args.world_size is not None:
        pytest_args.extend(["-k", f"[{args.world_size}]"])
    sys.exit(pytest.main(pytest_args))
