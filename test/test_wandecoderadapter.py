"""WanDecoderAdapter against the decoder it shards, over gloo on CPU.

Run from repo root:
  pytest test/test_wandecoderadapter.py -v
"""

import argparse
import os
import sys

import pytest
import torch
import torch.distributed as dist

from distvae.modules.adapters.vae.decoder_adapters import WanDecoderAdapter

from distributed_harness import assert_matches_reference, init_gloo, run_distributed

diffusers = pytest.importorskip("diffusers")

# The tiny stand-in xDiT builds this class from, small enough to decode on CPU.
CONFIG = dict(base_dim=8, z_dim=4, dim_mult=[1, 2, 4, 4], num_res_blocks=1)
LATENT_CHANNELS = 4


def build_decoder():
    return diffusers.AutoencoderKLWan(**CONFIG).eval().decoder


def worker(rank, world_size, frames, height, width, seed, master_port):
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
                expected = reference(latents, feat_cache=None, feat_idx=[0])

            adapter = WanDecoderAdapter(decoder, vae_group=None).eval()
            actual = adapter(latents)

        assert_matches_reference(rank, actual, expected, "WanDecoderAdapter")
    finally:
        dist.destroy_process_group()


def cached_worker(rank, world_size, seed, master_port):
    init_gloo(rank, world_size, master_port)
    try:
        torch.manual_seed(seed)
        adapter = WanDecoderAdapter(build_decoder(), vae_group=None).eval()
        latents = torch.randn(1, LATENT_CHANNELS, 1, 16, 16)

        with torch.no_grad():
            adapter(latents, feat_cache=[None] * 1000)
    finally:
        dist.destroy_process_group()


@pytest.mark.gloo
@pytest.mark.parametrize("world_size", [1, 2, 4])
def test_a_sharded_wan_decode_matches_a_single_rank_one(world_size, master_port, seed=42):
    run_distributed(worker, world_size, (1, 16, 16, seed), master_port)


@pytest.mark.gloo
def test_cached_decode_gets_a_fresh_cursor_when_one_is_omitted(master_port, seed=42):
    run_distributed(cached_worker, 1, (seed,), master_port)


@pytest.mark.gloo
def test_a_latent_taller_than_it_is_wide_still_decodes(master_port, seed=42):
    # The patch dimension defaults to H, so a non-square latent is the case where getting the
    # split wrong shows up as a wrongly shaped output rather than as wrong values.
    run_distributed(worker, 2, (1, 24, 16, seed), master_port)


@pytest.mark.gloo
def test_latent_rows_that_do_not_divide_by_the_rank_count(master_port, seed=42):
    # 16 rows over 3 ranks. This used to pad the latent up to a size that did divide and crop
    # the decode afterwards, which is not the same computation: the pad stops being zeros at the
    # first convolution and reaches every kept pixel through the mid block's attention.
    run_distributed(worker, 3, (1, 16, 16, seed), master_port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WanDecoderAdapter GLOO multi-rank tests")
    parser.add_argument("--world_size", type=int, default=None)
    args, remainder = parser.parse_known_args()
    pytest_args = [os.path.abspath(__file__), "-v"] + remainder
    if args.world_size is not None:
        pytest_args.extend(["-k", f"[{args.world_size}]"])
    sys.exit(pytest.main(pytest_args))
