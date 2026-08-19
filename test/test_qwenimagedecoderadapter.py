"""QwenImageDecoderAdapter against the decoder it shards, over gloo on CPU.

Qwen-Image, Qwen-Image-Edit, and Krea-2 share this decoder structure.

Run from repo root:
  pytest test/test_qwenimagedecoderadapter.py -v
"""

import argparse
import os
import sys

import pytest
import torch
import torch.distributed as dist

from distvae.modules.adapters.vae.decoder_adapters import QwenImageDecoderAdapter

from distributed_harness import assert_matches_reference, init_gloo, run_distributed

diffusers = pytest.importorskip("diffusers")
if not hasattr(diffusers, "AutoencoderKLQwenImage"):
    pytest.skip("installed diffusers has no AutoencoderKLQwenImage", allow_module_level=True)

# Four channel stages exercise every decoder upsampling transition.
CONFIG = dict(base_dim=8, z_dim=4, dim_mult=[1, 2, 4, 4], num_res_blocks=1, attn_scales=[])
LATENT_CHANNELS = 4


def build_decoder():
    return diffusers.AutoencoderKLQwenImage(**CONFIG).eval().decoder


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
                expected = reference(latents, feat_cache=None, feat_idx=[0])

            adapter = QwenImageDecoderAdapter(
                decoder, vae_group=None, conv_block_size=conv_block_size
            ).eval()
            actual = adapter(latents)

        assert_matches_reference(rank, actual, expected, "QwenImageDecoderAdapter")
    finally:
        dist.destroy_process_group()


@pytest.mark.gloo
def test_sharded_qwen_decode_matches_unsharded_decode(master_port, seed=42):
    run_distributed(worker, 2, (1, 16, 16, 0, seed), master_port)


@pytest.mark.gloo
def test_a_latent_taller_than_it_is_wide_still_decodes(master_port, seed=42):
    # The patch dimension defaults to H, so a non-square latent is the case where getting the
    # split wrong shows up as a wrongly shaped output rather than as wrong values.
    run_distributed(worker, 2, (1, 24, 16, 0, seed), master_port)


@pytest.mark.gloo
def test_more_than_one_frame_still_decodes(master_port, seed=42):
    # The frame axis is not the one being split, but the causal padding along it is applied by
    # the adapter rather than by PatchConv3d, which is where that could go wrong.
    run_distributed(worker, 2, (3, 16, 16, 0, seed), master_port)


@pytest.mark.gloo
def test_latent_rows_that_do_not_divide_by_the_rank_count(master_port, seed=42):
    # Padding to an even split changes the decode because convolution and mid-block attention
    # propagate padded values into the rows that survive cropping.
    run_distributed(worker, 3, (1, 16, 16, 0, seed), master_port)


@pytest.mark.gloo
def test_the_chunked_convolution_path_decodes_the_same(master_port, seed=42):
    run_distributed(worker, 2, (1, 16, 16, 32, seed), master_port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QwenImageDecoderAdapter GLOO multi-rank tests")
    parser.add_argument("--world_size", type=int, default=None)
    args, remainder = parser.parse_known_args()
    pytest_args = [os.path.abspath(__file__), "-v"] + remainder
    if args.world_size is not None:
        pytest_args.extend(["-k", f"[{args.world_size}]"])
    sys.exit(pytest.main(pytest_args))
