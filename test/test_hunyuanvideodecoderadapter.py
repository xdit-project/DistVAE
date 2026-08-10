"""HunyuanVideoDecoderAdapter against the decoder it shards, over gloo on CPU.

Two things here are unlike the Wan-derived families. The causal convolutions pad by replication,
so a rank left alone would repeat its own edge rows rather than read its neighbour's, and the
norms are GroupNorms, whose statistics span the axis being split.

Run from repo root:
  pytest test/test_hunyuanvideodecoderadapter.py -v
"""

import argparse
import os
import sys

import pytest
import torch
import torch.distributed as dist

from distvae.modules.adapters.vae.decoder_adapters import HunyuanVideoDecoderAdapter

from distributed_harness import assert_matches_reference, init_gloo, run_distributed

diffusers = pytest.importorskip("diffusers")
if not hasattr(diffusers, "AutoencoderKLHunyuanVideo"):
    pytest.skip("installed diffusers has no AutoencoderKLHunyuanVideo", allow_module_level=True)

# Four channel stages exercise every decoder upsampling transition.
CONFIG = dict(
    block_out_channels=(8, 8, 16, 16),
    layers_per_block=1,
    latent_channels=4,
    norm_num_groups=8,
)
LATENT_CHANNELS = 4


def build_decoder(mid_block_add_attention=True):
    vae = diffusers.AutoencoderKLHunyuanVideo(
        **CONFIG, mid_block_add_attention=mid_block_add_attention
    )
    return vae.eval().decoder


def worker(
    rank, world_size, frames, height, width, add_attention, conv_block_size, seed, master_port
):
    init_gloo(rank, world_size, master_port)
    try:
        torch.manual_seed(seed)
        decoder = build_decoder(add_attention)
        # Taken before the adapter runs, which rebuilds the decoder in place.
        weights = decoder.state_dict()

        latents = torch.randn(1, LATENT_CHANNELS, frames, height, width)

        with torch.no_grad():
            expected = None
            if rank == 0:
                reference = build_decoder(add_attention)
                reference.load_state_dict(weights)
                expected = reference(latents)

            adapter = HunyuanVideoDecoderAdapter(
                decoder, vae_group=None, conv_block_size=conv_block_size
            ).eval()
            actual = adapter(latents)

        assert_matches_reference(rank, actual, expected, "HunyuanVideoDecoderAdapter")
    finally:
        dist.destroy_process_group()


@pytest.mark.gloo
def test_a_sharded_hunyuan_decode_matches_a_single_rank_one(master_port, seed=42):
    run_distributed(worker, 2, (1, 16, 16, True, 0, seed), master_port)


@pytest.mark.gloo
def test_a_mid_block_without_attention_shards_its_resnets(master_port, seed=42):
    # Without attention the mid block is sharded rather than gathered around, which is a
    # different path through the adapter and the only one that reaches its resnet adapters.
    run_distributed(worker, 2, (1, 16, 16, False, 0, seed), master_port)


@pytest.mark.gloo
def test_a_latent_taller_than_it_is_wide_still_decodes(master_port, seed=42):
    run_distributed(worker, 2, (1, 24, 16, True, 0, seed), master_port)


@pytest.mark.gloo
def test_more_than_one_frame_still_decodes(master_port, seed=42):
    # The frame axis is not the one being split, and its causal padding stays in the adapter
    # rather than moving into PatchConv3d, so this is what checks that split was made correctly.
    run_distributed(worker, 2, (5, 16, 16, True, 0, seed), master_port)


@pytest.mark.gloo
def test_the_chunked_convolution_path_decodes_the_same(master_port, seed=42):
    run_distributed(worker, 2, (1, 16, 16, True, 32, seed), master_port)


@pytest.mark.gloo
def test_latent_rows_that_do_not_divide_by_the_rank_count(master_port, seed=42):
    # Uneven bands must preserve all 16 rows without padding the decoder input.
    run_distributed(worker, 3, (1, 16, 16, True, 0, seed), master_port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HunyuanVideoDecoderAdapter GLOO tests")
    parser.add_argument("--world_size", type=int, default=None)
    args, remainder = parser.parse_known_args()
    pytest_args = [os.path.abspath(__file__), "-v"] + remainder
    if args.world_size is not None:
        pytest_args.extend(["-k", f"[{args.world_size}]"])
    sys.exit(pytest.main(pytest_args))
