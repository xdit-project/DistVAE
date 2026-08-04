"""WanEncoderAdapter against the encoder it shards, over gloo on CPU.

Wan ships two encoder shapes: 2.2 groups each stage into a WanResidualDownBlock, 2.1 lays the
same residual blocks, attentions and resamples out flat. Both are covered here, since the
encoder class alone does not say which one a checkpoint carries.

Run from repo root:
  pytest test/test_wanencoderadapter.py -v
"""

import argparse
import os
import sys

import pytest
import torch
import torch.distributed as dist

from distvae.modules.adapters.vae.encoder_adapters import WanEncoderAdapter

from distributed_harness import assert_matches_reference, init_gloo, run_distributed

autoencoder_kl_wan = pytest.importorskip(
    "diffusers.models.autoencoders.autoencoder_kl_wan"
)

# Three spatial downsamples, so the encoder narrows by 8 the way the shipped ones do.
CONFIG = dict(
    in_channels=3,
    dim=32,
    z_dim=16,
    # The shipped ratios, kept because Wan 2.2's shortcut averages space into channels and
    # asserts the two divide; a last stage that widens would fail to build at all.
    dim_mult=[1, 2, 4, 4],
    num_res_blocks=1,
    attn_scales=[],
    dropout=0.0,
)
SCALE_FACTOR = 8
IN_CHANNELS = 3


def build_encoder(is_residual=False):
    # Wan 2.2's grouped down block averages frames in its shortcut whenever a stage downsamples
    # time, while the stage itself only does so through the feature cache. Called without one,
    # as here, the two disagree on the frame count and diffusers' own encoder raises. So the
    # grouped shape is exercised without temporal downsampling; splitting is spatial regardless.
    temporal = [False] * 4 if is_residual else [False, True, True, False]
    encoder = autoencoder_kl_wan.WanEncoder3d(
        **CONFIG, temperal_downsample=temporal, is_residual=is_residual
    )
    return encoder.eval()


def worker(
    rank, world_size, frames, height, width, is_residual, conv_block_size, seed, master_port
):
    init_gloo(rank, world_size, master_port)
    try:
        torch.manual_seed(seed)
        encoder = build_encoder(is_residual)
        # Taken before the adapter runs, which rebuilds the encoder in place.
        weights = encoder.state_dict()

        pixels = torch.randn(1, IN_CHANNELS, frames, height, width)

        with torch.no_grad():
            expected = None
            if rank == 0:
                reference = build_encoder(is_residual)
                reference.load_state_dict(weights)
                expected = reference(pixels)

            adapter = WanEncoderAdapter(
                encoder,
                vae_group=None,
                vae_scale_factor=SCALE_FACTOR,
                conv_block_size=conv_block_size,
            ).eval()
            actual = adapter(pixels)

        assert_matches_reference(rank, actual, expected, "WanEncoderAdapter")
    finally:
        dist.destroy_process_group()


@pytest.mark.gloo
@pytest.mark.parametrize("world_size", [1, 2, 4])
def test_a_sharded_wan_encode_matches_a_single_rank_one(world_size, master_port, seed=42):
    run_distributed(worker, world_size, (4, 64, 64, False, 0, seed), master_port)


@pytest.mark.gloo
@pytest.mark.parametrize("world_size", [1, 2])
def test_the_grouped_wan22_down_blocks_encode_the_same(world_size, master_port, seed=42):
    run_distributed(worker, world_size, (4, 64, 64, True, 0, seed), master_port)


@pytest.mark.gloo
def test_an_image_taller_than_it_is_wide_still_encodes(master_port, seed=42):
    run_distributed(worker, 2, (4, 96, 64, False, 0, seed), master_port)


@pytest.mark.gloo
def test_rows_that_do_not_divide_by_the_rank_count_still_encode(master_port, seed=42):
    # 80 rows over 3 ranks is where the padding Patchify adds and the crop that undoes it matter.
    run_distributed(worker, 3, (4, 80, 64, False, 0, seed), master_port)


@pytest.mark.gloo
def test_the_chunked_convolution_path_encodes_the_same(master_port, seed=42):
    run_distributed(worker, 2, (4, 64, 64, False, 32, seed), master_port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WanEncoderAdapter GLOO tests")
    parser.add_argument("--world_size", type=int, default=None)
    args, remainder = parser.parse_known_args()
    pytest_args = [os.path.abspath(__file__), "-v"] + remainder
    if args.world_size is not None:
        pytest_args.extend(["-k", f"[{args.world_size}]"])
    sys.exit(pytest.main(pytest_args))
