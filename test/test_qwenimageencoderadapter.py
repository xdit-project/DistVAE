"""QwenImageEncoderAdapter against the encoder it shards, over gloo on CPU.

Qwen-Image's encoder is Wan 2.1's laid out flat and renamed, so what this really checks is that
the resample's zero pad and stride-2 convolution survive being split, and that an attention
block in the middle of the down blocks gets gathered rather than left on its own patch.

Run from repo root:
  pytest test/test_qwenimageencoderadapter.py -v
"""

import argparse
import os
import sys

import pytest
import torch
import torch.distributed as dist

from distvae.modules.adapters.vae.encoder_adapters import QwenImageEncoderAdapter

from distributed_harness import assert_matches_reference, init_gloo, run_distributed

autoencoder_kl_qwenimage = pytest.importorskip(
    "diffusers.models.autoencoders.autoencoder_kl_qwenimage"
)

CONFIG = dict(
    dim=32,
    z_dim=16,
    dim_mult=[1, 2, 4, 8],
    num_res_blocks=1,
    temperal_downsample=[False, True, True],
    dropout=0.0,
)
SCALE_FACTOR = 8
IN_CHANNELS = 3


def build_encoder(attn_scales=()):
    encoder = autoencoder_kl_qwenimage.QwenImageEncoder3d(
        **CONFIG, attn_scales=list(attn_scales)
    )
    return encoder.eval()


def worker(
    rank, world_size, frames, height, width, attn_scales, conv_block_size, seed, master_port
):
    init_gloo(rank, world_size, master_port)
    try:
        torch.manual_seed(seed)
        encoder = build_encoder(attn_scales)
        # Taken before the adapter runs, which rebuilds the encoder in place.
        weights = encoder.state_dict()

        pixels = torch.randn(1, IN_CHANNELS, frames, height, width)

        with torch.no_grad():
            expected = None
            if rank == 0:
                reference = build_encoder(attn_scales)
                reference.load_state_dict(weights)
                expected = reference(pixels)

            adapter = QwenImageEncoderAdapter(
                encoder,
                vae_group=None,
                vae_scale_factor=SCALE_FACTOR,
                conv_block_size=conv_block_size,
            ).eval()
            actual = adapter(pixels)

        assert_matches_reference(rank, actual, expected, "QwenImageEncoderAdapter")
    finally:
        dist.destroy_process_group()


@pytest.mark.gloo
def test_sharded_qwen_encode_matches_unsharded_encode(master_port, seed=42):
    run_distributed(worker, 2, (4, 64, 64, (), 0, seed), master_port)


@pytest.mark.gloo
def test_an_attention_block_among_the_down_blocks_is_gathered(master_port, seed=42):
    # An attention reduces over every position, so a rank holding one patch of rows cannot do it
    # alone. attn_scales=(1.0,) puts one at the first stage, where the feature map is largest.
    run_distributed(worker, 2, (4, 64, 64, (1.0,), 0, seed), master_port)


@pytest.mark.gloo
def test_an_image_taller_than_it_is_wide_still_encodes(master_port, seed=42):
    run_distributed(worker, 2, (4, 96, 64, (), 0, seed), master_port)


@pytest.mark.gloo
def test_rows_that_do_not_divide_by_the_rank_count_still_encode(master_port, seed=42):
    run_distributed(worker, 3, (4, 80, 64, (), 0, seed), master_port)


@pytest.mark.gloo
def test_the_chunked_convolution_path_encodes_the_same(master_port, seed=42):
    run_distributed(worker, 2, (4, 64, 64, (), 32, seed), master_port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QwenImageEncoderAdapter GLOO tests")
    parser.add_argument("--world_size", type=int, default=None)
    args, remainder = parser.parse_known_args()
    pytest_args = [os.path.abspath(__file__), "-v"] + remainder
    if args.world_size is not None:
        pytest_args.extend(["-k", f"[{args.world_size}]"])
    sys.exit(pytest.main(pytest_args))
