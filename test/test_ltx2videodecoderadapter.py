"""LTX2VideoDecoderAdapter against the decoder it shards, over gloo on CPU.

LTX-2 is the easiest of these to shard and the hardest to read. Its mid block has no attention
at all, so nothing needs the whole image, and its spatial padding already lives inside the
convolution rather than being applied around it. The shipped LTX-2 pads by reflection, so the
default here is reflect rather than the zeros LTX-2.3 uses.

Run from repo root:
  pytest test/test_ltx2videodecoderadapter.py -v
"""

import argparse
import os
import sys

import pytest
import torch
import torch.distributed as dist

from distvae.modules.adapters.vae.decoder_adapters import LTX2VideoDecoderAdapter

from distributed_harness import assert_matches_reference, init_gloo, run_distributed

diffusers = pytest.importorskip("diffusers")
if not hasattr(diffusers, "AutoencoderKLLTX2Video"):
    pytest.skip("installed diffusers has no AutoencoderKLLTX2Video", allow_module_level=True)

# Four channel stages preserve the decoder's spatial compression structure.
CONFIG = dict(
    block_out_channels=(8, 16, 32, 32),
    latent_channels=8,
    layers_per_block=(1, 1, 1, 1, 1),
    spatial_compression_ratio=32,
)
LATENT_CHANNELS = 8


def build_decoder(spatial_padding_mode="reflect", inject_noise=False):
    vae = diffusers.AutoencoderKLLTX2Video(
        **CONFIG,
        decoder_spatial_padding_mode=spatial_padding_mode,
        decoder_inject_noise=inject_noise,
    )
    return vae.eval().decoder


def worker(
    rank, world_size, frames, height, width, padding_mode, conv_block_size, seed, master_port
):
    init_gloo(rank, world_size, master_port)
    try:
        torch.manual_seed(seed)
        decoder = build_decoder(padding_mode)
        # Taken before the adapter runs, which rebuilds the decoder in place.
        weights = decoder.state_dict()

        latents = torch.randn(1, LATENT_CHANNELS, frames, height, width)

        with torch.no_grad():
            expected = None
            if rank == 0:
                reference = build_decoder(padding_mode)
                reference.load_state_dict(weights)
                expected = reference(latents)

            adapter = LTX2VideoDecoderAdapter(
                decoder, vae_group=None, conv_block_size=conv_block_size
            ).eval()
            actual = adapter(latents)

        assert_matches_reference(rank, actual, expected, "LTX2VideoDecoderAdapter")
    finally:
        dist.destroy_process_group()


def refusal_worker(rank, world_size, master_port):
    init_gloo(rank, world_size, master_port)
    try:
        torch.manual_seed(0)
        LTX2VideoDecoderAdapter(build_decoder(inject_noise=True), vae_group=None)
    finally:
        dist.destroy_process_group()


@pytest.mark.gloo
def test_a_sharded_ltx2_decode_matches_a_single_rank_one(master_port, seed=42):
    run_distributed(worker, 2, (1, 16, 16, "reflect", 0, seed), master_port)


@pytest.mark.gloo
def test_the_zeros_padding_ltx23_ships_decodes_the_same(master_port, seed=42):
    run_distributed(worker, 2, (1, 16, 16, "zeros", 0, seed), master_port)


@pytest.mark.gloo
def test_a_latent_taller_than_it_is_wide_still_decodes(master_port, seed=42):
    run_distributed(worker, 2, (1, 24, 16, "reflect", 0, seed), master_port)


@pytest.mark.gloo
def test_more_than_one_frame_still_decodes(master_port, seed=42):
    # The temporal padding repeats the end frames rather than padding with anything, and it is
    # left to run in the wrapped module, so this is what confirms it survived the swap.
    run_distributed(worker, 2, (3, 16, 16, "reflect", 0, seed), master_port)


@pytest.mark.gloo
def test_the_chunked_convolution_path_decodes_the_same(master_port, seed=42):
    run_distributed(worker, 2, (1, 16, 16, "reflect", 32, seed), master_port)


@pytest.mark.gloo
def test_latent_rows_that_do_not_divide_by_the_rank_count(master_port, seed=42):
    # Uneven bands must preserve all 16 rows without padding the decoder input.
    run_distributed(worker, 3, (1, 16, 16, "reflect", 0, seed), master_port)


@pytest.mark.gloo
def test_injected_noise_is_refused_rather_than_drawn_per_rank(master_port):
    # Each rank would draw noise for its own rows, and together they would not reconstruct what
    # one rank draws, so the decode could not match its reference. No shipped config enables it.
    with pytest.raises(Exception) as caught:
        run_distributed(refusal_worker, 2, (), master_port)
    assert "inject_noise" in str(caught.value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LTX2VideoDecoderAdapter GLOO tests")
    parser.add_argument("--world_size", type=int, default=None)
    args, remainder = parser.parse_known_args()
    pytest_args = [os.path.abspath(__file__), "-v"] + remainder
    if args.world_size is not None:
        pytest_args.extend(["-k", f"[{args.world_size}]"])
    sys.exit(pytest.main(pytest_args))
