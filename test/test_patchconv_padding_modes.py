"""PatchConv2d and PatchConv3d under padding modes other than zeros, over gloo on CPU.

Replicate and reflect padding are what the HunyuanVideo and LTX-2 VAEs convolve with, and they
are the case a sharded convolution can get quietly wrong: left alone, a rank repeats or mirrors
its own edge rows at a boundary that is not an edge of the image at all.

Run from repo root:
  pytest test/test_patchconv_padding_modes.py -v
"""

import argparse
import os
import sys

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn

from distvae.models.layers.conv2d import PatchConv2d
from distvae.models.layers.conv3d import PatchConv3d
from distvae.modules.patch_utils import DePatchify, Patchify

from distributed_harness import (
    assert_matches_reference,
    init_gloo,
    make_parallel_context,
    run_distributed,
)


def worker(
    rank, world_size, ndim, padding_mode, kernel_size, padding, block_size, patch_dim, seed,
    master_port,
):
    init_gloo(rank, world_size, master_port)
    try:
        torch.manual_seed(seed)
        in_channels, out_channels = 4, 8
        context = make_parallel_context(patch_dim)
        if ndim == 5:
            shape = (1, in_channels, 3, 16, 16)
            reference = nn.Conv3d(
                in_channels, out_channels, kernel_size, padding=padding,
                padding_mode=padding_mode,
            ).eval()
            sharded = PatchConv3d(
                in_channels, out_channels, kernel_size, padding=padding,
                padding_mode=padding_mode, block_size=block_size,
                parallel_context=context,
            ).eval()
        else:
            shape = (1, in_channels, 16, 16)
            reference = nn.Conv2d(
                in_channels, out_channels, kernel_size, padding=padding,
                padding_mode=padding_mode,
            ).eval()
            sharded = PatchConv2d(
                in_channels, out_channels, kernel_size, padding=padding,
                padding_mode=padding_mode, block_size=block_size,
                parallel_context=context,
            ).eval()
        sharded.weight.data = reference.weight.data
        sharded.bias.data = reference.bias.data

        x = torch.randn(*shape)
        patchify = Patchify(context)
        depatchify = DePatchify(context)

        with torch.no_grad():
            expected = reference(x) if rank == 0 else None
            actual = depatchify(sharded(patchify(x)))

        assert_matches_reference(
            rank, actual, expected, f"a {padding_mode}-padded convolution", atol=1e-5
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.gloo
@pytest.mark.parametrize("padding_mode", ["replicate", "reflect"])
@pytest.mark.parametrize("world_size", [1, 2, 4])
def test_a_3d_convolution_matches_an_unsharded_one(padding_mode, world_size, master_port, seed=42):
    run_distributed(worker, world_size, (5, padding_mode, 3, 1, 0, -2, seed), master_port)


@pytest.mark.gloo
def test_circular_padding_is_refused_rather_than_wrapped_within_a_patch(master_port, seed=42):
    # Circular padding reads from the far edge of the image, which no neighbour holds. Wrapping
    # within the patch instead would be silently wrong, so the convolution has to say so.
    with pytest.raises(Exception) as caught:
        run_distributed(worker, 2, (5, "circular", 3, 1, 0, -2, seed), master_port)
    assert "circular" in str(caught.value)


@pytest.mark.gloo
def test_circular_padding_is_still_allowed_on_a_single_rank(master_port, seed=42):
    run_distributed(worker, 1, (5, "circular", 3, 1, 0, -2, seed), master_port)


@pytest.mark.gloo
@pytest.mark.parametrize("padding_mode", ["replicate", "reflect"])
@pytest.mark.parametrize("world_size", [1, 2, 4])
def test_a_2d_convolution_matches_an_unsharded_one(padding_mode, world_size, master_port, seed=42):
    run_distributed(worker, world_size, (4, padding_mode, 3, 1, 0, -2, seed), master_port)


@pytest.mark.gloo
@pytest.mark.parametrize("padding_mode", ["replicate", "reflect"])
def test_the_chunked_path_pads_the_same_way(padding_mode, master_port, seed=42):
    run_distributed(worker, 2, (5, padding_mode, 3, 1, 4, -2, seed), master_port)


@pytest.mark.gloo
def test_splitting_the_width_instead_pads_the_same_way(master_port, seed=42):
    run_distributed(worker, 2, (5, "replicate", 3, 1, 0, -1, seed), master_port)


@pytest.mark.gloo
def test_a_wider_kernel_pads_the_same_way(master_port, seed=42):
    # Kernel 3 with padding 1 has a fast path of its own; a kernel of 5 goes the other way.
    run_distributed(worker, 2, (5, "replicate", 5, 2, 0, -2, seed), master_port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PatchConv padding mode GLOO tests")
    _, remainder = parser.parse_known_args()
    sys.exit(pytest.main([os.path.abspath(__file__), "-v"] + remainder))
