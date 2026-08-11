"""PatchConv2d against nn.Conv2d, over gloo on CPU, at sizes that do not divide evenly.

This was a torchrun script whose only verdict was a print: it computed the difference, printed
"FAILED" when it was too large, and exited 0 either way, with the one real assertion commented
out at the bottom. Nothing ran it and nothing could have failed it, which is a shame, because
the sizes it swept are the interesting ones - odd extents, and extents that do not divide by the
rank count, at stride 1 and stride 2. Those are what exercise the halo widths and the
global-position cropping, and they are what is kept here.

Sizes are smaller than the original 1024x1024 at 64 channels, which was sized for a GPU. What
makes a size interesting here is its remainder against the rank count and its parity, not its
magnitude.

Run from repo root:
  pytest test/test_conv2d.py -v
"""

import argparse
import os
import sys

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn

from distvae.modules.adapters.layers.conv_adapters import Conv2dAdapter
from distvae.modules.patch_utils import DePatchify, Patchify

from distributed_harness import (
    assert_matches_reference,
    init_gloo,
    make_parallel_context,
    run_distributed,
)


def worker(rank, world_size, size, kernel, stride, padding, patch_dim, seed, master_port):
    init_gloo(rank, world_size, master_port)
    try:
        torch.manual_seed(seed)
        height, width = size
        conv = nn.Conv2d(4, 3, kernel, stride=stride, padding=padding).eval()
        x = torch.randn(1, 4, height, width)

        with torch.no_grad():
            expected = conv(x) if rank == 0 else None
            context = make_parallel_context(patch_dim)
            sharded = Conv2dAdapter(conv, parallel_context=context)
            actual = DePatchify(context)(sharded(Patchify(context)(x)))

        assert_matches_reference(rank, actual, expected, "PatchConv2d", atol=1e-5)
    finally:
        dist.destroy_process_group()


# Odd against even, and extents whose remainder against the rank count differs between the two
# axes, so a run cannot pass by having every band the same size.
SIZES = [(32, 32), (33, 31), (31, 33), (45, 28)]


@pytest.mark.gloo
@pytest.mark.parametrize("world_size", [1, 2, 4])
@pytest.mark.parametrize("size", SIZES)
def test_it_matches_conv2d_at_unit_stride(world_size, size, master_port, seed=42):
    run_distributed(worker, world_size, (size, 3, 1, 1, -2, seed), master_port)


@pytest.mark.gloo
@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.parametrize("size", SIZES)
def test_it_matches_conv2d_when_it_halves(world_size, size, master_port, seed=42):
    """Stride 2, which is where the crop has to know where its band starts

    At unit stride every output row is an input row and the halo alone lines the bands up. A
    strided convolution steps a grid the whole image shares, so a band starting at a row that is
    not on that grid has to be cropped from where the grid next lands rather than from its own
    first row - which is the arithmetic an even split never exercises.
    """
    run_distributed(worker, world_size, (size, 3, 2, 1, -2, seed), master_port)


@pytest.mark.gloo
@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.parametrize("size", SIZES)
def test_it_matches_conv2d_when_the_width_is_split(world_size, size, master_port, seed=42):
    # The same convolution against the other axis, which the layer supports and nothing above it
    # used to check at anything but a square.
    run_distributed(worker, world_size, (size, 3, 1, 1, -1, seed), master_port)


@pytest.mark.gloo
@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.parametrize("kernel,padding", [(1, 0), (5, 2), (7, 3)])
def test_it_matches_conv2d_across_kernel_widths(world_size, kernel, padding, master_port, seed=42):
    # The halo is kernel // 2 rows either side, so a wider kernel asks more of a neighbour than
    # the thin bands an uneven split leaves have to spare.
    run_distributed(worker, world_size, ((33, 31), kernel, 1, padding, -2, seed), master_port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PatchConv2d GLOO multi-rank tests")
    parser.add_argument("--world_size", type=int, default=None)
    args, remainder = parser.parse_known_args()
    pytest_args = [os.path.abspath(__file__), "-v"] + remainder
    if args.world_size is not None:
        pytest_args.extend(["-k", f"[{args.world_size}]"])
    sys.exit(pytest.main(pytest_args))
