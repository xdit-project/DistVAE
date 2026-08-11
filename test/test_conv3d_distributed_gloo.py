"""
Multi-rank integration tests for PatchConv3d using GLOO backend (CPU).

Run from repo root:
  pytest test/test_conv3d_distributed_gloo.py -v
  pytest test/test_conv3d_distributed_gloo.py -v -k "2--2"
  python test/test_conv3d_distributed_gloo.py --world_size 4 --patch_dim -1
"""

import argparse
import os
import sys

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn

from distvae.modules.patch_utils import Patchify, DePatchify
from distvae.modules.adapters.layers.conv_adapters import Conv3dAdapter

from distributed_harness import make_parallel_context, run_distributed


def worker(
    rank: int,
    world_size: int,
    patch_dim: int,
    kernel_size: int,
    stride: int,
    padding: int,
    block_size: int,
    size: tuple,
    seed: int,
    master_port: int,
) -> None:
    device = torch.device("cpu")
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group(backend="gloo", init_method="env://")

    torch.manual_seed(seed)
    in_ch, out_ch = 4, 8
    # (N, C, F, H, W): patch_dim -2 => H, -1 => W
    # For stride>1 tests, use sizes that stress-test alignment logic
    # For stride=1, use sizes divisible by world_size for even splitting
    n, c, f = 1, in_ch, 4
    # Both branches of the conditional this replaces set 8 by 8, so the comment about needing
    # even sizes for stride > 1 described the only case there was, and the assertions below it
    # enforced an even split that no shipped decode gets. The split axis is now given by the
    # caller, so a test can ask for a size that leaves the ranks holding different amounts -
    # which is the case the halo widths and the crop are actually difficult for.
    h, w = size

    x_full = torch.randn(n, c, f, h, w, device=device, dtype=torch.float32)
    ref_conv = nn.Conv3d(
        in_ch, out_ch, kernel_size, stride=stride, padding=padding
    ).to(device)
    ref_conv.eval()

    context = make_parallel_context(patch_dim)
    patchify = Patchify(context)
    depatchify = DePatchify(context)
    adapter = Conv3dAdapter(
        ref_conv, block_size=block_size, parallel_context=context
    )
    adapter.eval()

    with torch.no_grad():
        y_ref = ref_conv(x_full) if rank == 0 else None

        x_local = patchify(x_full)
        y_local = adapter(x_local)
        y_patch = depatchify(y_local)

    success = torch.ones(1, dtype=torch.int64, device=device)
    if rank == 0:
        if not torch.allclose(y_ref, y_patch, atol=1e-5):
            success.zero_()
    dist.broadcast(success, src=0)
    dist.barrier()
    dist.destroy_process_group()
    if success.item() == 0:
        raise AssertionError("PatchConv3d output did not match reference nn.Conv3d")


def _run_one(
    world_size: int,
    patch_dim: int,
    kernel_size: int,
    stride: int,
    padding: int,
    block_size: int,
    seed: int,
    master_port: int,
    size: tuple = (8, 8),
) -> None:
    """Spawn processes and run worker; raises on failure."""
    # Through the shared harness rather than spawn directly, so a port claimed between being
    # found free and being bound is retried rather than failing the test.
    run_distributed(
        worker,
        world_size,
        (patch_dim, kernel_size, stride, padding, block_size, size, seed),
        master_port,
    )


# The port comes from conftest, which keys it on a crc32 of the test's id rather than on hash().
# hash() over a str is salted per process, so the fixture that used to live here picked a
# different port every run - and a run that fails on a port collision is then a run nobody can
# reproduce. Its range overlapped conftest's as well, so the two could hand out the same port.


@pytest.mark.gloo
@pytest.mark.parametrize("world_size,patch_dim", [(4, -2), (4, -1)])
def test_patch_conv3d_gloo_direct(world_size, patch_dim, master_port, seed=42):
    """PatchConv3d with GLOO: direct path (block_size=0), multiple world sizes and patch dims."""
    _run_one(
        world_size=world_size,
        patch_dim=patch_dim,
        kernel_size=3,
        stride=1,
        padding=1,
        block_size=0,
        seed=seed,
        master_port=master_port,
    )


@pytest.mark.gloo
@pytest.mark.parametrize("world_size,patch_dim", [(4, -2), (4, -1), (3, -2)])
def test_patch_conv3d_gloo_on_bands_of_different_sizes(
    world_size, patch_dim, master_port, seed=42
):
    """The split axis not dividing by the rank count, which is what the 8 by 8 above never gives

    Every band being the same size is the easy case: the halo each rank asks of its neighbour is
    the same, and the crop starts at the same offset into each. Nine rows over four ranks gives
    3, 2, 2, 2, and the sizes stop being interchangeable - a rank that assumes its neighbour
    matches it reads the wrong rows, and a global quantity derived from a local one is wrong on
    every rank but one.
    """
    _run_one(
        world_size=world_size,
        patch_dim=patch_dim,
        kernel_size=3,
        stride=1,
        padding=1,
        block_size=0,
        seed=seed,
        master_port=master_port,
        size=(9, 7),
    )


@pytest.mark.gloo
def test_patch_conv3d_gloo_chunked_path(master_port, seed=42):
    """PatchConv3d with GLOO: chunked path (block_size=4 so _use_direct_path is False, and chunks >= kernel_size=3)."""
    _run_one(
        world_size=2,
        patch_dim=-2,
        kernel_size=3,
        stride=1,
        padding=1,
        block_size=4,
        seed=seed,
        master_port=master_port,
    )


@pytest.mark.gloo
@pytest.mark.parametrize("world_size,patch_dim", [(4, -2), (2, -1)])
def test_patch_conv3d_stride2_alignment(world_size, patch_dim, master_port, seed=42):
    """PatchConv3d at stride 2, where the crop has to be placed from the global position

    A strided convolution's output grid is set by where a rank's patch begins in the whole
    image, not by where it begins in that rank, so build_crop_slice is given the global start
    and the ranks would otherwise cut their outputs at offsets that do not join up.
    """
    _run_one(
        world_size=world_size,
        patch_dim=patch_dim,
        kernel_size=3,
        stride=2,
        padding=1,
        block_size=0,  # Direct path
        seed=seed,
        master_port=master_port,
    )


@pytest.mark.gloo
@pytest.mark.parametrize("block_size", [0, 2, 4])
@pytest.mark.parametrize("size", [(9, 7), (15, 11)])
@pytest.mark.parametrize("world_size,patch_dim", [(4, -2), (3, -2), (2, -1)])
def test_patch_conv3d_stride2_on_bands_of_different_sizes(
    world_size, patch_dim, size, block_size, master_port, seed=42
):
    """Halving an uneven split, which the sizes elsewhere in this file were chosen to avoid

    A TODO used to sit beside those even sizes saying odd extents at stride > 1 produce
    off-by-one errors, and the test was shaped around it rather than at it. Written first as a
    non-strict xfail so a known bug would not be quietly forgotten, it passed on both of its
    cases, so the claim is checked here instead of recorded: three splits, two shapes that
    divide by none of the rank counts, and both the direct and the chunked convolution.

    A block of 2 against a kernel of 3 was tried first and cut chunks no convolution can run on.
    So did a block of 4, once the frame axis was chunked as well: 4 frames padded to 6, cut in
    two at stride 2, ends on a chunk of 2. That is chunk_bounds' to answer rather than each
    caller's to avoid, and it now takes no more chunks than leave every one of them a kernel
    long, so both blocks work and the small one is kept here.

    If the off-by-one is real it is not this. Should one of these ever fail, it is the arithmetic
    that is wrong and not the expectation - a strided convolution over an uneven split has to
    match nn.Conv3d, or a decode of any image whose rows do not divide by the rank count is wrong.
    """
    _run_one(
        world_size=world_size,
        patch_dim=patch_dim,
        kernel_size=3,
        stride=2,
        padding=1,
        block_size=block_size,
        seed=seed,
        master_port=master_port,
        size=size,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PatchConv3d GLOO multi-rank tests")
    parser.add_argument("--world_size", type=int, default=None)
    parser.add_argument("--patch_dim", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args, remainder = parser.parse_known_args()
    # Pass through any remaining args to pytest (e.g. -v, -k)
    pytest_args = [os.path.abspath(__file__), "-v"] + remainder
    if args.world_size is not None and args.patch_dim is not None:
        pytest_args.extend(["-k", f"{args.world_size}--{args.patch_dim}"])
    sys.exit(pytest.main(pytest_args))
