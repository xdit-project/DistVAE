"""
Multi-rank integration tests for AsymmetricZeroPadConv2d (GLOO / CPU).

Compares merged distributed output (Patchify -> AsymmetricZeroPadConv2d -> DePatchify)
to the single-rank reference math (must stay in sync with
AsymmetricZeroPadConv2d._conv_forward's group_world_size==1 branch).

Run from repo root:
  pytest test/test_asymmetric_zero_pad_conv2d.py -v -m gloo
  python test/test_asymmetric_zero_pad_conv2d.py
"""

from __future__ import annotations

import argparse
import os
import sys
import zlib

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.multiprocessing import spawn

from distvae.models.layers.asymmetric_zero_pad_conv2d import AsymmetricZeroPadConv2d
from distvae.modules.patch_utils import DePatchify, Patchify
from distributed_harness import make_parallel_context


def reference_asymmetric_zero_pad_conv2d(
    x: torch.Tensor, module: AsymmetricZeroPadConv2d
) -> torch.Tensor:
    pad = tuple(module.reversed_zero_padding)
    x = F.pad(x, pad, mode="constant", value=0)
    y = F.conv2d(
        x,
        module.weight,
        module.bias,
        module.stride,
        module.padding,
        module.dilation,
        module.groups,
    )

    return y


def worker(
    rank: int,
    world_size: int,
    patch_dim: int,
    block_size: int,
    height: int,
    width: int,
    patch_scale_factor: int,
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
    in_ch, out_ch = 8, 8
    n, h, w = 1, height, width
    context = make_parallel_context(patch_dim)

    x_full = torch.randn(n, in_ch, h, w, device=device, dtype=torch.float32)
    layer = AsymmetricZeroPadConv2d(
        in_channels=in_ch,
        out_channels=out_ch,
        kernel_size=3,
        stride=(2, 2),
        dilation=1,
        groups=1,
        bias=True,
        device=device,
        dtype=torch.float32,
        reversed_zero_padding=(0, 1, 0, 1),
        block_size=block_size,
        parallel_context=context,
    ).eval()

    patchify = Patchify(context, scale_factor=patch_scale_factor)
    depatchify = DePatchify(context)

    try:
        with torch.no_grad():
            y_ref = reference_asymmetric_zero_pad_conv2d(x_full, layer)
            x_local = patchify(x_full)
            y_local = layer(x_local)
            y_merged = depatchify(y_local)
        if not torch.allclose(y_ref, y_merged, atol=1e-5, rtol=1e-5):
            raise AssertionError(
                f"AsymmetricZeroPadConv2d distributed output mismatch "
                f"(max diff {(y_ref - y_merged).abs().max().item():.6g})"
            )
        # Leave together. A rank that tears its Gloo context down while another is still holding
        # one exits through std::terminate, which pytest can only report as a spawned process
        # dying on SIGABRT - a teardown race wearing the costume of a failed assertion.
        dist.barrier()
    finally:
        dist.destroy_process_group()


def _run_one(
    world_size: int,
    patch_dim: int,
    block_size: int,
    seed: int,
    master_port: int,
    height: int = 16,
    width: int = 16,
    patch_scale_factor: int = 1,
) -> None:
    spawn(
        worker,
        nprocs=world_size,
        args=(
            world_size,
            patch_dim,
            block_size,
            height,
            width,
            patch_scale_factor,
            seed,
            master_port,
        ),
        join=True,
    )


@pytest.fixture
def master_port(request):
    """Unique port per test to avoid Address already in use when tests run sequentially."""
    # crc32 rather than hash(): the built-in is salted per interpreter, so the port a test binds
    # moved every run and a failure could not be reproduced by asking for that test again.
    base = 29600
    nodeid = request.node.nodeid
    return base + (zlib.crc32(nodeid.encode()) % 10000)


@pytest.mark.gloo
@pytest.mark.parametrize("world_size,patch_dim", [(2, -2), (4, -2), (2, -1)])
def test_asymmetric_zero_pad_conv2d_gloo_matches_single_rank_reference(
    world_size, patch_dim, master_port, seed=42
):
    """Direct path (block_size=0): merged multi-rank output equals single-rank reference."""
    _run_one(
        world_size=world_size,
        patch_dim=patch_dim,
        block_size=0,
        seed=seed,
        master_port=master_port,
    )


@pytest.mark.gloo
@pytest.mark.parametrize("block_size", [1, 4])
def test_asymmetric_zero_pad_conv2d_gloo_chunked_path(
    block_size, master_port, seed=42
):
    """Chunked paths clamp every input chunk to at least the kernel size."""
    _run_one(
        world_size=2,
        patch_dim=-2,
        block_size=block_size,
        seed=seed,
        master_port=master_port,
    )


@pytest.mark.gloo
@pytest.mark.parametrize("patch_dim,block_size", [(-2, 0), (-2, 4), (-1, 0), (-1, 4)])
def test_asymmetric_zero_pad_conv2d_matches_reference_for_unequal_patch_bands(
    patch_dim, block_size, master_port, seed=42
):
    height, width = (40, 16) if patch_dim == -2 else (16, 40)
    _run_one(
        world_size=3,
        patch_dim=patch_dim,
        block_size=block_size,
        height=height,
        width=width,
        patch_scale_factor=8,
        seed=seed,
        master_port=master_port,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AsymmetricZeroPadConv2d GLOO multi-rank tests"
    )
    parser.add_argument("--world_size", type=int, default=None)
    parser.add_argument("--patch_dim", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args, remainder = parser.parse_known_args()
    pytest_args = [os.path.abspath(__file__), "-v"] + remainder
    if args.world_size is not None and args.patch_dim is not None:
        pytest_args.extend(["-k", f"{args.world_size}--{args.patch_dim}"])
    sys.exit(pytest.main(pytest_args))
