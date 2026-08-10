"""PatchGroupNorm against nn.GroupNorm over multiple ranks.

GroupNorm statistics include the split spatial axis, so group sums and variances must be
aggregated across ranks.
"""

import argparse
import os
import sys

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn

from distvae.modules.adapters.layers.norm_adapters import GroupNormAdapter
from distvae.modules.patch_utils import DePatchify, Patchify
from distvae.utils import ParallelContext

from distributed_harness import (
    assert_matches_reference,
    assert_no_less_precise_than,
    init_gloo,
    run_distributed,
)


def worker(rank, world_size, shape, num_groups, patch_dim, seed, affine, master_port):
    init_gloo(rank, world_size, master_port)
    try:
        torch.manual_seed(seed)
        channels = shape[1]
        norm = nn.GroupNorm(
            num_groups=num_groups, num_channels=channels, eps=1e-6, affine=affine
        ).eval()
        # Shifted per channel, so the group statistics are not already near zero mean and unit
        # variance and an incorrect reduction has somewhere to show up.
        x = torch.randn(*shape) * 3.0 + 2.0

        patchify = Patchify(patch_dim=patch_dim)
        depatchify = DePatchify(patch_dim=patch_dim)
        sharded = GroupNormAdapter(norm, patch_dim=patch_dim)

        with torch.no_grad():
            expected = norm(x) if rank == 0 else None
            actual = depatchify(sharded(patchify(x)))

        assert_matches_reference(rank, actual, expected, "PatchGroupNorm", atol=1e-5)
    finally:
        dist.destroy_process_group()


@pytest.mark.gloo
@pytest.mark.parametrize("world_size", [1, 2, 4])
def test_it_matches_group_norm_on_a_feature_map(world_size, master_port, seed=42):
    run_distributed(worker, world_size, ((1, 16, 16, 16), 8, -2, seed, True), master_port)


@pytest.mark.gloo
@pytest.mark.parametrize("world_size", [1, 2])
def test_it_matches_group_norm_on_a_video_feature_map(world_size, master_port, seed=42):
    # Video GroupNorm reduces over all of (F, H, W), including the axes around the split axis.
    run_distributed(worker, world_size, ((1, 16, 3, 8, 8), 4, -2, seed, True), master_port)


@pytest.mark.gloo
def test_it_matches_group_norm_when_the_width_is_split(master_port, seed=42):
    run_distributed(worker, 2, ((1, 16, 16, 16), 8, -1, seed, True), master_port)


@pytest.mark.gloo
@pytest.mark.parametrize(
    "shape,patch_dim",
    [
        pytest.param((1, 16, 10, 8), -2, id="uneven-height"),
        pytest.param((1, 16, 8, 10), -1, id="uneven-width"),
    ],
)
def test_it_matches_group_norm_on_uneven_spatial_bands_without_affine(
    shape, patch_dim, master_port, seed=42
):
    run_distributed(
        worker, 3, (shape, 8, patch_dim, seed, False), master_port
    )


def test_video_frame_axis_is_rejected_in_its_positive_spelling():
    norm = GroupNormAdapter(nn.GroupNorm(1, 2), patch_dim=2)
    with pytest.raises(ValueError, match="frame axis"):
        norm(torch.randn(1, 2, 3, 4, 4))


def test_constructing_a_second_norm_adapter_does_not_reconfigure_the_first(monkeypatch):
    first_group, second_group = object(), object()
    first_context = ParallelContext(first_group, rank=0, world_size=2, patch_dim=-2)
    second_context = ParallelContext(second_group, rank=0, world_size=2, patch_dim=-1)
    first = GroupNormAdapter(nn.GroupNorm(1, 2), parallel_context=first_context)
    GroupNormAdapter(nn.GroupNorm(1, 2), parallel_context=second_context)
    used_groups = []

    monkeypatch.setattr(
        dist,
        "all_reduce",
        lambda tensor, group=None: used_groups.append(group),
    )
    first(torch.randn(1, 2, 2, 2))

    assert used_groups == [first_group, first_group]


def bfloat16_worker(rank, world_size, shape, num_groups, patch_dim, seed, master_port):
    """PatchGroupNorm's bf16 rounding against nn.GroupNorm's own, both judged by the fp32 answer"""
    init_gloo(rank, world_size, master_port)
    try:
        torch.manual_seed(seed)
        channels = shape[1]
        norm = nn.GroupNorm(
            num_groups=num_groups, num_channels=channels, eps=1e-6, affine=True
        ).eval()
        x = torch.randn(*shape) * 3.0 + 2.0

        with torch.no_grad():
            # Before the cast: nn.Module.to is in place, and this needs the float32 answer.
            gold = norm(x).to(torch.bfloat16) if rank == 0 else None
            norm = norm.to(torch.bfloat16)
            x = x.to(torch.bfloat16)
            stock = norm(x) if rank == 0 else None

            patchify = Patchify(patch_dim=patch_dim)
            depatchify = DePatchify(patch_dim=patch_dim)
            actual = depatchify(GroupNormAdapter(norm, patch_dim=patch_dim)(patchify(x)))

        assert_no_less_precise_than(rank, actual, stock, gold, "PatchGroupNorm in bfloat16")
    finally:
        dist.destroy_process_group()


@pytest.mark.gloo
@pytest.mark.parametrize("world_size", [1, 2, 4])
def test_it_rounds_no_worse_than_group_norm_in_bfloat16(world_size, master_port, seed=42):
    run_distributed(bfloat16_worker, world_size, ((1, 32, 64, 64), 32, -2, seed), master_port)


@pytest.mark.gloo
@pytest.mark.parametrize("world_size", [1, 2])
def test_it_rounds_no_worse_than_group_norm_in_bfloat16_on_video(world_size, master_port, seed=42):
    run_distributed(bfloat16_worker, world_size, ((1, 16, 3, 8, 8), 4, -2, seed), master_port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PatchGroupNorm GLOO multi-rank tests")
    parser.add_argument("--world_size", type=int, default=None)
    args, remainder = parser.parse_known_args()
    pytest_args = [os.path.abspath(__file__), "-v"] + remainder
    if args.world_size is not None:
        pytest_args.extend(["-k", f"[{args.world_size}]"])
    sys.exit(pytest.main(pytest_args))
