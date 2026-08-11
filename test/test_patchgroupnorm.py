"""PatchGroupNorm against nn.GroupNorm over multiple ranks.

GroupNorm statistics include the split spatial axis, so group sums and variances must be
aggregated across ranks.

Run from repo root:
  pytest test/test_patchgroupnorm.py -v
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
    make_parallel_context,
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

        context = make_parallel_context(patch_dim)
        patchify = Patchify(context)
        depatchify = DePatchify(context)
        sharded = GroupNormAdapter(norm, parallel_context=context)

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
def test_it_matches_group_norm_when_an_odd_height_is_split(master_port, seed=42):
    """The height case that catches a norm summing across the wrong axis

    The square even split above cannot: the axis only reaches the arithmetic through the element
    count, and counting columns where the split is on rows over-counts by exactly the factor it
    under-counts by. At 16x16 over two ranks both readings come to 512, so a norm reducing along
    W passes a test named for H. Fifteen rows over two ranks gives one rank 8 and the other 7,
    which is what stops the two cancelling.
    """
    run_distributed(
        worker, 2, ((1, 16, 15, 4), 8, -2, seed, True), master_port
    )


@pytest.mark.gloo
@pytest.mark.parametrize("world_size", [1, 2])
def test_it_matches_group_norm_on_a_video_feature_map(world_size, master_port, seed=42):
    # Video GroupNorm reduces over all of (F, H, W), including the axes around the split axis.
    run_distributed(worker, world_size, ((1, 16, 3, 8, 8), 4, -2, seed, True), master_port)


@pytest.mark.gloo
def test_it_matches_group_norm_on_a_video_map_of_three_different_extents(master_port, seed=42):
    # F, H and W all different and the split uneven, so confusing the split axis for either of
    # the two it is reduced alongside changes the count rather than cancelling against it.
    run_distributed(
        worker, 2, ((1, 16, 3, 7, 8), 4, -2, seed, True), master_port
    )


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
    context = ParallelContext(None, rank=0, world_size=1, patch_dim=2)
    norm = GroupNormAdapter(nn.GroupNorm(1, 2), parallel_context=context)
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


@pytest.mark.gloo
def test_it_matches_group_norm_when_an_odd_width_is_split(master_port, seed=42):
    """The case that catches a norm summing across the wrong axis

    A width of 15 over two ranks gives one rank 8 columns and the other 7. That unevenness is
    what makes the axis matter: split evenly, counting rows where the split is on columns
    happens to arrive at the same element count anyway - the row count is over-counted by
    exactly the factor the column count is under-counted by, and the two cancel. The square
    width-split case above therefore passed while the norm was reducing along height.
    """
    run_distributed(
        worker, 2, ((1, 16, 4, 15), 8, -1, seed, True), master_port
    )


def told_worker(rank, world_size, shape, num_groups, patch_dim, seed, master_port):
    """A norm reads its axis from its own context."""
    init_gloo(rank, world_size, master_port)
    try:
        torch.manual_seed(seed)
        norm = nn.GroupNorm(
            num_groups=num_groups, num_channels=shape[1], eps=1e-6, affine=True
        ).eval()
        x = torch.randn(*shape) * 3.0 + 2.0
        context = make_parallel_context(patch_dim)

        with torch.no_grad():
            expected = norm(x) if rank == 0 else None
            sharded = GroupNormAdapter(norm, parallel_context=context)
            actual = DePatchify(context)(sharded(Patchify(context)(x)))

        assert_matches_reference(rank, actual, expected, "PatchGroupNorm told", atol=1e-5)
    finally:
        dist.destroy_process_group()


@pytest.mark.gloo
@pytest.mark.parametrize("patch_dim", [-2, -1])
def test_the_context_axis_selects_the_uneven_spatial_band(patch_dim, master_port, seed=42):
    # Uneven along whichever axis is split, so that being told the wrong one would show.
    shape = (1, 16, 15, 4) if patch_dim == -2 else (1, 16, 4, 15)
    run_distributed(told_worker, 2, (shape, 8, patch_dim, seed), master_port)


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

            context = make_parallel_context(patch_dim)
            patchify = Patchify(context)
            depatchify = DePatchify(context)
            actual = depatchify(
                GroupNormAdapter(norm, parallel_context=context)(patchify(x))
            )

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
