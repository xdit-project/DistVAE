"""PatchGroupNorm against nn.GroupNorm, over gloo on CPU.

GroupNorm is the one normalisation in a VAE decoder whose statistics span the axis being split,
so it is the one that has to be summed across ranks. The equivalent check exists in
test_groupnorm.py, but only as a torchrun script needing NCCL and a GPU.

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

from distributed_harness import (
    assert_matches_reference,
    assert_no_less_precise_than,
    init_gloo,
    run_distributed,
)


def worker(rank, world_size, shape, num_groups, patch_dim, seed, master_port):
    init_gloo(rank, world_size, master_port)
    try:
        torch.manual_seed(seed)
        channels = shape[1]
        norm = nn.GroupNorm(
            num_groups=num_groups, num_channels=channels, eps=1e-6, affine=True
        ).eval()
        # Shifted per channel, so the group statistics are not already near zero mean and unit
        # variance and an incorrect reduction has somewhere to show up.
        x = torch.randn(*shape) * 3.0 + 2.0

        patchify = Patchify(patch_dim=patch_dim)
        depatchify = DePatchify(patch_dim=patch_dim)
        sharded = GroupNormAdapter(norm)

        with torch.no_grad():
            expected = norm(x) if rank == 0 else None
            actual = depatchify(sharded(patchify(x)))

        assert_matches_reference(rank, actual, expected, "PatchGroupNorm", atol=1e-5)
    finally:
        dist.destroy_process_group()


@pytest.mark.gloo
@pytest.mark.parametrize("world_size", [1, 2, 4])
def test_it_matches_group_norm_on_a_feature_map(world_size, master_port, seed=42):
    run_distributed(worker, world_size, ((1, 16, 16, 16), 8, -2, seed), master_port)


@pytest.mark.gloo
@pytest.mark.parametrize("world_size", [1, 2])
def test_it_matches_group_norm_on_a_video_feature_map(world_size, master_port, seed=42):
    # The video VAEs normalise over (F, H, W), so the reduction has to cover the axes either
    # side of the one being split, not just the split one.
    run_distributed(worker, world_size, ((1, 16, 3, 8, 8), 4, -2, seed), master_port)


@pytest.mark.gloo
def test_it_matches_group_norm_when_the_width_is_split(master_port, seed=42):
    run_distributed(worker, 2, ((1, 16, 16, 16), 8, -1, seed), master_port)


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
            actual = depatchify(GroupNormAdapter(norm)(patchify(x)))

        assert_no_less_precise_than(rank, actual, stock, gold, "PatchGroupNorm in bfloat16")
    finally:
        dist.destroy_process_group()


# One rank is the interesting case rather than the lenient one: nothing is sharded, so any loss
# here is the substitution of PatchGroupNorm for nn.GroupNorm and nothing else. It is also the
# case the benchmark harness cannot excuse - it allows bf16 sharding a few percent on the grounds
# that splitting reorders the arithmetic, which at one rank has not happened.
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
