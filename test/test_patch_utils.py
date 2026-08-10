"""Splitting rows across ranks and gathering them back, over gloo on CPU.

The pair has to round-trip exactly for row counts that do not divide by the rank count, because
that is where it used to pad the tensor and crop afterwards, and padding is not free: it stops
being zeros at the first convolution and reaches the kept rows from then on. Bands are now cut
unevenly instead, so the gather has to cope with ranks holding different amounts.

Run from repo root:
  pytest test/test_patch_utils.py -v
"""

import argparse
import os
import sys

import pytest
import torch
import torch.distributed as dist

from distvae.modules.patch_utils import DePatchify, Patchify, gather_patches
from distvae.utils import ParallelContext, normalize_patch_dim

from distributed_harness import assert_matches_reference, init_gloo, run_distributed


def round_trip_worker(rank, world_size, rows, scale_factor, patch_dim, seed, master_port):
    init_gloo(rank, world_size, master_port)
    try:
        torch.manual_seed(seed)
        whole = torch.randn(1, 4, rows, rows)

        band = Patchify(patch_dim=patch_dim, scale_factor=scale_factor)(whole)
        # Every band is a whole number of scale_factor rows, which is what keeps a rank's share
        # of a strided convolution on the same grid as the reference's.
        assert band.shape[patch_dim] % scale_factor == 0, (
            f"rank {rank} got {band.shape[patch_dim]} rows, not a multiple of {scale_factor}"
        )
        rebuilt = DePatchify(patch_dim=patch_dim)(band)

        assert_matches_reference(rank, rebuilt, whole if rank == 0 else None, "Patchify round trip")
    finally:
        dist.destroy_process_group()


def gather_worker(rank, world_size, rows, seed, master_port):
    init_gloo(rank, world_size, master_port)
    try:
        torch.manual_seed(seed + rank)
        # Deliberately lopsided: rank r contributes r + 1 rows, so no two ranks agree.
        band = torch.full((1, 2, rank + 1, 3), float(rank))
        bands, sizes = gather_patches(band, patch_dim=2)

        assert sizes == [r + 1 for r in range(world_size)], f"rank {rank} read sizes {sizes}"
        for r, gathered in enumerate(bands):
            assert gathered.shape[2] == r + 1, f"band {r} has {gathered.shape[2]} rows"
            # The pad added for transport must not survive into what callers read back.
            assert torch.equal(gathered, torch.full_like(gathered, float(r))), (
                f"band {r} carries transport padding"
            )
    finally:
        dist.destroy_process_group()


@pytest.mark.gloo
@pytest.mark.parametrize("world_size", [1, 2, 3, 4])
def test_rows_that_divide_by_the_rank_count_round_trip(world_size, master_port, seed=42):
    run_distributed(round_trip_worker, world_size, (24, 1, -2, seed), master_port)


@pytest.mark.gloo
@pytest.mark.parametrize("world_size", [2, 3, 4])
def test_rows_that_do_not_divide_by_the_rank_count_round_trip(world_size, master_port, seed=42):
    # 25 is prime to every rank count here, so at least one band is short in each case.
    run_distributed(round_trip_worker, world_size, (25, 1, -2, seed), master_port)


@pytest.mark.gloo
@pytest.mark.parametrize("world_size", [2, 3])
def test_bands_stay_whole_multiples_of_the_vae_ratio(world_size, master_port, seed=42):
    # 40 rows at a ratio of 8 is 5 bands to share out, which no rank count here divides.
    run_distributed(round_trip_worker, world_size, (40, 8, -2, seed), master_port)


@pytest.mark.gloo
def test_splitting_along_width_round_trips_too(master_port, seed=42):
    run_distributed(round_trip_worker, 3, (25, 1, -1, seed), master_port)


@pytest.mark.gloo
@pytest.mark.parametrize("world_size", [1, 2, 3])
def test_the_gather_hands_back_every_rank_its_own_rows(world_size, master_port, seed=42):
    run_distributed(gather_worker, world_size, (0, seed), master_port)


def refusal_worker(rank, world_size, rows, scale_factor, expected, master_port):
    init_gloo(rank, world_size, master_port)
    try:
        with pytest.raises(ValueError, match=expected):
            Patchify(scale_factor=scale_factor)(torch.randn(1, 2, rows, 4))
    finally:
        dist.destroy_process_group()


@pytest.mark.gloo
def test_rows_that_are_not_a_multiple_of_the_ratio_are_refused(master_port):
    # The encoder narrows by 8, so 20 rows cannot be cut into bands whose latent rows line up.
    run_distributed(refusal_worker, 2, (20, 8, "multiples of 8"), master_port)


@pytest.mark.gloo
def test_more_ranks_than_bands_is_refused(master_port):
    # 16 rows at a ratio of 8 leaves two bands, which three ranks cannot share.
    run_distributed(refusal_worker, 3, (16, 8, "at most 2 ranks"), master_port)


@pytest.mark.parametrize("patch_dim", [-2, 3])
def test_video_height_spellings_normalize_to_the_same_axis(patch_dim):
    assert normalize_patch_dim(patch_dim, ndim=5, spatial_only=True) == -2


@pytest.mark.parametrize("patch_dim", [-1, 4])
def test_video_width_spellings_normalize_to_the_same_axis(patch_dim):
    assert normalize_patch_dim(patch_dim, ndim=5, spatial_only=True) == -1


@pytest.mark.parametrize("patch_dim", [-3, 2])
def test_video_frame_axis_spellings_are_rejected(patch_dim):
    with pytest.raises(ValueError, match="frame axis"):
        normalize_patch_dim(patch_dim, ndim=5, spatial_only=True)


def test_patchifiers_keep_their_own_parallel_context():
    first_context = ParallelContext(group=None, rank=0, world_size=2, patch_dim=-2)
    first = Patchify(parallel_context=first_context)
    second_context = ParallelContext(group=None, rank=1, world_size=2, patch_dim=-1)
    second = Patchify(parallel_context=second_context)
    whole = torch.arange(24).reshape(1, 1, 4, 6)

    assert torch.equal(first(whole), whole[:, :, :2, :])
    assert torch.equal(second(whole), whole[:, :, :, 3:])
    assert first.parallel_context is first_context


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Patchify and gather GLOO tests")
    parser.add_argument("--world_size", type=int, default=None)
    args, remainder = parser.parse_known_args()
    pytest_args = [os.path.abspath(__file__), "-v"] + remainder
    if args.world_size is not None:
        pytest_args.extend(["-k", f"[{args.world_size}]"])
    sys.exit(pytest.main(pytest_args))
