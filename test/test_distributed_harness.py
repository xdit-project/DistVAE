"""The comparison the adapter tests rest on, checked against a mismatch it has to catch.

Every family test reports success by reaching the end of assert_matches_reference, so a bug that
made it accept anything would turn the whole suite green and mean nothing.
"""

import argparse
import os
import sys

import pytest
import torch
import torch.distributed as dist

from distributed_harness import assert_matches_reference, init_gloo, run_distributed


def mismatch_worker(rank, world_size, master_port):
    init_gloo(rank, world_size, master_port)
    try:
        actual = torch.zeros(2, 3)
        expected = torch.ones(2, 3) if rank == 0 else None
        assert_matches_reference(rank, actual, expected, "a deliberately wrong result")
    finally:
        dist.destroy_process_group()


def shape_mismatch_worker(rank, world_size, master_port):
    init_gloo(rank, world_size, master_port)
    try:
        actual = torch.zeros(2, 3)
        expected = torch.zeros(2, 4) if rank == 0 else None
        assert_matches_reference(rank, actual, expected, "a deliberately wrong shape")
    finally:
        dist.destroy_process_group()


def agreement_worker(rank, world_size, master_port):
    init_gloo(rank, world_size, master_port)
    try:
        actual = torch.full((2, 3), 0.5)
        expected = torch.full((2, 3), 0.5) if rank == 0 else None
        assert_matches_reference(rank, actual, expected, "matching results")
    finally:
        dist.destroy_process_group()


@pytest.mark.gloo
@pytest.mark.parametrize("worker", [mismatch_worker, shape_mismatch_worker])
def test_a_wrong_result_fails_every_rank(worker, master_port):
    # Rank 0 is the only one holding a reference, so the failure has to travel: a rank that
    # returned instead would sit in the next collective and hang the run rather than fail it.
    with pytest.raises(Exception) as caught:
        run_distributed(worker, 2, (), master_port)
    assert "did not match the single-rank reference" in str(caught.value)


@pytest.mark.gloo
def test_a_matching_result_passes(master_port):
    run_distributed(agreement_worker, 2, (), master_port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed test harness self-checks")
    _, remainder = parser.parse_known_args()
    sys.exit(pytest.main([os.path.abspath(__file__), "-v"] + remainder))
