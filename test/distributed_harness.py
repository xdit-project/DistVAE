"""Scaffolding shared by the multi-rank CPU tests.

Every adapter test asks the same question: does sharding a module across ranks reproduce what
the unsharded module returns. That means the same preamble (a gloo group over CPU), the same
epilogue (compare on rank 0, then fail everywhere rather than deadlocking the ranks that
passed), and the same spawn call. Only the module under test differs.
"""

import os
from typing import Optional

import torch
import torch.distributed as dist
from torch.multiprocessing import spawn

from distvae.utils import DistributedEnv


def init_gloo(rank: int, world_size: int, master_port: int) -> torch.device:
    """Join this rank to a gloo group over CPU and return the device to build on"""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group(backend="gloo", init_method="env://")
    DistributedEnv.initialize(None)
    return torch.device("cpu")


def assert_matches_reference(
    rank: int,
    actual: torch.Tensor,
    expected: Optional[torch.Tensor],
    what: str,
    atol: float = 1e-4,
    rtol: float = 1e-3,
) -> None:
    """Compare on rank 0, then raise on every rank

    Only rank 0 holds the reference, which is why expected is optional elsewhere. Raising there
    alone would leave the other ranks waiting on the next collective, and the test would hang
    instead of failing.
    """
    detail = ""
    ok = torch.ones(1, dtype=torch.int64)
    if rank == 0:
        if actual.shape != expected.shape:
            detail = f"shape {tuple(actual.shape)} != reference {tuple(expected.shape)}"
            ok.zero_()
        elif not torch.allclose(actual, expected, atol=atol, rtol=rtol):
            diff = (actual - expected).abs()
            detail = (
                f"max diff {diff.max().item():.3g}, mean diff {diff.mean().item():.3g} "
                f"(atol={atol}, rtol={rtol})"
            )
            ok.zero_()
    dist.broadcast(ok, src=0)
    dist.barrier()
    if ok.item() == 0:
        raise AssertionError(f"{what} did not match the single-rank reference: {detail}")


def run_distributed(worker, world_size: int, args: tuple, master_port: int) -> None:
    """Spawn world_size ranks running worker(rank, *args); raises if any rank does"""
    spawn(worker, nprocs=world_size, args=(world_size, *args, master_port), join=True)
