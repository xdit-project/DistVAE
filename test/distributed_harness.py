"""Scaffolding shared by the multi-rank CPU tests.

Every adapter test asks the same question: does sharding a module across ranks reproduce what
the unsharded module returns. That means the same preamble (a gloo group over CPU), the same
epilogue (compare on rank 0, then fail everywhere rather than deadlocking the ranks that
passed), and the same spawn call. Only the module under test differs.
"""

import os
import socket
from typing import Optional

import torch
import torch.distributed as dist
from torch.multiprocessing import spawn
from torch.multiprocessing.spawn import ProcessRaisedException

from distvae.utils import ParallelContext

# How many ports to try before giving up on finding a free one.
_RENDEZVOUS_ATTEMPTS = 4


def init_gloo(rank: int, world_size: int, master_port: int) -> torch.device:
    """Join this rank to a gloo group over CPU and return the device to build on"""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group(backend="gloo", init_method="env://")
    return torch.device("cpu")


def make_parallel_context(patch_dim: int = -2) -> ParallelContext:
    """Capture the current test process group, or a one-rank local context."""
    if not dist.is_initialized():
        return ParallelContext(None, rank=0, world_size=1, patch_dim=patch_dim)
    group = dist.group.WORLD
    world_size = dist.get_world_size(group)
    return ParallelContext(
        group,
        rank=dist.get_rank(group),
        world_size=world_size,
        patch_dim=patch_dim,
        global_ranks=tuple(range(world_size)),
    )


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


def assert_no_less_precise_than(
    rank: int,
    actual: torch.Tensor,
    stock: Optional[torch.Tensor],
    gold: Optional[torch.Tensor],
    what: str,
    slack: float = 1.5,
) -> None:
    """Compare our rounding against the stock operator's, then raise on every rank

    A low-precision dtype cannot reproduce a float32 answer, so asserting equality there is
    asserting a failure: every test in this suite runs in float32, where the `.to(x.dtype)` casts
    inside the sharded norms are no-ops, and so the precision those casts cost is invisible to all
    of them. What can fairly be asked in bf16 is that the replacement rounds no worse than the
    operator it replaces. Both are measured against `gold` - the float32 result rounded once, the
    best the narrow dtype can hold - and the sharded path is allowed `slack` times the stock op's
    own error, floored at one quantum of the dtype so an exact stock answer does not demand one.
    """
    detail = ""
    ok = torch.ones(1, dtype=torch.int64)
    if rank == 0:
        scale = gold.float().abs().max().clamp(min=1e-12)
        ours = ((actual.float() - gold.float()).abs().max() / scale).item()
        theirs = ((stock.float() - gold.float()).abs().max() / scale).item()
        allowed = max(theirs * slack, torch.finfo(actual.dtype).eps)
        if ours > allowed:
            detail = (
                f"{ours * 100:.3f}% of scale, against the stock operator's {theirs * 100:.3f}% "
                f"(allowed {allowed * 100:.3f}%, slack x{slack})"
            )
            ok.zero_()
    dist.broadcast(ok, src=0)
    dist.barrier()
    if ok.item() == 0:
        raise AssertionError(f"{what} rounds worse than the operator it replaces: {detail}")


def _free_port() -> int:
    """A port nothing is listening on, as of asking"""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def run_distributed(worker, world_size: int, args: tuple, master_port: int) -> None:
    """Spawn world_size ranks running worker(rank, *args); raises if any rank does

    Rank 0 opens the rendezvous socket, so a port taken between the fixture choosing it and rank 0
    binding it fails the test for a reason that has nothing to do with sharding. Retried on a
    fresh port, which is the only thing that can be done about it from here: no port can be held
    open for the ranks, since rank 0 has to bind it itself.
    """
    for attempt in range(_RENDEZVOUS_ATTEMPTS):
        try:
            spawn(worker, nprocs=world_size, args=(world_size, *args, master_port), join=True)
            return
        except ProcessRaisedException as raised:
            last = attempt == _RENDEZVOUS_ATTEMPTS - 1
            if last or "EADDRINUSE" not in str(raised):
                raise
            master_port = _free_port()
