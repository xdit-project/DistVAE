"""Distributed process lifecycle and exact collective accounting."""

import importlib
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import timedelta

import torch
import torch.distributed as dist


def exception_record(error, rank):
    """Represent a local exception without losing its originating rank or type."""
    preserved = getattr(error, "rank_error", None)
    if preserved is not None:
        return preserved
    return {"type": type(error).__name__, "message": str(error), "rank": int(rank)}


def gather_rank_errors(local_error, runtime):
    """Collect one optional error from every rank in collective order."""
    failures = [None] * runtime.world_size
    dist.all_gather_object(failures, local_error, group=runtime.group)
    return failures


DESYNCHRONIZED = "DesynchronizedRanks"


def _is_failure_record(failure):
    return isinstance(failure, dict) and "rank" in failure


def ranks_diverged(failures):
    """Return whether the group can still be trusted to run collectives together.

    A cell that fails on EVERY rank leaves the group in step - an out-of-memory on a decode
    nobody can fit is the ordinary way a matrix run reports "this shape does not fit", and the
    next cell measures normally afterwards. A cell that fails on SOME ranks does not: the ranks
    that failed stopped issuing collectives while the others carried on, so from that point the
    two are matching up different calls and nothing the group produces means anything.

    The distinction is the whole point of the check. Stopping on the first kind would end most
    sweeps at their first unsharded cell; continuing through the second kind produces numbers
    that look ordinary and are not.
    """
    if any(
        failure is not None and not _is_failure_record(failure) for failure in failures
    ):
        return True
    reported = [failure is not None for failure in failures]
    return any(reported) and not all(reported)


def aggregate_rank_errors(failures):
    """Combine rank errors while preserving each original failure record.

    An entry that is not a failure record is reported as one rather than raising. Once the ranks
    diverge, the gather that collects the errors pairs with whatever call the other ranks are
    still inside, so what comes back can be another call site's payload - and reaching into it
    for a failure record used to raise an AttributeError that both hid the failure underneath
    and took the rest of the run down with it.
    """
    details = []
    for rank, failure in enumerate(failures):
        if failure is None:
            continue
        if not _is_failure_record(failure):
            details.append(
                {
                    "type": DESYNCHRONIZED,
                    "message": (
                        "the ranks are no longer running the same sequence of "
                        f"collectives: the failure gathered for rank {rank} came back "
                        f"as {type(failure).__name__}, which is another call's payload "
                        "rather than a failure record"
                    ),
                    "rank": rank,
                }
            )
            continue
        nested = failure.get("failures")
        details.extend(nested if nested is not None else [failure])
    by_rank = {}
    for failure in details:
        by_rank.setdefault(failure["rank"], failure)
    details = list(by_rank.values())
    if not details:
        return None
    return {
        **details[0],
        "failed_ranks": [failure["rank"] for failure in details],
        "failures": details,
    }


class RankError(RuntimeError):
    """Propagate an aggregated rank failure without wrapping its identity."""

    def __init__(self, error, context):
        self.rank_error = error
        super().__init__(
            f"{context} failed on rank {error['rank']}: "
            f"{error['type']}: {error['message']}"
        )


def accelerator_backend():
    """Return the available accelerator API and its distributed backend."""
    if torch.cuda.is_available():
        return "cuda", torch.cuda, "nccl"
    try:
        importlib.import_module("torch_musa")
    except ModuleNotFoundError as error:
        raise RuntimeError("measurement requires CUDA or MUSA") from error
    musa = getattr(torch, "musa", None)
    if musa is None or not musa.is_available():
        raise RuntimeError("measurement requires CUDA or MUSA")
    return "musa", musa, "mccl"


class CollectiveLog:
    """Count collective calls and tensor bytes by operation and call site."""

    WRAPPED = (
        "all_reduce",
        "all_gather",
        "all_gather_into_tensor",
        "broadcast",
        "isend",
        "irecv",
        "recv",
        "send",
        "barrier",
        "batch_isend_irecv",
    )

    def __init__(self):
        self.enabled = False
        self.by_call = defaultdict(lambda: {"calls": 0, "bytes": 0})
        self.by_site = defaultdict(lambda: {"calls": 0, "bytes": 0})
        self._originals = {}

    @staticmethod
    def _nbytes(args):
        total = 0
        for arg in args:
            values = arg if isinstance(arg, (list, tuple)) else (arg,)
            total += sum(
                value.numel() * value.element_size()
                for value in values
                if isinstance(value, torch.Tensor)
            )
        return total

    def _wrap(self, name, original):
        def wrapper(*args, **kwargs):
            if self.enabled:
                frame = sys._getframe(1)
                site = f"{os.path.basename(frame.f_code.co_filename)}:{frame.f_lineno}"
                nested = (
                    os.path.basename(frame.f_code.co_filename) == "distributed_c10d.py"
                )
                label = f"{name} (batched)" if nested else name
                size = self._nbytes(args)
                for entry in (self.by_call[label], self.by_site[f"{name} @ {site}"]):
                    entry["calls"] += 1
                    entry["bytes"] += size
            return original(*args, **kwargs)

        return wrapper

    def install(self):
        from torch.distributed import distributed_c10d

        for name in self.WRAPPED:
            original = getattr(dist, name, None)
            if original is None:
                continue
            self._originals[name] = original
            wrapper = self._wrap(name, original)
            setattr(dist, name, wrapper)
            if getattr(distributed_c10d, name, None) is original:
                setattr(distributed_c10d, name, wrapper)

    def uninstall(self):
        from torch.distributed import distributed_c10d

        for name, original in self._originals.items():
            setattr(dist, name, original)
            setattr(distributed_c10d, name, original)
        self._originals.clear()

    def reset(self):
        self.by_call.clear()
        self.by_site.clear()

    def report(self):
        return {
            "by_call": {
                name: dict(value) for name, value in sorted(self.by_call.items())
            },
            "by_site": {
                name: dict(value)
                for name, value in sorted(
                    self.by_site.items(), key=lambda item: -item[1]["calls"]
                )
            },
            "total_calls": sum(
                value["calls"]
                for name, value in self.by_call.items()
                if "(batched)" not in name
            ),
            "total_bytes": sum(value["bytes"] for value in self.by_call.values()),
        }


def across_ranks(by_call, world_size, group):
    """Report operation counts for every rank and the busiest rank."""
    gathered = [None] * world_size
    counts = {name: entry["calls"] for name, entry in by_call.items()}
    dist.all_gather_object(gathered, counts, group=group)

    def total(values):
        return sum(calls for name, calls in values.items() if "(batched)" not in name)

    names = sorted({name for values in gathered for name in values})
    return {
        "by_call_max": {
            name: max(values.get(name, 0) for values in gathered) for name in names
        },
        "total_calls_max": max(total(values) for values in gathered),
        "total_calls_by_rank": [total(values) for values in gathered],
    }


@dataclass
class Runtime:
    rank: int
    world_size: int
    local_rank: int
    device: torch.device
    group: object
    log: CollectiveLog
    device_api: object

    @classmethod
    def start(cls, timeout_min):
        """Initialize the accelerator process group used by measurements."""
        device_type, device_api, backend = accelerator_backend()
        missing = [name for name in ("RANK", "WORLD_SIZE") if name not in os.environ]
        if missing:
            raise RuntimeError(
                "measurement requires a distributed launch environment; missing "
                + ", ".join(missing)
            )
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        device_api.set_device(local_rank)
        device = torch.device(device_type, local_rank)
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            timeout=timedelta(minutes=timeout_min),
        )
        log = CollectiveLog()
        log.install()
        group = dist.group.WORLD
        dist.all_reduce(torch.zeros(1, device=device), group=group)
        return cls(rank, world_size, local_rank, device, group, log, device_api)

    def close(self):
        """Synchronize, restore wrapped calls, and destroy the process group."""
        try:
            dist.barrier(group=self.group)
        finally:
            self.log.uninstall()
            dist.destroy_process_group()
