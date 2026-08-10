"""Distributed process lifecycle and exact collective accounting."""

import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import timedelta

import torch
import torch.distributed as dist


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

    @classmethod
    def start(cls, timeout_min):
        """Initialize the accelerator process group used by measurements."""
        if not torch.cuda.is_available():
            raise RuntimeError("measurement requires CUDA; use --describe-only on CPU")
        missing = [name for name in ("RANK", "WORLD_SIZE") if name not in os.environ]
        if missing:
            raise RuntimeError(
                "measurement requires a distributed launch environment; missing "
                + ", ".join(missing)
            )
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=timedelta(minutes=timeout_min),
        )
        log = CollectiveLog()
        log.install()
        group = dist.group.WORLD
        dist.all_reduce(torch.zeros(1, device=device), group=group)
        return cls(rank, world_size, local_rank, device, group, log)

    def close(self):
        """Synchronize, restore wrapped calls, and destroy the process group."""
        try:
            dist.barrier(group=self.group)
        finally:
            self.log.uninstall()
            dist.destroy_process_group()
