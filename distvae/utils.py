import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple


def cache_cursor(feat_idx: Optional[List[int]]) -> List[int]:
    """The caller's position in the feature cache, or a fresh one at the start of it

    The causal video decoders walk their feature cache with a one-element list, advancing it as
    each layer takes its slot. That cursor cannot be a default argument: Python binds one list
    per function at definition, so every call omitting it would share the same one, and a second
    decode would carry on reading from wherever the first one stopped. What that gives is not an
    error but a video conditioned on the tail of the previous decode.
    """
    return [0] if feat_idx is None else feat_idx


def normalize_patch_dim(patch_dim: int, ndim: int, *, spatial_only: bool = False) -> int:
    """Return a canonical negative patch axis after validating it for the tensor rank."""
    if not isinstance(patch_dim, int) or isinstance(patch_dim, bool):
        raise ValueError(f"patch_dim must be an integer, got {patch_dim!r}")
    if ndim not in (4, 5):
        raise ValueError(f"patch_dim validation supports 4D or 5D tensors, got {ndim}D")
    positive = patch_dim if patch_dim >= 0 else ndim + patch_dim
    if positive < 2 or positive >= ndim:
        raise ValueError(f"patch_dim {patch_dim} is not a data axis of a {ndim}D tensor")
    if spatial_only and ndim == 5 and positive == 2:
        raise ValueError(
            f"patch_dim {patch_dim} selects the frame axis; only H (-2 or 3) and "
            "W (-1 or 4) are supported"
        )
    return positive - ndim


@dataclass(frozen=True)
class ParallelContext:
    """Immutable distributed settings owned by one adapted VAE."""

    group: Optional[ProcessGroup]
    rank: int
    world_size: int
    patch_dim: int
    global_ranks: Tuple[int, ...] = ()

    def global_rank(self, group_rank: int) -> int:
        if self.global_ranks:
            return self.global_ranks[group_rank]
        if self.world_size == 1:
            return dist.get_rank() if dist.is_initialized() else 0
        return dist.get_global_rank(self.group, group_rank)


def parallel_context(
    vae_group: Optional[ProcessGroup], patch_dim: int, *, ndim: int
) -> ParallelContext:
    """Capture one adapter's group and axis without changing process-global state."""
    group = dist.group.WORLD if vae_group is None else vae_group
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    global_ranks = tuple(dist.get_global_rank(group, one) for one in range(world_size))
    return ParallelContext(
        group=group,
        rank=rank,
        world_size=world_size,
        patch_dim=normalize_patch_dim(patch_dim, ndim, spatial_only=True),
        global_ranks=global_ranks,
    )


class DistributedEnv:
    @classmethod
    def get_local_rank(cls) -> int:
        return int(os.environ.get("LOCAL_RANK", 0))

    @classmethod
    def get_device(cls) -> torch.device:
        if torch.cuda.is_available():
            return torch.device(f"cuda:{cls.get_local_rank()}")
        elif hasattr(torch, "musa") and torch.musa.is_available():
            return torch.device(f"musa:{cls.get_local_rank()}")
        else:
            return torch.device("cpu")

    @classmethod
    def get_device_type(cls) -> str:
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch, "musa") and torch.musa.is_available():
            return "musa"
        else:
            return "cpu"

    @classmethod
    def get_torch_distributed_backend(cls) -> str:
        if torch.cuda.is_available():
            return "nccl"
        elif hasattr(torch, "musa") and torch.musa.is_available():
            return "mccl"
        else:
            # Sharding is correctness-testable without an accelerator, and gloo is the only
            # backend that gets there. Raising instead would make every distributed entry point
            # unreachable on a CPU-only machine, tests included.
            return "gloo"

    @classmethod
    def record_memory_history(cls):
        device_type = cls.get_device_type()
        if device_type == "cuda":
            torch.cuda.memory._record_memory_history(enabled=None)
        elif device_type == "musa":
            torch.musa.memory._record_memory_history(enabled=None)
        else:
            print(f"[Warning] Unknown device type: {device_type}, memory history not recorded.")

    @classmethod
    def get_peak_memory(cls, device):
        device_type = cls.get_device_type()
        if device_type == "cuda":
            return torch.cuda.max_memory_allocated(device)
        elif device_type == "musa":
            return torch.musa.max_memory_allocated(device)
        else:
            print(f"[Warning] Unknown device type: {device_type}, peak memory not available.")
            return None