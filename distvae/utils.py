import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup
import os
from typing import List, Optional

try:
    import torch_musa
except ModuleNotFoundError:
    pass


def cache_cursor(feat_idx: Optional[List[int]]) -> List[int]:
    """The caller's position in the feature cache, or a fresh one at the start of it

    The causal video decoders walk their feature cache with a one-element list, advancing it as
    each layer takes its slot. That cursor cannot be a default argument: Python binds one list
    per function at definition, so every call omitting it would share the same one, and a second
    decode would carry on reading from wherever the first one stopped. What that gives is not an
    error but a video conditioned on the tail of the previous decode.
    """
    return [0] if feat_idx is None else feat_idx


class DistributedEnv:
    _vae_group = None
    _local_rank = None
    _world_size = None  # 添加新的类变量
    _patch_dim = -2  # -3=F, -2=H, -1=W; same for 2D/3D

    @classmethod
    def initialize(cls, vae_group: ProcessGroup):
        if vae_group is None:
            cls._vae_group = dist.group.WORLD
        else:
            cls._vae_group = vae_group
        cls._local_rank = int(os.environ.get('LOCAL_RANK', 0)) # FIXME: in ray all local_rank is 0
        cls._rank_mapping = None
        cls._init_rank_mapping()
    
    @classmethod
    def get_vae_group(cls) -> ProcessGroup:
        if cls._vae_group is None:
            raise RuntimeError("DistributedEnv not initialized. Call initialize() first.")
        return cls._vae_group

    @classmethod
    def get_global_rank(cls) -> int:
        return dist.get_rank()
    
    @classmethod
    def _init_rank_mapping(cls):
        """Initialize the mapping between group ranks and global ranks"""
        if cls._rank_mapping is not None:
            return
        # The only member of a one-rank group is this rank, which it can answer without asking.
        # Worth the branch because initialize() clears the mapping and every adapter constructor
        # calls it, so the gather is paid once per adapter rather than once per model.
        if cls.get_group_world_size() == 1:
            cls._rank_mapping = [cls.get_global_rank()]
            return
        # Get all ranks in the group
        ranks = [None] * cls.get_group_world_size()
        dist.all_gather_object(ranks, cls.get_global_rank(), group=cls.get_vae_group())
        cls._rank_mapping = ranks

    @classmethod
    def get_global_rank_from_group_rank(cls, group_rank: int) -> int:
        """Convert a rank in VAE group to global rank using cached mapping.
        
        Args:
            group_rank: The rank in VAE group
            
        Returns:
            The corresponding global rank
            
        Raises:
            RuntimeError: If the group_rank is invalid
        """
        if cls._rank_mapping is None:
            cls._init_rank_mapping()
            
        if group_rank < 0 or group_rank >= cls.get_group_world_size():
            raise RuntimeError(f"Invalid group rank: {group_rank}. Must be in range [0, {cls.get_group_world_size()-1}]")
            
        return cls._rank_mapping[group_rank]
    
    @classmethod
    def get_rank_in_vae_group(cls) -> int:
        return dist.get_rank(cls.get_vae_group())

    @classmethod
    def get_group_world_size(cls) -> int:
        return dist.get_world_size(cls.get_vae_group())

    @classmethod
    def set_patch_dim(cls, dim: int):
        cls._patch_dim = dim

    @classmethod
    def get_patch_dim(cls) -> int:
        return cls._patch_dim

    @classmethod
    def get_local_rank(cls) -> int:
        return cls._local_rank

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