from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn

from distvae.utils import DistributedEnv

class WanAttentionBlockAdapter(torch.nn.Module):
    """Runs attention on the full sequence by gathering along the patch dim, then narrows back to the local patch.

    Supports unequal patch sizes across ranks (e.g. after Patchify without padding).
    """

    def __init__(
        self,
        module: nn.Module,
        patch_dim: int = -2,
    ) -> None:
        super().__init__()
        self.module = module
        self.patch_dim = patch_dim

    def forward(self, hidden_states: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        patch_dim = self.patch_dim if self.patch_dim >= 0 else hidden_states.ndim + self.patch_dim
        rank = DistributedEnv.get_rank_in_vae_group()
        world_size = DistributedEnv.get_group_world_size()
        device = hidden_states.device

        # Gather chunk sizes from all ranks
        size_list = [torch.empty(1, dtype=torch.int64, device=device) for _ in range(world_size)]
        dist.all_gather(
            size_list,
            torch.tensor([hidden_states.shape[patch_dim]], dtype=torch.int64, device=device),
            group=DistributedEnv.get_vae_group(),
        )
        chunk_sizes = [size_list[i].item() for i in range(world_size)]

        base_shape = list(hidden_states.shape)
        gathered_tensors = []
        for i in range(world_size):
            shape = base_shape.copy()
            shape[patch_dim] = chunk_sizes[i]
            gathered_tensors.append(torch.empty(shape, dtype=hidden_states.dtype, device=device))
        dist.all_gather(gathered_tensors, hidden_states.contiguous(), group=DistributedEnv.get_vae_group())

        combined_tensor = torch.cat(gathered_tensors, dim=patch_dim)
        forward_output = self.module(combined_tensor, *args, **kwargs)
        start_idx = sum(chunk_sizes[:rank])
        local_output = torch.narrow(
            forward_output, patch_dim, start_idx, chunk_sizes[rank]
        )
        return local_output
