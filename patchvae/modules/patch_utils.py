import torch
import torch.nn as nn
import torch.distributed as dist

class Patchify(nn.Module):
    def __init__(self):
        super().__init__()
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
    
    def forward(self, hidden_state):
        height = hidden_state.shape[2]
        start_idx = (height + self.world_size - 1) // self.world_size * self.rank
        end_idx = min((height + self.world_size - 1) // self.world_size * (self.rank + 1), height)

        return hidden_state[:, :, start_idx: end_idx, :].clone()


class DePatchify(nn.Module):
    def __init__(self):
        super().__init__()
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
    
    def forward(self, patch_hidden_state):
        patch_height_list = [torch.empty([1], dtype=torch.int64, device=f"cuda:{self.rank}") for _ in range(self.world_size)]
        dist.all_gather(
            patch_height_list, 
            torch.tensor(
                [patch_hidden_state.shape[2]], 
                dtype=torch.int64, 
                device=f"cuda:{self.rank}"
            )
        )
        patch_hidden_state_list = [
            torch.empty(
                [1, patch_hidden_state.shape[1], patch_height_list[i].item(), patch_hidden_state.shape[-1]], 
                dtype=patch_hidden_state.dtype,
                device=f"cuda:{self.rank}"
            ) for i in range(self.world_size)
        ]
        dist.all_gather(
            patch_hidden_state_list, 
            patch_hidden_state.contiguous()
        )
        return torch.cat(patch_hidden_state_list, dim=2)

        