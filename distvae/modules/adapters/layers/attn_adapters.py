from typing import Any

import torch
import torch.nn as nn

from distvae.modules.patch_utils import gather_patches
from distvae.utils import DistributedEnv


class GatheredAttentionAdapter(torch.nn.Module):
    """Runs attention on the full sequence by gathering along the patch dim, then narrows back to the local patch.

    Attention is the one layer in a VAE that relates every position to every other, so unlike a
    convolution it cannot be satisfied with a halo. Nothing here reads the wrapped module, only
    calls it, so this covers whichever attention block a family happens to use.

    Patches need not be the same size across ranks: the gather below pads for transport only.
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

        patches, sizes = gather_patches(hidden_states, patch_dim)
        whole = self.module(torch.cat(patches, dim=patch_dim), *args, **kwargs)
        return torch.narrow(whole, patch_dim, sum(sizes[:rank]), sizes[rank])


# The name this was introduced under, before other families turned out to need the same thing.
WanAttentionBlockAdapter = GatheredAttentionAdapter