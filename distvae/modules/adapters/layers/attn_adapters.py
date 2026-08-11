from typing import Any

import torch
import torch.nn as nn

from distvae.modules.patch_utils import gather_patches
from distvae.utils import ParallelContext, normalize_patch_dim


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
        parallel_context: ParallelContext = None,
    ) -> None:
        super().__init__()
        if not isinstance(parallel_context, ParallelContext):
            raise TypeError("GatheredAttentionAdapter requires a ParallelContext")
        self.module = module
        self.parallel_context = parallel_context
        self.patch_dim = parallel_context.patch_dim

    def forward(self, hidden_states: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        patch_dim = hidden_states.ndim + normalize_patch_dim(
            self.patch_dim, hidden_states.ndim, spatial_only=True
        )
        rank = self.parallel_context.rank

        patches, sizes = gather_patches(hidden_states, self.parallel_context)
        whole = self.module(torch.cat(patches, dim=patch_dim), *args, **kwargs)
        return torch.narrow(whole, patch_dim, sum(sizes[:rank]), sizes[rank])


# The name this was introduced under, before other families turned out to need the same thing.
WanAttentionBlockAdapter = GatheredAttentionAdapter