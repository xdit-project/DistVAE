from dataclasses import dataclass
from typing import Optional, Tuple

import torch.nn as nn
from torch.distributed import ProcessGroup

from distvae.modules.adapters.layers.norm_adapters import GroupNormAdapter
from distvae.modules.patch_utils import DePatchify, Patchify, widest_halo
from distvae.utils import (
    ParallelContext,
    normalize_patch_dim,
    parallel_context,
)


@dataclass(frozen=True)
class CausalVAEAdapterSetup:
    """Immutable setup shared by causal encoder and decoder halves."""

    adapter: str
    conv_adapter: object
    block_adapters: Tuple[Tuple[Optional[type], object], ...]
    conv_block_size: object
    patch_dim: int
    parallel_context: ParallelContext

    @classmethod
    def create(
        cls,
        *,
        adapter,
        conv_adapter,
        block_adapters,
        conv_block_size,
        patch_dim,
        vae_group: ProcessGroup,
    ):
        patch_dim = normalize_patch_dim(patch_dim, 5, spatial_only=True)
        return cls(
            adapter=adapter,
            conv_adapter=conv_adapter,
            block_adapters=block_adapters,
            conv_block_size=conv_block_size,
            patch_dim=patch_dim,
            parallel_context=parallel_context(vae_group, patch_dim, ndim=5),
        )

    @property
    def options(self):
        return {"parallel_context": self.parallel_context}

    def adapt_convolution(self, convolution):
        return self.conv_adapter(
            convolution, block_size=self.conv_block_size, **self.options
        )

    def adapt_blocks(self, blocks, kind):
        return nn.ModuleList([self.adapt_block(one, kind) for one in blocks])

    def adapt_block(self, block, kind):
        for block_type, block_adapter in self.block_adapters:
            if block_type is not None and isinstance(block, block_type):
                return block_adapter(
                    block, conv_block_size=self.conv_block_size, **self.options
                )
        handled = ", ".join(
            block_type.__name__
            for block_type, _ in self.block_adapters
            if block_type is not None
        )
        raise TypeError(
            f"{self.adapter} cannot shard a {kind} block of type "
            f"{type(block).__name__}. It handles "
            f"{handled or 'no block type the installed diffusers provides'}."
        )

    def adapt_group_norm(self, norm):
        if isinstance(norm, nn.GroupNorm):
            return GroupNormAdapter(norm, **self.options)
        return norm

    def patchers(self, module, scale_factor=1):
        return (
            Patchify(
                scale_factor=scale_factor,
                halo=widest_halo(module),
                **self.options,
            ),
            DePatchify(**self.options),
        )
