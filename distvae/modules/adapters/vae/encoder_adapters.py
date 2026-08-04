from typing import Optional

import torch
import torch.nn as nn
from torch.distributed import ProcessGroup

from distvae.modules.adapters.layers.conv_adapters import WanCausalConv3dAdapter
from distvae.modules.adapters.midblock_adapters import WanMidBlockAdapter
from distvae.modules.adapters.downsampling_adapters import (
    WanResidualDownBlockAdapter,
    WanResampleDownAdapter
)
from distvae.modules.adapters.resnet_adapters import WanResidualBlockAdapter
from distvae.modules.adapters.layers.attn_adapters import WanAttentionBlockAdapter
from distvae.modules.patch_utils import Patchify, DePatchify
from distvae.utils import DistributedEnv

from diffusers.models.autoencoders.autoencoder_kl_wan import (
    WanResidualDownBlock,
    WanResidualBlock,
    WanResample,
    WanAttentionBlock,
)

class WanEncoderAdapter(nn.Module):
    def __init__(
        self,
        encoder,
        vae_group: ProcessGroup = None,
        *,
        vae_scale_factor: int = 8,
        conv_block_size = 0,
        patch_dim: int = -2,
    ):
        super().__init__()
        if patch_dim == -3:
            raise ValueError("WanEncoderAdapter does not support patch_dim F (-3); use H (-2) or W (-1).")

        DistributedEnv.initialize(vae_group)
        self.patch_dim = patch_dim
        DistributedEnv.set_patch_dim(patch_dim)
        self.vae_scale_factor = vae_scale_factor
        self.encoder = encoder

        # Patch the conv_in layer
        self.encoder.conv_in = WanCausalConv3dAdapter(
            encoder.conv_in,
            block_size=conv_block_size,
            patch_dim=patch_dim,
            use_uniform_patch=False,
        )
        # Patch the down_blocks
        down_blocks = []
        for i, down_block in enumerate(encoder.down_blocks):
            if isinstance(down_block, WanResidualDownBlock):
                # Wan2.2 style: wrapped in WanResidualDownBlock
                down_blocks.append(
                    WanResidualDownBlockAdapter(
                        down_block,
                        conv_block_size=conv_block_size,
                        patch_dim=patch_dim,
                        use_uniform_patch=False,
                    )
                )
            elif isinstance(down_block, WanResidualBlock):
                # Wan2.1 style: individual residual block
                down_blocks.append(
                    WanResidualBlockAdapter(
                        down_block,
                        conv_block_size=conv_block_size,
                        patch_dim=patch_dim,
                        use_uniform_patch=False,
                    )
                )
            elif isinstance(down_block, WanResample):
                # Wan2.1 style: individual downsample block
                down_blocks.append(
                    WanResampleDownAdapter(
                        down_block,
                        conv_block_size=conv_block_size,
                        patch_dim=patch_dim,
                        use_uniform_patch=False,
                    )
                )
            elif isinstance(down_block, WanAttentionBlock):
                # Attention blocks need to see full spatial context, so wrap with adapter
                down_blocks.append(
                    WanAttentionBlockAdapter(down_block, patch_dim=patch_dim)
                )
            else:
                # Unknown block type - keep as-is and log warning
                import warnings
                warnings.warn(
                    f"Unsupported down_block type {type(down_block).__name__} at index {i} in encoder, "
                    f"keeping original. This may cause issues with parallel VAE."
                )
                down_blocks.append(down_block)
        self.encoder.down_blocks = nn.ModuleList(down_blocks)
        # Patch the mid_block
        self.encoder.mid_block = WanMidBlockAdapter(
            encoder.mid_block,
            conv_block_size=conv_block_size,
            patch_dim=patch_dim,
            use_uniform_patch=False,
        )
        # Patch the conv_out layer
        self.encoder.conv_out = WanCausalConv3dAdapter(
            encoder.conv_out,
            block_size=conv_block_size,
            patch_dim=patch_dim,
            use_uniform_patch=False,
        )
        # Each band is a whole multiple of what the encoder narrows by, so it starts on the grid
        # the strided convolutions step along and the latent rows it produces are its own.
        self.patchify = Patchify(patch_dim=patch_dim, scale_factor=vae_scale_factor)
        self.depatchify = DePatchify(patch_dim=patch_dim)

    def _forward(
        self,
        sample: torch.FloatTensor,
        feat_cache: Optional[torch.FloatTensor] = None,
        feat_idx: Optional[int] = 0,
        patchify: bool = True,
    ):
        """Internal forward with optional patchify."""
        if patchify:
            sample = self.patchify(sample)
        return self.depatchify(self.encoder(sample, feat_cache=feat_cache, feat_idx=feat_idx))

    def forward(
        self,
        sample: torch.FloatTensor,
        feat_cache: Optional[torch.FloatTensor] = None,
        feat_idx: Optional[int] = 0,
        patchify: bool = True,
    ):
        """
        Forward pass through the encoder.

        Args:
            sample: Input tensor to encode
            feat_cache: Optional feature cache for temporal consistency
            feat_idx: Feature index for caching
            patchify: Whether to apply patchify/depatchify (default: True)

        Returns:
            Encoded latent tensor
        """
        return self._forward(sample, feat_cache, feat_idx, patchify)
