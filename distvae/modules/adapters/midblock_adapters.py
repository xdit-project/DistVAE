import torch.nn as nn
from diffusers.models.autoencoders.autoencoder_kl_wan import WanMidBlock

from distvae.models.layers.wan.ringattentionblock import WanRingAttentionBlock
from distvae.modules.adapters.resnet_adapters import WanResidualBlockAdapter

class WanMidBlockAdapter(nn.Module):
    def __init__(
        self,
        wan_mid_block: WanMidBlock,
        conv_block_size = 0,
        patch_dim: int = -2,
        use_uniform_patch: bool = False,
    ):
        super().__init__()

        assert isinstance(wan_mid_block, WanMidBlock), "WanMidBlockAdapter does not support mid block except WanMidBlock"
        self.mid_block = wan_mid_block
        self.mid_block.resnets = nn.ModuleList([
            WanResidualBlockAdapter(
                resnet,
                conv_block_size=conv_block_size,
                patch_dim=patch_dim,
                use_uniform_patch=use_uniform_patch,
            ) for resnet in wan_mid_block.resnets
        ])
        self.mid_block.attentions = nn.ModuleList([
            WanRingAttentionBlock(attn, patch_dim=patch_dim)
            for attn in wan_mid_block.attentions
        ])

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        return self.mid_block(x, feat_cache=feat_cache, feat_idx=feat_idx)  
