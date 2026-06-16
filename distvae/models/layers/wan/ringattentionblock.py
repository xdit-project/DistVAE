import os
import sys

import iris
import torch
import torch.nn as nn
import torch.nn.functional as F

from distvae.utils import DistributedEnv

iris_pkg_dir = os.path.dirname(iris.__file__)
sys.path.append(os.path.join(os.path.dirname(iris_pkg_dir), "examples", "32_ring_attention"))
from ring_attention_layer import RingAttention


class WanRingAttentionBlock(torch.nn.Module):
    """WanRingAttentionBlock is a wrapper for the RingAttention layer from the Iris package.
    """

    def __init__(
        self,
        module: nn.Module,
        patch_dim: int = -2,
    ) -> None:
        super().__init__()
        self.norm = module.norm
        self.to_qkv = module.to_qkv
        self.proj = module.proj

        self.shmem = iris.iris()
        self.attn = RingAttention(
            self.shmem,
            num_heads=1,
            head_dim=384,
            causal=False,
            scale=384**-0.5,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        batch_size, channels, time, height, width = x.size()

        x = x.permute(0, 2, 1, 3, 4).reshape(batch_size * time, channels, height, width)
        x = self.norm(x)

        # compute query, key, value
        qkv = self.to_qkv(x)
        qkv = qkv.reshape(batch_size * time, 1, channels * 3, -1)
        qkv = qkv.permute(0, 1, 3, 2).contiguous()
        q, k, v = qkv.chunk(3, dim=-1)

        # apply attention
        head_dim = 384
        seqlen_pad_size = (64 - (height * width) % 64) % 64
        q, k, v= (
            F.pad(
                t,
                (0, 0, 0, seqlen_pad_size, 0, 0, 0, 0),
                mode='constant',
                value=0,
            ).permute(2, 0, 1, 3).reshape(-1, batch_size * time, head_dim)
            for t in [q, k, v]
        )
        x = self.attn(q, k, v)
        x = torch.narrow(x, 0, 0, height * width)
        x = x.permute(1, 2, 0).reshape(batch_size * time, channels, height, width)

        # output projection
        x = self.proj(x)

        # Reshape back: [(b*t), c, h, w] -> [b, c, t, h, w]
        x = x.view(batch_size, time, channels, height, width)
        x = x.permute(0, 2, 1, 3, 4)

        return x + identity
