from distvae.modules.patch_utils import Patchify, DePatchify
from distvae.modules.adapters.resnet_adapters import ResnetBlock2DAdapter
from torch.nn import GroupNorm

from diffusers.models.resnet import ResnetBlock2D

import torch
import random
import argparse
import torch.distributed as dist
from torch import nn

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def main():
    set_seed()
    torch.backends.cudnn.deterministic = True
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="The height of image",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="The width of image",
    )
    args = parser.parse_args() 
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.device('cuda', rank)

    resnet = ResnetBlock2D(
        in_channels=64,
        out_channels=32,
        temb_channels=None,
        eps=1e-6,
        groups=4,
        dropout=0.0,
        time_embedding_norm="default",
        non_linearity="swish",
        output_scale_factor=1.0,
        pre_norm=True,
    ).to(f"cuda:{rank}")
    patch_resnet = ResnetBlock2DAdapter(resnet).to(f"cuda:{rank}")

    hidden_state = torch.randn(1, 64, args.height, args.width, device=f"cuda:{rank}")

    result = resnet(hidden_state, None)
    # if rank == 0:
        # print("result: ", result)

    patch = Patchify()
    depatch = DePatchify()
    patch_result = patch_resnet(patch(hidden_state))
    # print("patch_res:", rank, patch_result)
    patch_result = depatch(patch_result)


    if rank == 0:
        assert torch.allclose(result, patch_result, atol=1e-2), "two hidden states are not equal"


if __name__ == "__main__":
    main()