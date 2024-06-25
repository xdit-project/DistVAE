from distvae.modules.patch_utils import Patchify, DePatchify
from distvae.modules.adapters.layers.norm_adapters import GroupNormAdapter
from torch.nn import GroupNorm

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
    parser.add_argument(
        "--channels",
        type=int,
        default=512,
        help="The width of image",
    )
    args = parser.parse_args() 
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.device('cuda', rank)

    norm = GroupNorm(num_groups=32, num_channels=args.channels, eps=1e-6, affine=True).to(f"cuda:{rank}")
    patch_norm = GroupNormAdapter(norm).to(f"cuda:{rank}")

    hidden_state = torch.randn(1, args.channels, args.height, args.width, device=f"cuda:{rank}")

    result = norm(hidden_state)
    # if rank == 0:
        # print("result: ", result)

    patch = Patchify()
    depatch = DePatchify()
    patch_result = patch_norm(patch(hidden_state))
    # print("patch_res:", rank, patch_result)
    patch_result = depatch(patch_result)


    if rank == 0:
        assert torch.allclose(result, patch_result), "two hidden states are not equal"


if __name__ == "__main__":
    main()