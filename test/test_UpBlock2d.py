from distvae.modules.adapters.unets.unet_2d_blocks_adapters import UpDecoderBlock2DAdapter, UpDecoderBlock2D
from distvae.modules.patch_utils import Patchify, DePatchify

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

    up_block = UpDecoderBlock2D(num_layers = 3, in_channels=256, out_channels=128).to(f"cuda:{rank}")
    patch_up_block = UpDecoderBlock2DAdapter(up_block).to(f"cuda:{rank}")

    hidden_state = torch.randn(1, 256, args.height, args.width, device=f"cuda:{rank}")
    print("hidden state shape: ", hidden_state.shape)

    result = up_block(hidden_state)
    # if rank == 0:
        # print("result: ", result)

    patch = Patchify()
    depatch = DePatchify()
    patch_result = patch_up_block(patch(hidden_state))
    # print("patch_res:", rank, patch_result)
    patch_result = depatch(patch_result)
    print("result shape: ", patch_result.shape)


    if rank == 0:
        assert torch.allclose(result, patch_result, atol=1e-3), "two hidden states are not equal"


if __name__ == "__main__":
    main()