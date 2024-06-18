from diffusers.models.autoencoders.vae import Decoder
from diffusers.models.upsampling import Upsample2D

from patchvae.models.vae import PatchDecoder
from patchvae.modules.adapters.upsampling_adapters import Upsample2DAdapter
from patchvae.modules.adapters.vae.decoder_adapters import DecoderAdapter

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
        default=128,
        help="The height of image",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=128,
        help="The width of image",
    )
    args = parser.parse_args() 
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.device('cuda', rank)

    decoder = Decoder(
        in_channels=4, 
        out_channels=3, 
        up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"],
        block_out_channels=(128, 256, 512, 512),
        layers_per_block=2,
        norm_num_groups=32,
        act_fn="silu",
    ).to(f"cuda:{rank}")
    patch_decoder = DecoderAdapter(decoder).to(f"cuda:{rank}")
    if rank == 0:
        print("decoder: ", decoder)
        print("patch decoder: ", patch_decoder)
    hidden_state = torch.randn(1, 4, args.height, args.width, device=f"cuda:{rank}")

    result = decoder(hidden_state)

    patch_result = patch_decoder(hidden_state)

    print("result shape: ", patch_result.shape)
    if rank == 0:
        assert torch.allclose(result, patch_result, atol=1e-2), "two hidden states are not equal"


if __name__ == "__main__":
    main()