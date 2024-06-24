from diffusers.models.autoencoders.vae import Decoder
from patchvae.modules.adapters.vae.decoder_adapters import DecoderAdapter

import time
import torch
import random
import argparse
import torch.distributed as dist

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

@torch.no_grad()
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
    torch.device('cuda', rank)
    # input 
    # create vae.decoder instance
    decoder = Decoder(
        in_channels=4, 
        out_channels=3, 
        up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"],
        block_out_channels=(128, 256, 512, 512),
        layers_per_block=2,
        norm_num_groups=32,
        act_fn="silu",
    ).to(f"cuda:{rank}")
    # transform vae.decoder to patchvae.decoder
    patch_decoder = DecoderAdapter(decoder).to(f"cuda:{rank}")
    # forward
    result = decoder(hidden_state)
    torch.cuda.memory._record_memory_history(enabled=None)
    for width in [512, 1536, 1024, 2048, 2560, 3072]:
        for height in [512, 1024, 1536, 2048, 2560, 3072]:
            torch.cuda.reset_max_memory_allocated(device=f"cuda:{rank}")
            hidden_state = torch.randn(1, 4, height // 8, width // 8, device=f"cuda:{rank}")
            start_time = time.time()
            patch_result = decoder(hidden_state)
            end_time = time.time()
            peak_memory = torch.cuda.max_memory_allocated(device=f"cuda:{rank}")
            if rank == 0:
                print(f"resolution: {height}x{width}, time: {end_time - start_time} sec, peak memory: {peak_memory / 2 ** 30} GB")
                assert torch.allclose(result, patch_result, atol=1e-2), "two hidden states are not equal"
    # if rank == 0:


if __name__ == "__main__":
    main()