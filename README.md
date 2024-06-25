# DistVAE: A patch parallelism distributed VAE implement for high resolution generation

By providing a set of adapter interfaces, this project allows users to quickly convert vae-related implementations in the diffusers library into parallel versions on multiple gpu's, enabling non-intrusive parallelisation of the vae portion of an existing model, thus reducing the memory footprint of the image generation process, and avoiding vae-induced memory spikes.

## Installation

``` bash
python setup.py install
```

## Usage

Refering to the file in `test/` directory. In general, you only need to use the corresponding adapter for the diffusers module to make it work on multiple gpu in parallel.

As an example, we can transform an initialised vae decoder into a parallel versions:


``` python
from diffusers.models.autoencoders.vae import Decoder
from distvae.modules.adapters.vae.decoder_adapters import DecoderAdapter

import torch
import random
import torch.distributed as dist

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def main():
    # init
    set_seed()
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.device('cuda', rank)

    # input 
    hidden_state = torch.randn(1, 4, 128, 128, device=f"cuda:{rank}")
    # create vae.decoder instance
    decoder = Decoder(
        in_channels=4, out_channels=3, 
        up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"],
        block_out_channels=(128, 256, 512, 512), layers_per_block=2,
        norm_num_groups=32, act_fn="silu",
    ).to(f"cuda:{rank}")
    # transform vae.decoder to distvae.decoder
    patch_decoder = DecoderAdapter(decoder).to(f"cuda:{rank}")
    # forward
    result = decoder(hidden_state)
    patch_result = patch_decoder(hidden_state)

    print("result shape: ", patch_result.shape)
    if rank == 0:
        assert torch.allclose(result, patch_result, atol=1e-2), "two hidden states are not equal"

if __name__ == "__main__":
    main()
```