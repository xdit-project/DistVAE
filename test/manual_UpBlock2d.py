from distvae.modules.adapters.unets.unet_2d_blocks_adapters import UpDecoderBlock2DAdapter, UpDecoderBlock2D
from distvae.modules.patch_utils import Patchify, DePatchify
from distvae.utils import DistributedEnv

import torch
import random
import argparse
import torch.distributed as dist
from torch import nn
from torch.cuda import set_device, device_count
from torch.cuda import manual_seed as device_manual_seed
try:
    import torch_musa
    from torch_musa.core.device import set_device, device_count
    from torch_musa.core.random import manual_seed as device_manual_seed
except ModuleNotFoundError:
    pass

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    device_manual_seed(seed)

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
    backend = DistributedEnv.get_torch_distributed_backend()
    dist.init_process_group(backend=backend)
    device = torch.distributed.get_rank() % device_count()
    set_device(device)
    DistributedEnv.initialize(None)

    up_block = UpDecoderBlock2D(num_layers = 3, in_channels=256, out_channels=128).to(device)
    patch_up_block = UpDecoderBlock2DAdapter(up_block).to(device)

    hidden_state = torch.randn(1, 256, args.height, args.width, device=device)
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

    if dist.get_rank() == 0:
        assert torch.allclose(result, patch_result, atol=1e-3), "two hidden states are not equal"

    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()