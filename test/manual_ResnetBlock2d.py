from distvae.modules.patch_utils import Patchify, DePatchify
from distvae.modules.adapters.resnet_adapters import ResnetBlock2DAdapter
from distvae.utils import DistributedEnv, parallel_context

from diffusers.models.resnet import ResnetBlock2D

import torch
import random
import argparse
import torch.distributed as dist
from torch.cuda import set_device, device_count
from torch.cuda import manual_seed as device_manual_seed
try:
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
    context = parallel_context(None, -2, ndim=4)

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
    ).to(device)
    patch_resnet = ResnetBlock2DAdapter(
        resnet, parallel_context=context
    ).to(device)

    hidden_state = torch.randn(1, 64, args.height, args.width, device=device)

    result = resnet(hidden_state, None)
    # if rank == 0:
        # print("result: ", result)

    patch = Patchify(context)
    depatch = DePatchify(context)
    patch_result = patch_resnet(patch(hidden_state))
    # print("patch_res:", rank, patch_result)
    patch_result = depatch(patch_result)

    if dist.get_rank() == 0:
        assert torch.allclose(result, patch_result, atol=1e-2), "two hidden states are not equal"

    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()