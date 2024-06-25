from distvae.models.layers.conv2d import PatchConv2d
from distvae.modules.patch_utils import Patchify, DePatchify
from distvae.modules.adapters.layers.conv_adapters import Conv2dAdapter

import torch
import random
import argparse
import torch.distributed as dist
from torch import nn

class Conv2dModules(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.convs = nn.ModuleList([
            # nn.Conv2d(512, 256, kernel_size, stride, padding),
            # nn.Conv2d(256, 128, kernel_size, stride, padding),
            # nn.Conv2d(128, 64, kernel_size, stride, padding),
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        ])

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        return x
            

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

    in_channels = 64
    out_channels = 3
    for kernel_size in range(3,4):
        for stride in range(1,2):
            for padding in range(1,2):
    # for kernel_size in range(3,10):
    #     for stride in range(1,kernel_size+1):
    #         for padding in range(1,kernel_size):
                convs = Conv2dModules(in_channels, out_channels, kernel_size, stride, padding).to(f"cuda:{rank}")
                patch_convs = nn.ModuleList()
                for conv in convs.convs:
                    patch_convs.append(Conv2dAdapter(conv))
                patch_convs = patch_convs.to(f"cuda:{rank}")

                hidden_state = torch.randn(1, 64, args.height, args.width, device=f"cuda:{rank}")
                result = convs(hidden_state)

                
                if rank == 0: 
                    print(kernel_size, stride, padding, "start", flush=True)
                patch = Patchify()
                depatch = DePatchify()

                patch_hidden_state = patch(hidden_state)
                for conv in patch_convs:
                    patch_hidden_state = conv(patch_hidden_state)
                ppresult = depatch(patch_hidden_state)



                if rank == 0:
                    max_height = (hidden_state.shape[2] + padding * 2 - kernel_size + 1 + stride - 1) // stride
                    # print(result)
                    # print(ppresult)

                    # print(result.shape)
                    # print(ppresult.shape)
                    # flag = 1
                    # for i in range(out_channels):
                    #     for j in range(max_height):
                    #         for k in range(max_height):
                    #             if (result[0, i, j, k] - ppresult[0, i, j, k]) > 1e-3:
                    #                 flag = 0
                                    # print(f"result: {result[0, i, j, k]}, ppresult: {ppresult[0, i, j, k]}")
                                    # print(f"i: {i}, j: {j}, k: {k}\n")
                    if not torch.allclose(result, ppresult, atol=1e-6):
                        print("in kernel size: ", kernel_size, "stride: ", stride, "padding: ", padding, flush=True)
                        print("two hidden states are not equal\n", flush=True)
                    else:
                        print(kernel_size, stride, padding, "end", flush=True)

    # assert torch.equal(result, ppresult), "two hidden states are not equal"


if __name__ == "__main__":
    main()