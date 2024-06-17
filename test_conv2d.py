from patchvae.models.layers.conv2d import PatchConv2d
from patchvae.modules.patch_utils import Patchify, DePatchify

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

class PatchConv2dModules(nn.Module):
    def __init__(self, convs: nn.ModuleDict):
        super().__init__()
        self.patchify = Patchify()
        self.patch_convs = nn.ModuleList()
        for conv in convs:
            patched_conv = PatchConv2d(
                conv.in_channels,
                conv.out_channels,
                conv.kernel_size,
                conv.stride,
                conv.padding,
                conv.dilation,
                conv.groups,
                conv.bias is not None,
                conv.padding_mode,
                conv.weight.device,
                conv.weight.dtype
            )
            patched_conv.weight.data = conv.weight.data
            patched_conv.bias.data = conv.bias.data
            self.patch_convs.append(patched_conv)
        self.depatchify = DePatchify()
    
    def forward(self, x):
        x = self.patchify(x)
        for patch_conv in self.patch_convs:
            x = patch_conv(x)
        return self.depatchify(x)
            

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
    # for kernel_size in range(3,4):
    #     for stride in range(1,2):
    #         for padding in range(1,2):
    for kernel_size in range(3,10):
        for stride in range(1,kernel_size+1):
            for padding in range(1,kernel_size):
                convs = Conv2dModules(in_channels, out_channels, kernel_size, stride, padding).to(f"cuda:{rank}")
                patch_convs = PatchConv2dModules(convs.convs).to(f"cuda:{rank}")

                hidden_state = torch.randn(1, 64, args.height, args.width, device=f"cuda:{rank}")
                result = convs(hidden_state)

                
                if rank == 0: 
                    print(kernel_size, stride, padding, "start", flush=True)

                ppresult = patch_convs(hidden_state.clone())



                if rank == 0:
                    max_height = (hidden_state.shape[2] + padding * 2 - kernel_size + 1 + stride - 1) // stride
                    # print(result)
                    # print(ppresult)

                    # print(result.shape)
                    # print(ppresult.shape)
                    flag = 1
                    for i in range(out_channels):
                        for j in range(max_height):
                            for k in range(max_height):
                                if (result[0, i, j, k] - ppresult[0, i, j, k]) > 1e-3:
                                    flag = 0
                                    # print(f"result: {result[0, i, j, k]}, ppresult: {ppresult[0, i, j, k]}")
                                    # print(f"i: {i}, j: {j}, k: {k}\n")
                    if flag == 0:
                        print("in kernel size: ", kernel_size, "stride: ", stride, "padding: ", padding, flush=True)
                        print("two hidden states are not equal\n", flush=True)
                    else:
                        print(kernel_size, stride, padding, "end", flush=True)

    # assert torch.equal(result, ppresult), "two hidden states are not equal"


if __name__ == "__main__":
    main()