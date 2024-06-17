import random
from typing import Optional, Union

import torch
import torch.distributed as dist
from torch import nn
from torch import Tensor
from torch.nn import functional as F
from torch.nn.modules.utils import _pair
from torch.nn.common_types import _size_2_t

# to simplify the code, we only consider the case where padding = 1, stride = 1, kernel_size = 3
class PatchConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',  # TODO: refine this type
        device=None,
        dtype=None
    ) -> None:

        assert dilation != 1, "dilation is not supported in PatchConv2d"
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, 
            groups, bias, padding_mode, device, dtype)
        
    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        channels, h, w = input.shape
        rank = dist.get_rank()

        # p2p communication to exchange halo data
        if rank == 0:
            bottom_halo_send = input[:, -1:, :].contiguous()
            dist.send(bottom_halo_send, dst=1)
        else:
            top_halo_recv = torch.empty([channels, 1, w], device=f"cuda:{rank}")
            dist.recv(top_halo_recv, src=0)
            
        if rank == 1:
            top_halo_send = input[:, :1, :].contiguous()
            dist.send(top_halo_send, dst=0)
        else:
            bottom_halo_recv = torch.empty([channels, 1, w], device=f"cuda:{rank}")
            dist.recv(bottom_halo_recv, src=1)
        
        if rank == 0:
            input = torch.cat([input, bottom_halo_recv], dim=1)
        elif rank == 1:
            input = torch.cat([top_halo_recv, input], dim=1)
            
        # padding: [left_padding, right_padding, top_padding, bottom_padding]
        padding = [1] * 4
        if rank == 0:
            padding[3] = 0
        elif rank == 1:
            padding[2] = 0

        conv_res = F.conv2d(input, weight, bias, self.stride,
                                self.padding, self.dilation, self.groups)

        # print(rank, conv_res.shape, flush=True)
        if rank == 0:
            conv_res = conv_res[:, :-1, :].contiguous()
        elif rank == 1:
            conv_res = conv_res[:, 1:, :].contiguous()

        return conv_res

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def main():
    set_seed()
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.device('cuda', rank)

    conv = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, device=f"cuda:{rank}")
    patch_conv = PatchConv2d(
        conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride,
        conv.padding, conv.dilation, conv.groups, conv.bias is not None,
        conv.padding_mode, conv.weight.device, conv.weight.dtype
    )
    patch_conv.weight.data = conv.weight.data
    patch_conv.bias.data = conv.bias.data
    patch_conv.to(f"cuda:{rank}")

    height = 128
    width = 128
    # [in_channels, height, width]
    hidden_state = torch.randn(64, height, width, device=f"cuda:{rank}")

# calc nn.Conv2d result
    result = conv(hidden_state)

# calc PatchConv2d result
    height_index = [0, height // 2, width]
    # split hidden_state in height dimension
    hidden_state_patch = hidden_state[:, height_index[rank]: height_index[rank+1], :].clone()
    patch_result = patch_conv(hidden_state_patch)
    # patch_result.shape: [out_channel, 16 / 2, 16]
    patch_result_list = [torch.empty([1, height // 2, width], device=f"cuda:{rank}") for i in range(world_size)]
    dist.all_gather(patch_result_list, patch_result.contiguous())
    patch_result = torch.cat(patch_result_list, dim=1)

    if rank == 0:
        for i in range(height):
            for j in range(width):
                if abs(result[0, i, j] - patch_result[0, i, j]) > 1e-5:
                    print(f"in [{i}, {j}], result: {result[0, i, j]}, patch_result: {patch_result[0, i, j]}")


if __name__ == "__main__":
    main()