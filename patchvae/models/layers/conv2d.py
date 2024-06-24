import torch
import torch.nn as nn
import torch.distributed as dist
from torch import Tensor
from torch.nn import functional as F
from torch.nn.modules.utils import _pair


from torch.nn.common_types import _size_2_t
from typing import Optional, List, Tuple, Union


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
        dtype=None,
        block_size: Union[int, Tuple[int, int]] = 0
    ) -> None:

        if isinstance(dilation, int):
            assert dilation == 1, "dilation is not supported in PatchConv2d"
        else:
            for i in dilation:
                assert i == 1, "dilation is not supported in PatchConv2d"
        self.block_size = block_size
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, 
            groups, bias, padding_mode, device, dtype)
        
    def _get_world_size_and_rank(self):
        world_size = 1
        rank = 0
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        return world_size, rank

    def _calc_patch_height_index(self, patch_height_list: List[Tensor]):
        height_index = []
        cur = 0
        for t in patch_height_list:
            height_index.append(cur)
            cur += t.item()
        height_index.append(cur)
        return height_index

    def _calc_bottom_halo_width(self, rank, height_index, kernel_size, padding = 0, stride = 1):
        assert rank >= 0, "rank should not be smaller than 0"
        assert rank < len(height_index) - 1, "rank should be smaller than the length of height_index - 1"
        assert padding >= 0, "padding should not smaller than 0"
        assert stride > 0, "stride should be larger than 0"

        if rank == dist.get_world_size() - 1:
            return 0
        nstep_before_bottom = (height_index[rank + 1] + padding - (kernel_size - 1) // 2 + stride - 1) // stride
        assert nstep_before_bottom > 0, "nstep_before_bottom should be larger than 0"
        bottom_halo_width =  (nstep_before_bottom - 1) * stride + kernel_size - padding - height_index[rank + 1]
        return max(0, bottom_halo_width)

    def _calc_top_halo_width(self, rank, height_index, kernel_size, padding = 0, stride = 1):
        assert rank >= 0, "rank should not be smaller than 0"
        assert rank < len(height_index) - 1, "rank should be smaller than the length of height_index - 1"
        assert padding >= 0, "padding should not smaller than 0"
        assert stride > 0, "stride should be larger than 0"

        if rank == 0:
            return 0
        nstep_before_top = (height_index[rank] + padding - (kernel_size - 1) // 2 + stride - 1) // stride
        top_halo_width = height_index[rank] - (nstep_before_top * stride - padding)
        return top_halo_width


    def _calc_halo_width_in_h_dim(self, rank, height_index, kernel_size, padding = 0, stride = 1):
        ''' 
            Calculate the width of halo region in height dimension. 
            The halo region is the region that is used for convolution but not included in the output.
            return value: (top_halo_width, bottom_halo_width)
        '''
        halo_width = [
            self._calc_top_halo_width(rank, height_index, kernel_size, padding, stride),
            self._calc_bottom_halo_width(rank, height_index, kernel_size, padding, stride)
        ]
        if rank == 0:
            halo_width[0] = 0
        elif rank == dist.get_world_size() - 1:
            halo_width[1] = 0
        return tuple(halo_width)
        

    # in 2d case, padding is a tuple of 4 integers: [left_pad, right_pad, top_pad, bottom_pad]
    def _adjust_padding_for_patch(self, padding, rank, world_size):
        if isinstance(padding, tuple):
            padding = list(padding)
        elif isinstance(padding, int):
            padding = [padding] * 4

        if rank == 0:
            padding[-1] = 0
        elif rank == world_size - 1:
            padding[-2] = 0
        else:
            padding[-2:] = [0, 0]
        return tuple(padding)

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        bs, channels, h, w = input.shape

        world_size, rank = self._get_world_size_and_rank()

        if (world_size == 1):
            if self.padding_mode != 'zeros':
                return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                                weight, bias, self.stride,
                                _pair(0), self.dilation, self.groups)
            return F.conv2d(input, weight, bias, self.stride,
                            self.padding, self.dilation, self.groups)
            
        else:
        # 1. get the meta data of input tensor and conv operation
            patch_height_list = [torch.zeros(1, dtype=torch.int64, device=f"cuda:{rank}") for _ in range(dist.get_world_size())]
            dist.all_gather(patch_height_list, torch.tensor([h], dtype=torch.int64, device=f"cuda:{rank}"))
            patch_height_index = self._calc_patch_height_index(patch_height_list)
            halo_width = self._calc_halo_width_in_h_dim(rank, patch_height_index, self.kernel_size[0], self.padding[0], self.stride[0])
            prev_bottom_halo_width: int = 0
            next_top_halo_width: int = 0
            if rank != 0:
                prev_bottom_halo_width = self._calc_bottom_halo_width(rank - 1, patch_height_index, self.kernel_size[0], self.padding[0], self.stride[0])
            if rank != world_size - 1:
                next_top_halo_width = self._calc_top_halo_width(rank + 1, patch_height_index, self.kernel_size[0], self.padding[0], self.stride[0])
                next_top_halo_width = max(0, next_top_halo_width)
            

            assert halo_width[0] <= h and halo_width[1] <= h, "halo width is larger than the height of input tensor"


        # 2. get the halo region from other ranks
        # up to down
            to_next = None
            to_prev = None
            top_halo_recv = None
            bottom_halo_recv = None
            if next_top_halo_width > 0:
                # isend to next
                bottom_halo_send = input[:, :, -next_top_halo_width:, :].contiguous()
                to_next = dist.isend(bottom_halo_send, rank + 1)
                
            if halo_width[0] > 0:
                # recv from prev
                assert patch_height_index[rank] - halo_width[0] >= patch_height_index[rank-1], \
                    "width of top halo region is larger than the height of input tensor of last rank"
                top_halo_recv = torch.empty([bs, channels, halo_width[0], w], dtype=input.dtype, device=f"cuda:{rank}")
                dist.recv(top_halo_recv, rank - 1)

        # down to up
            if prev_bottom_halo_width > 0:
                # isend to prev
                top_halo_send = input[:, :, :prev_bottom_halo_width, :].contiguous()
                to_prev = dist.isend(top_halo_send, rank - 1)
            
            if halo_width[1] > 0:
                # recv from next
                assert patch_height_index[rank+1] + halo_width[1] < patch_height_index[rank+2], \
                    "width of bottom halo region is larger than the height of input tensor of next rank"
                bottom_halo_recv = torch.empty([bs, channels, halo_width[1], w], dtype=input.dtype, device=f"cuda:{rank}")
                dist.recv(bottom_halo_recv, rank + 1)
        
        # Remove redundancy at the top of the input
            if halo_width[0] < 0:
                input = input[:, :, -halo_width[0]:, :]
        # concat the halo region to the input tensor            
            if top_halo_recv is not None:
                input = torch.cat([top_halo_recv, input], dim=-2)
            if bottom_halo_recv is not None:
                input = torch.cat([input, bottom_halo_recv], dim=-2)
            
        # wait for the communication to finish
            if to_next is not None:
                to_next.wait()
            if to_prev is not None:
                to_prev.wait()

        # 3. do convolution and postprocess
            conv_res: Tensor
            padding = self._adjust_padding_for_patch(self._reversed_padding_repeated_twice, rank=rank, world_size=world_size)
            bs, channels, h, w = input.shape
            if self.block_size == 0 or (h <= self.block_size and w <= self.block_size):
                if self.padding_mode != 'zeros':
                    conv_res = F.conv2d(F.pad(input, padding, mode=self.padding_mode),
                                    weight, bias, self.stride,
                                    _pair(0), self.dilation, self.groups)
                else:
                    if self.stride[0] == 1 and self.padding[0] == 1 and self.kernel_size[0] == 3:
                        conv_res = F.conv2d(input, weight, bias, self.stride,
                                    self.padding, self.dilation, self.groups)
                        if halo_width[1] == 0:
                            conv_res = conv_res[:, :, halo_width[0]:, :].contiguous()
                        else:
                            conv_res = conv_res[:, :, halo_width[0]:-halo_width[1], :]
                        # print(rank, conv_res.shape, flush=True)
                    else:
                        conv_res = F.conv2d(F.pad(input, padding, "constant", 0.0),
                                        weight, bias, self.stride,
                                        _pair(0), self.dilation, self.groups)
                return conv_res

        # 3.1. if block_size is not 0, split patch to block and do convolution to 
                # reduce memory spike
            else:
                if self.padding_mode != "zeros":
                    input = F.pad(input, padding, mode=self.padding_mode)
                elif self.padding != 0:
                    input = F.pad(input, padding, mode="constant")

                _, _, h, w = input.shape
                num_chunks_in_h = 0
                num_chunks_in_w = 0
                if isinstance(self.block_size, int):
                    num_chunks_in_h = (h + self.block_size - 1) // self.block_size
                    num_chunks_in_w = (w + self.block_size - 1) // self.block_size
                elif isinstance(self.block_size, tuple):
                    num_chunks_in_h = (h + self.block_size[0] - 1) // self.block_size[0]
                    num_chunks_in_w = (w + self.block_size[1] - 1) // self.block_size[1]
                unit_chunk_size_h = h // num_chunks_in_h
                unit_chunk_size_w = w // num_chunks_in_w
                if isinstance(self.kernel_size, int):
                    kernel_size_h, kernel_size_w = self.kernel_size, self.kernel_size
                elif isinstance(self.kernel_size, tuple):
                    kernel_size_h, kernel_size_w = self.kernel_size
                else:
                    raise ValueError(
                        f"kernel_size should be int or tuple, type:{type(self.kernel_size)}"
                    )

                if isinstance(self.stride, int):
                    stride_h, stride_w = self.stride, self.stride
                elif isinstance(self.stride, tuple):
                    stride_h, stride_w = self.stride
                else:
                    raise ValueError(
                        f"stride should be int or tuple, type: {type(self.stride)}"
                    )

                def correct_end(end, kernel_size, stride):
                    return ((end + stride - 1) // stride - 1) * stride + kernel_size

                def correct_start(start, stride):
                    return ((start + stride - 1) // stride) * stride

                outputs = []
                for idx_h in range(num_chunks_in_h):
                    inner_output = []
                    for idx_w in range(num_chunks_in_w):
                        start_w = idx_w * unit_chunk_size_w
                        start_h = idx_h * unit_chunk_size_h
                        end_w = (idx_w + 1) * unit_chunk_size_w
                        end_h = (idx_h + 1) * unit_chunk_size_h
                        if idx_w + 1 < num_chunks_in_w:
                            end_w = correct_end(end_w, kernel_size_w, stride_w)
                        else:
                            end_w = w
                        if idx_h + 1 < num_chunks_in_h:
                            end_h = correct_end(end_h, kernel_size_h, stride_h)
                        else:
                            end_h = h

                        if idx_w > 0:
                            start_w = correct_start(start_w, stride_w)
                        if idx_h > 0:
                            start_h = correct_start(start_h, stride_h)

                        inner_output.append(
                            F.conv2d(
                                input[:, :, start_h:end_h, start_w:end_w],
                                weight,
                                bias,
                                self.stride,
                                0,
                                self.dilation,
                                self.groups,
                            )
                        )
                    outputs.append(torch.cat(inner_output, dim=-1))
                return torch.cat(outputs, dim=-2)