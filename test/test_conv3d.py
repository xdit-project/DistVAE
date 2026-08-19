"""Unit tests for distvae.models.layers.conv3d.PatchConv3d."""

import pytest
import torch
import torch.nn as nn

from distvae.models.layers.conv3d import PatchConv3d
from distributed_harness import make_parallel_context


class TestPatchConv3dConstructor:
    """Tests for PatchConv3d constructor."""

    @pytest.mark.parametrize(
        "patch_dim,expected", [(-2, -2), (3, -2), (-1, -1), (4, -1)]
    )
    def test_valid_patch_dim(self, patch_dim, expected):
        module = PatchConv3d(
            4, 8, 3, parallel_context=make_parallel_context(patch_dim)
        )
        assert module.patch_dim == expected
        assert module.block_size == 0

    @pytest.mark.parametrize("patch_dim", [-3, 2])
    def test_frame_patch_dim_raises(self, patch_dim):
        with pytest.raises(ValueError, match="frame axis"):
            PatchConv3d(
                4, 8, 3, parallel_context=make_parallel_context(patch_dim)
            )

    @pytest.mark.parametrize("patch_dim", [0, 1, 5])
    def test_invalid_patch_dim_raises(self, patch_dim):
        with pytest.raises(ValueError):
            PatchConv3d(
                4, 8, 3, parallel_context=make_parallel_context(patch_dim)
            )

    def test_dilation_int_raises(self):
        with pytest.raises(AssertionError) as exc_info:
            PatchConv3d(
                4, 8, 3, dilation=2, parallel_context=make_parallel_context()
            )
        assert "dilation is not supported" in str(exc_info.value)

    def test_dilation_tuple_raises(self):
        with pytest.raises(AssertionError) as exc_info:
            PatchConv3d(
                4,
                8,
                3,
                dilation=(1, 2, 1),
                parallel_context=make_parallel_context(),
            )
        assert "dilation is not supported" in str(exc_info.value)

    def test_block_size_int(self):
        module = PatchConv3d(
            4, 8, 3, block_size=0, parallel_context=make_parallel_context()
        )
        assert module.block_size == 0

    def test_block_size_tuple(self):
        module = PatchConv3d(
            4,
            8,
            3,
            block_size=(2, 2, 2),
            parallel_context=make_parallel_context(),
        )
        assert module.block_size == (2, 2, 2)


class TestPatchConv3dSingleRankForward:
    """Single-rank forward: PatchConv3d matches nn.Conv3d when world size is 1."""

    def test_forward_matches_conv3d(self):
        in_ch, out_ch = 4, 8
        k, s, p = 3, 1, 1
        ref_conv = nn.Conv3d(in_ch, out_ch, k, stride=s, padding=p)
        patch_conv = PatchConv3d(
            in_ch,
            out_ch,
            k,
            stride=s,
            padding=p,
            parallel_context=make_parallel_context(),
        )
        with torch.no_grad():
            patch_conv.weight.copy_(ref_conv.weight)
            patch_conv.bias.copy_(ref_conv.bias)
        ref_conv.eval()
        patch_conv.eval()
        x = torch.randn(1, in_ch, 3, 8, 8)
        with torch.no_grad():
            ref_out = ref_conv(x)
            patch_out = patch_conv(x)
        assert torch.allclose(patch_out, ref_out, atol=1e-5)

    def test_forward_padding_mode_zeros(self):
        ref_conv = nn.Conv3d(4, 8, 3, stride=1, padding=1, padding_mode="zeros")
        patch_conv = PatchConv3d(
            4,
            8,
            3,
            stride=1,
            padding=1,
            padding_mode="zeros",
            parallel_context=make_parallel_context(),
        )
        with torch.no_grad():
            patch_conv.weight.copy_(ref_conv.weight)
            patch_conv.bias.copy_(ref_conv.bias)
        x = torch.randn(1, 4, 4, 10, 10)
        with torch.no_grad():
            assert torch.allclose(patch_conv(x), ref_conv(x), atol=1e-5)


class TestPatchConv3dOutputShape:
    """Single-rank output shape matches standard 3D conv formula."""

    def test_output_shape_k3_s1_p1(self):
        # (F + 2*p - (k-1) - 1) / s + 1 = (4 + 2 - 2) / 1 + 1 = 5 per spatial dim
        conv = PatchConv3d(
            4,
            8,
            kernel_size=3,
            stride=1,
            padding=1,
            parallel_context=make_parallel_context(),
        )
        x = torch.randn(2, 4, 4, 8, 8)
        out = conv(x)
        assert out.shape == (2, 8, 4, 8, 8)

    def test_output_shape_k3_s2_p0(self):
        # Standard 3D conv: (L + 2*pad - (k-1) - 1) // stride + 1; L=5,9 k=3 s=2 p=0 -> 2, 4, 4
        conv = PatchConv3d(
            4,
            8,
            kernel_size=3,
            stride=2,
            padding=0,
            parallel_context=make_parallel_context(),
        )
        x = torch.randn(1, 4, 5, 9, 9)
        out = conv(x)
        assert out.shape == (1, 8, 2, 4, 4)
