"""Unit tests for distvae.models.layers.conv_utils."""

import pytest
import torch
from unittest.mock import patch

from distvae.models.layers.conv_utils import (
    calc_patch_index,
    calc_top_halo_width,
    calc_bottom_halo_width,
    calc_halo_width,
    calc_halo_width_unit_stride,
    correct_end,
    correct_start,
    build_crop_slice,
    adjust_padding_for_patch,
)


class TestCalcPatchIndex:
    """Tests for calc_patch_index."""

    def test_single_rank(self):
        patch_list = [torch.tensor([10])]
        assert calc_patch_index(patch_list) == [0, 10]

    def test_two_equal_patches(self):
        patch_list = [torch.tensor([4]), torch.tensor([4])]
        assert calc_patch_index(patch_list) == [0, 4, 8]

    def test_unequal_patches(self):
        patch_list = [
            torch.tensor([4]),
            torch.tensor([3]),
            torch.tensor([5]),
        ]
        assert calc_patch_index(patch_list) == [0, 4, 7, 12]


class TestCalcTopHaloWidth:
    """Tests for calc_top_halo_width"""

    def test_rank_zero_returns_zero(self):
        height_index = [0, 10, 20]
        assert calc_top_halo_width(0, height_index, 3, 0, 1) == 0

    def test_middle_rank(self):
        height_index = [0, 8, 16, 24]
        result = calc_top_halo_width(1, height_index, 3, 1, 1)
        assert result == 1

    def test_invalid_rank_negative(self):
        with pytest.raises(AssertionError, match="rank should not be smaller than 0"):
            calc_top_halo_width(-1, [0, 10], 3, 0, 1)

    def test_invalid_rank_too_large(self):
        with pytest.raises(
            AssertionError, match="rank should be smaller than the length of height_index"
        ):
            calc_top_halo_width(2, [0, 10, 20], 3, 0, 1)

    def test_invalid_stride(self):
        with pytest.raises(AssertionError, match="stride should be larger than 0"):
            calc_top_halo_width(1, [0, 10, 20], 3, 0, 0)

    def test_invalid_padding(self):
        with pytest.raises(AssertionError, match="padding should not be smaller than 0"):
            calc_top_halo_width(1, [0, 10, 20], 3, -1, 1)


class TestCalcBottomHaloWidth:
    """Tests for calc_bottom_halo_width"""

    def test_last_rank_returns_zero(self):
        height_index = [0, 10, 20]
        assert calc_bottom_halo_width(1, height_index, 3, 0, 1) == 0

    def test_middle_rank(self):
        height_index = [0, 8, 16, 24]
        result = calc_bottom_halo_width(1, height_index, 3, 1, 1)
        assert result == 1

    def test_invalid_rank_negative(self):
        with pytest.raises(AssertionError, match="rank should not be smaller than 0"):
            calc_bottom_halo_width(-1, [0, 10], 3, 0, 1)

    def test_invalid_rank_too_large(self):
        with pytest.raises(
            AssertionError, match="rank should be smaller than the length of height_index"
        ):
            calc_bottom_halo_width(2, [0, 10, 20], 3, 0, 1)

    def test_invalid_stride(self):
        with pytest.raises(AssertionError, match="stride should be larger than 0"):
            calc_bottom_halo_width(1, [0, 10, 20], 3, 0, 0)

    def test_invalid_padding(self):
        with pytest.raises(AssertionError, match="padding should not be smaller than 0"):
            calc_bottom_halo_width(1, [0, 10, 20], 3, -1, 1)


class TestCalcHaloWidth:
    """Tests for calc_halo_width."""

    @patch("distvae.models.layers.conv_utils.DistributedEnv.get_group_world_size")
    def test_first_rank_top_zero(self, mock_world_size):
        mock_world_size.return_value = 3
        height_index = [0, 8, 16, 24]
        top, bottom = calc_halo_width(0, height_index, 3, 0, 1)
        assert top == 0
        assert bottom >= 0

    @patch("distvae.models.layers.conv_utils.DistributedEnv.get_group_world_size")
    def test_last_rank_bottom_zero(self, mock_world_size):
        mock_world_size.return_value = 3
        height_index = [0, 8, 16, 24]
        top, bottom = calc_halo_width(2, height_index, 3, 0, 1)
        assert bottom == 0
        assert top >= 0

    @patch("distvae.models.layers.conv_utils.DistributedEnv.get_group_world_size")
    def test_middle_rank_both_nonzero(self, mock_world_size):
        mock_world_size.return_value = 3
        height_index = [0, 8, 16, 24]
        top, bottom = calc_halo_width(1, height_index, 3, 1, 1)
        expected_top = calc_top_halo_width(1, height_index, 3, 1, 1)
        expected_bottom = calc_bottom_halo_width(1, height_index, 3, 1, 1)
        assert top == expected_top
        assert bottom == expected_bottom


class TestCalcHaloWidthUnitStride:
    """The stride-1 shortcut has to answer exactly what the gathered boundaries answer.

    It is what every unit-stride convolution uses in place of an all_gather, so if it ever
    disagreed with calc_halo_width the ranks would exchange the wrong rows and the seam
    between two patches would be quietly wrong rather than loudly broken.
    """

    @pytest.mark.parametrize("kernel_size", [1, 2, 3, 4, 5, 7])
    @pytest.mark.parametrize("padding", [0, 1, 2, 3])
    @pytest.mark.parametrize(
        "patch_sizes",
        [
            [8, 8],
            [8, 8, 8, 8],
            [9, 8, 8, 8],  # the uneven split Patchify makes when rows do not divide by ranks
            [3, 2, 2],  # patches barely wider than the kernel
            [64, 63, 63, 63],
        ],
    )
    @patch("distvae.models.layers.conv_utils.DistributedEnv.get_group_world_size")
    def test_it_agrees_with_the_gathered_boundaries(
        self, mock_world_size, patch_sizes, padding, kernel_size
    ):
        world_size = len(patch_sizes)
        mock_world_size.return_value = world_size
        height_index = calc_patch_index([torch.tensor([s]) for s in patch_sizes])

        for rank in range(world_size):
            assert calc_halo_width_unit_stride(rank, world_size, kernel_size) == calc_halo_width(
                rank, height_index, kernel_size, padding, 1
            )

    def test_the_edge_ranks_have_nothing_beyond_them(self):
        assert calc_halo_width_unit_stride(0, 4, 3)[0] == 0
        assert calc_halo_width_unit_stride(3, 4, 3)[1] == 0

    def test_a_lone_rank_needs_no_halo_at_all(self):
        assert calc_halo_width_unit_stride(0, 1, 7) == (0, 0)


class TestCorrectEnd:
    """Tests for correct_end (pure)."""

    def test_formula_stride1(self):
        # ((end + 0) // 1 - 1) * 1 + k = end - 1 + k
        assert correct_end(8, 3, 1) == ((8 + 1 - 1) // 1 - 1) * 1 + 3
        assert correct_end(8, 3, 1) == 10

    def test_formula_stride2(self):
        # end=6, k=3, s=2: ((6+2-1)//2 - 1)*2 + 3 = (3 - 1)*2 + 3 = 7
        assert correct_end(6, 3, 2) == 7

    def test_small_end(self):
        result = correct_end(4, 3, 1)
        assert result == 6  # (4-1)*1 + 3


class TestCorrectStart:
    """Tests for correct_start (pure)."""

    def test_aligned(self):
        assert correct_start(0, 1) == 0
        assert correct_start(0, 2) == 0

    def test_unaligned(self):
        # (2+1-1)//1 * 1 = 2
        assert correct_start(2, 1) == 2
        # (3+2-1)//2 * 2 = 4
        assert correct_start(3, 2) == 4


class TestBuildCropSlice:
    """Tests for build_crop_slice (pure)."""

    def test_out_len_equals_patch_size_ndim4(self):
        # patch_slice = slice(0, patch_size)
        result = build_crop_slice(
            patch_dim=2, patch_size=4, halo_width=(1, 1), out_len=4, ndim=4
        )
        assert len(result) == 4
        assert result[0] == slice(None)
        assert result[1] == slice(None)
        assert result[2] == slice(0, 4)
        assert result[3] == slice(None)

    def test_out_len_ne_patch_size_ndim4(self):
        result = build_crop_slice(
            patch_dim=2, patch_size=4, halo_width=(2, 1), out_len=10, ndim=4
        )
        assert result[2] == slice(2, 6)  # halo_width[0], halo_width[0] + patch_size

    def test_ndim4_patch_dim3(self):
        result = build_crop_slice(
            patch_dim=3, patch_size=5, halo_width=(0, 0), out_len=5, ndim=4
        )
        assert len(result) == 4
        assert result[3] == slice(0, 5)
        assert result[2] == slice(None)

    def test_ndim5_patch_dim2(self):
        result = build_crop_slice(
            patch_dim=2, patch_size=3, halo_width=(1, 2), out_len=8, ndim=5
        )
        assert len(result) == 5
        assert result[2] == slice(1, 4)
        assert all(result[i] == slice(None) for i in (0, 1, 3, 4))

    def test_ndim5_patch_dim4(self):
        result = build_crop_slice(
            patch_dim=4, patch_size=6, halo_width=(0, 0), out_len=6, ndim=5
        )
        assert len(result) == 5
        assert result[4] == slice(0, 6)


class TestAdjustPaddingForPatch:
    """Tests for adjust_padding_for_patch (pure)."""

    def test_ndim4_int_padding_rank0(self):
        # rank 0: zero right edge for patch_dim
        # patch_dim=2 -> right_idx=3, left_idx=2
        result = adjust_padding_for_patch(1, rank=0, world_size=3, patch_dim=2, ndim=4)
        assert result == (1, 1, 1, 0)

    def test_ndim4_int_padding_last_rank(self):
        # last rank: zero left edge
        result = adjust_padding_for_patch(1, rank=2, world_size=3, patch_dim=2, ndim=4)
        assert result == (1, 1, 0, 1)

    def test_ndim4_int_padding_middle_rank(self):
        result = adjust_padding_for_patch(1, rank=1, world_size=3, patch_dim=2, ndim=4)
        assert result == (1, 1, 0, 0)

    def test_ndim4_tuple_padding(self):
        result = adjust_padding_for_patch(
            (2, 2, 2, 2), rank=0, world_size=2, patch_dim=3, ndim=4
        )
        # patch_dim=3 -> right_idx=1, left_idx=0; rank 0 zeros right
        assert result == (2, 0, 2, 2)

    def test_ndim5_int_padding_rank0(self):
        # patch_dim=2 -> left_idx=4, right_idx=5; rank 0 zeros right
        result = adjust_padding_for_patch(1, rank=0, world_size=2, patch_dim=2, ndim=5)
        assert len(result) == 6
        assert result[5] == 0
        assert result[4] == 1

    def test_ndim5_int_padding_last_rank(self):
        result = adjust_padding_for_patch(1, rank=1, world_size=2, patch_dim=2, ndim=5)
        assert result[4] == 0
        assert result[5] == 1

    def test_ndim5_int_padding_middle_rank(self):
        result = adjust_padding_for_patch(1, rank=1, world_size=3, patch_dim=3, ndim=5)
        # patch_dim=3 -> (2, 3)
        assert result[2] == 0
        assert result[3] == 0

    def test_ndim5_patch_dim4(self):
        result = adjust_padding_for_patch(2, rank=0, world_size=2, patch_dim=4, ndim=5)
        # patch_dim=4 -> left_idx=0, right_idx=1; rank 0 zeros right
        assert result[0] == 2
        assert result[1] == 0
