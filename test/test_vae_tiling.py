import unittest
from unittest import mock

from distvae.vae import tiling as vae_tiling


class StubVAE:
    """Minimal VAE stub that stores the supplied tiling attributes."""

    def __init__(self, **attrs):
        for name, value in attrs.items():
            setattr(self, name, value)


def _diffusers_vae(testcase, name, kwargs, *, require_tiling=False):
    """Build an installed VAE, skipping only this test/subtest when its API is unavailable."""
    import diffusers

    cls = getattr(diffusers, name, None)
    if cls is None:
        testcase.skipTest(f"{name} is not in diffusers {diffusers.__version__}")
    vae = cls(**kwargs).eval()
    if require_tiling and not hasattr(vae, "use_tiling"):
        testcase.skipTest(f"diffusers {diffusers.__version__} cannot tile {name}")
    return vae


def legacy_pair_vae():
    """Stub with shared square sample and latent windows and one overlap factor."""
    return StubVAE(
        tile_sample_min_size=256, tile_latent_min_size=32, tile_overlap_factor=0.25
    )


def stride_vae():
    """Stub with per-axis sample windows and explicit pixel strides."""
    return StubVAE(
        tile_sample_min_height=256,
        tile_sample_min_width=256,
        tile_sample_stride_height=192,
        tile_sample_stride_width=192,
        spatial_compression_ratio=8,
    )


def overlap_hw_vae():
    """Stub with per-axis sample windows, latent windows, and overlap factors."""
    return StubVAE(
        tile_sample_min_height=256,
        tile_sample_min_width=256,
        tile_latent_min_height=32,
        tile_latent_min_width=32,
        tile_overlap_factor_height=0.25,
        tile_overlap_factor_width=0.25,
    )


def asymmetric_vae():
    """Stub with rectangular per-axis sample and latent windows."""
    return StubVAE(
        tile_sample_min_height=240,
        tile_sample_min_width=360,
        tile_latent_min_height=30,
        tile_latent_min_width=45,
        tile_overlap_factor_height=1 / 6,
        tile_overlap_factor_width=0.2,
    )


def overlap_factor_vae(sample=256):
    """Stub with shared square tiling attributes and blend methods."""
    return StubVAE(
        tile_sample_min_size=sample,
        tile_latent_min_size=sample // 8,
        tile_overlap_factor=0.25,
        blend_v=lambda above, tile, extent: tile,
        blend_h=lambda left, tile, extent: tile,
    )


def overlap_keyed_vae():
    """Stub with per-axis windows, one shared overlap factor, and blend methods."""
    return StubVAE(
        tile_sample_min_height=256,
        tile_sample_min_width=256,
        tile_latent_min_height=16,
        tile_latent_min_width=16,
        tile_overlap_factor=0.25,
        blend_v=lambda above, tile, extent: tile,
        blend_h=lambda left, tile, extent: tile,
    )


def per_axis_overlap_vae():
    """Stub with unequal per-axis overlap factors and latent windows."""
    return StubVAE(
        tile_sample_min_height=256,
        tile_sample_min_width=256,
        tile_latent_min_height=32,
        tile_latent_min_width=40,
        tile_overlap_factor_height=0.25,
        tile_overlap_factor_width=0.2,
    )


class TestDiffusersCompatibility(unittest.TestCase):
    def test_an_unavailable_optional_vae_is_skipped(self):
        import diffusers

        with mock.patch.object(diffusers, "AutoencoderKLFlux2", None, create=True):
            with self.assertRaises(unittest.SkipTest):
                _diffusers_vae(self, "AutoencoderKLFlux2", {}, require_tiling=True)

    def test_a_class_without_a_tiling_api_is_skipped(self):
        import diffusers

        vae = StubVAE()
        vae.eval = lambda: vae
        with mock.patch.object(diffusers, "AutoencoderKLFlux2", return_value=vae):
            with self.assertRaises(unittest.SkipTest):
                _diffusers_vae(self, "AutoencoderKLFlux2", {}, require_tiling=True)


class TestSupportProbe(unittest.TestCase):

    def test_the_method_alone_does_not_count_as_support(self):
        # Diffusers provides enable_tiling through a mixin whether or not the class implements it,
        # so a VAE can carry the method and still raise NotImplementedError when called.
        unsupported = StubVAE(enable_tiling=lambda: None)
        with self.assertRaises(ValueError):
            vae_tiling.require_vae_support(unsupported, "tiling", "--enable_tiling")

    def test_the_state_flag_counts_as_support(self):
        vae_tiling.require_vae_support(
            StubVAE(use_tiling=False), "tiling", "--enable_tiling"
        )
        vae_tiling.require_vae_support(
            StubVAE(use_slicing=False), "slicing", "--enable_slicing"
        )


class TestTilePaddingError(unittest.TestCase):
    """The padding failure a too-thin tile causes, told apart from failures with other causes"""

    # Verbatim from AutoencoderKLLTX2Video decoding a 16x16 latent at a 128px window.
    REAL = (
        "Argument #4: Padding size should be less than the corresponding input dimension, "
        "but got: padding (1, 1) at dimension 4 of input [1, 8, 3, 4, 1]"
    )

    def test_the_padding_failure_is_recognised(self):
        self.assertTrue(vae_tiling.is_tile_padding_error(RuntimeError(self.REAL)))

    def test_other_decode_failures_are_not(self):
        for message in (
            "expected scalar type BFloat16 but found Float",
            "Expected all tensors to be on the same device",
            "shape '[1, 8, 16, 16]' is invalid for input of size 1024",
            "CUDA error: an illegal memory access was encountered",
        ):
            with self.subTest(error=message):
                self.assertFalse(
                    vae_tiling.is_tile_padding_error(RuntimeError(message))
                )


class TestTileShape(unittest.TestCase):

    def test_reads_the_pixel_shape_of_each_family(self):
        self.assertEqual(vae_tiling.tile_shape(legacy_pair_vae()), (256, 256))
        self.assertEqual(vae_tiling.tile_shape(stride_vae()), (256, 256))
        self.assertEqual(vae_tiling.tile_shape(overlap_hw_vae()), (256, 256))

    def test_a_vae_without_a_shape_reports_none(self):
        vae = StubVAE(tile_overlap_h=0.25)
        self.assertIsNone(vae_tiling.tile_shape(vae))
        self.assertIsNone(vae_tiling.tile_shape_plan(vae, 128, 128))

    def test_a_native_rectangle_is_preserved(self):
        self.assertEqual(vae_tiling.tile_shape(asymmetric_vae()), (240, 360))

    def test_spatial_ratio_falls_back_from_config_to_the_module(self):
        self.assertEqual(vae_tiling.spatial_ratio(stride_vae()), 8)
        self.assertIsNone(vae_tiling.spatial_ratio(legacy_pair_vae()))


class TestSquareTileShapePlan(unittest.TestCase):

    def test_every_attribute_is_rescaled_by_the_same_factor(self):
        plan = vae_tiling.tile_shape_plan(stride_vae(), 128, 128)
        self.assertEqual(
            plan,
            {
                "tile_sample_min_height": 128,
                "tile_sample_min_width": 128,
                "tile_sample_stride_height": 96,
                "tile_sample_stride_width": 96,
            },
        )

    def test_a_window_that_does_not_divide_whole_is_refused(self):
        # 100px would put the latent window at 12.5, which no VAE can hold.
        self.assertIsNone(vae_tiling.tile_shape_plan(legacy_pair_vae(), 100, 100))

    def test_an_overlap_that_does_not_land_whole_is_refused(self):
        # 200px gives a latent window of 25, and 25 x 0.75 truncates to a stride the pixel crop
        # does not agree with, which assembles an image of the wrong size.
        self.assertIsNone(vae_tiling.tile_shape_plan(legacy_pair_vae(), 200, 200))
        self.assertIsNone(vae_tiling.tile_shape_plan(overlap_hw_vae(), 200, 200))
        self.assertIsNotNone(vae_tiling.tile_shape_plan(legacy_pair_vae(), 192, 192))

    def test_each_axis_must_keep_its_latent_step_and_pixel_crop_consistent(self):
        # The width maps 256 pixels to 40 latents, a non-integral 6.4x ratio. Its latent stride
        # and pixel crop cannot describe the same distance, even though 40 x 0.8 is whole.
        self.assertIsNone(
            vae_tiling.tile_shape_plan(per_axis_overlap_vae(), 256, 256)
        )
        self.assertIsNone(
            vae_tiling.tile_shape_plan(per_axis_overlap_vae(), 224, 224)
        )

    def test_an_unkeyed_overlap_fraction_covers_both_axes(self):
        vae = StubVAE(
            tile_sample_min_height=256,
            tile_sample_min_width=256,
            tile_latent_min_height=16,
            tile_latent_min_width=16,
            tile_overlap_factor=0.25,
        )
        self.assertIsNotNone(vae_tiling.tile_shape_plan(vae, 64, 64))
        # 32px puts each latent window at 2, and 2 x 0.75 truncates to a stride of 1.
        self.assertIsNone(vae_tiling.tile_shape_plan(vae, 32, 32))

    def test_a_stride_below_one_latent_pixel_is_refused(self):
        # 8px would leave a 6px stride, under this VAE's 8px latent pixel, and diffusers steps
        # through the latents in a range() that would then be empty.
        self.assertIsNone(vae_tiling.tile_shape_plan(stride_vae(), 8, 8))

    def test_a_window_above_the_default_still_plans(self):
        # _apply_vae_tile_size declines these itself, having the config to say why.
        plan = vae_tiling.tile_shape_plan(stride_vae(), 512, 512)
        self.assertEqual(plan["tile_sample_stride_height"], 384)


class TestTileShapePlan(unittest.TestCase):

    def test_a_legacy_square_window_can_be_planned_rectangularly(self):
        vae = legacy_pair_vae()

        plan = vae_tiling.tile_shape_plan(vae, 128, 192)

        self.assertEqual(
            plan,
            {
                "tile_sample_min_size": 128,
                "tile_sample_min_height": 128,
                "tile_sample_min_width": 192,
                "tile_latent_min_size": 16,
                "tile_latent_min_height": 16,
                "tile_latent_min_width": 24,
            },
        )
        vae_tiling.apply_tile_plan(vae, plan)
        self.assertEqual(
            vae_tiling.overlap_windows(vae), ((16, 24), (128, 192))
        )

    def test_a_stored_stride_is_rescaled_independently_on_each_axis(self):
        self.assertEqual(
            vae_tiling.tile_shape_plan(stride_vae(), 128, 192),
            {
                "tile_sample_min_height": 128,
                "tile_sample_min_width": 192,
                "tile_sample_stride_height": 96,
                "tile_sample_stride_width": 144,
            },
        )

    def test_either_non_integral_axis_rejects_the_rectangle(self):
        self.assertIsNone(
            vae_tiling.tile_shape_plan(legacy_pair_vae(), 128, 100)
        )
        # The scaled width stride is 99 pixels, which cannot step an 8-pixel
        # latent grid without truncating.
        self.assertIsNone(vae_tiling.tile_shape_plan(stride_vae(), 128, 132))

    def test_the_shape_reader_never_squares_a_native_rectangle(self):
        self.assertEqual(vae_tiling.tile_shape(legacy_pair_vae()), (256, 256))
        self.assertEqual(vae_tiling.tile_shape(asymmetric_vae()), (240, 360))

    def test_square_planning_uses_the_rectangular_mechanics(self):
        self.assertEqual(
            vae_tiling.tile_shape_plan(legacy_pair_vae(), 128, 128),
            {
                "tile_sample_min_size": 128,
                "tile_sample_min_height": 128,
                "tile_sample_min_width": 128,
                "tile_latent_min_size": 16,
                "tile_latent_min_height": 16,
                "tile_latent_min_width": 16,
            },
        )

    def test_rectangular_legacy_windows_install_a_local_replacement(self):
        import torch
        import torch.nn.functional as functional

        vae = overlap_factor_vae()
        vae.decoder = lambda tile: functional.interpolate(
            tile, scale_factor=8, mode="nearest"
        )
        plan = vae_tiling.tile_shape_plan(vae, 128, 192)
        vae_tiling.apply_tile_plan(vae, plan)

        decode = vae_tiling.tiled_decode_for(vae)

        self.assertIsNotNone(decode)
        sample = decode(torch.randn(1, 4, 24, 32)).sample
        self.assertEqual(sample.shape, (1, 4, 192, 256))

    def test_legacy_threshold_enters_tiling_when_the_smaller_axis_is_exceeded(self):
        import torch
        from diffusers.models.autoencoders.vae import DecoderOutput

        kwargs, _, _ = TestEverySupportedVAE.VAES["AutoencoderKL"]
        vae = _diffusers_vae(self, "AutoencoderKL", kwargs, require_tiling=True)
        vae.enable_tiling()
        plan = vae_tiling.tile_shape_plan(vae, 128, 384)
        vae_tiling.apply_tile_plan(vae, plan)
        vae.tiled_decode = mock.Mock(
            return_value=DecoderOutput(sample=torch.empty(1, 4, 136, 192))
        )

        vae._decode(torch.randn(1, 4, 17, 24))

        self.assertEqual(vae.tile_latent_min_size, 16)
        vae.tiled_decode.assert_called_once()

    def test_native_keyed_rectangles_keep_the_upstream_local_loop(self):
        vae = overlap_keyed_vae()
        plan = vae_tiling.tile_shape_plan(vae, 128, 384)
        vae_tiling.apply_tile_plan(vae, plan)

        self.assertIsNotNone(vae_tiling.tiled_decode_for(vae))


class TestLatentRows(unittest.TestCase):
    """How many rows a planned tile leaves available for spatial sharding"""

    def test_rectangular_window_uses_height_instead_of_the_smaller_axis(self):
        vae = legacy_pair_vae()
        self.assertEqual(
            vae_tiling.latent_rows(
                vae, vae_tiling.tile_shape_plan(vae, 256, 64)
            ),
            32,
        )

    def test_rows_come_from_the_latent_window_where_the_vae_carries_one(self):
        vae = legacy_pair_vae()
        self.assertEqual(
            vae_tiling.latent_rows(
                vae, vae_tiling.tile_shape_plan(vae, 128, 128)
            ),
            16,
        )

    def test_rows_come_from_the_compression_ratio_otherwise(self):
        vae = stride_vae()
        self.assertEqual(
            vae_tiling.latent_rows(
                vae, vae_tiling.tile_shape_plan(vae, 128, 128)
            ),
            16,
        )

    def test_a_vae_that_says_neither_reports_none(self):
        vae = StubVAE(tile_sample_min_height=256, tile_sample_min_width=256)
        self.assertIsNone(
            vae_tiling.latent_rows(
                vae, vae_tiling.tile_shape_plan(vae, 128, 128)
            )
        )

    def test_with_no_plan_the_vae_s_own_window_is_the_plan(self):
        # DistVAE must validate the VAE's default window when tiling was enabled before the
        # integration applied an explicit plan; every tile is subsequently split across ranks.
        self.assertEqual(vae_tiling.latent_rows(legacy_pair_vae()), 32)
        self.assertEqual(vae_tiling.latent_rows(stride_vae()), 32)
        self.assertIsNone(vae_tiling.latent_rows(StubVAE(tile_overlap_factor=0.25)))

    def test_a_plan_is_read_ahead_of_what_the_vae_still_holds(self):
        # The plan describes what is about to be set, so a caller weighing one against the ranks
        # has to be answered about the plan and not about the window it is replacing.
        vae = legacy_pair_vae()
        self.assertEqual(
            vae_tiling.latent_rows(
                vae, vae_tiling.tile_shape_plan(vae, 128, 128)
            ),
            16,
        )
        self.assertEqual(vae_tiling.latent_rows(vae), 32)


class TestEverySupportedVAE(unittest.TestCase):
    """Every supported VAE accepts a resized tile window without changing output size"""

    # Minimal configs preserve each class's decoder topology. LTX2 pins its compression ratio
    # because the default describes more encoder stages than its decoder upsamples.
    VAES = {
        "AutoencoderKL": (
            dict(
                block_out_channels=[8, 8, 16, 16],
                layers_per_block=1,
                latent_channels=4,
                norm_num_groups=8,
                sample_size=256,
                down_block_types=["DownEncoderBlock2D"] * 4,
                up_block_types=["UpDecoderBlock2D"] * 4,
            ),
            False,
            4,
        ),
        "AutoencoderKLFlux2": (
            dict(
                block_out_channels=[8, 8, 16, 16],
                layers_per_block=1,
                latent_channels=4,
                norm_num_groups=8,
                sample_size=256,
            ),
            False,
            4,
        ),
        "AutoencoderKLWan": (
            dict(base_dim=8, z_dim=4, dim_mult=[1, 2, 4, 4], num_res_blocks=1),
            True,
            4,
        ),
        "AutoencoderKLQwenImage": (
            dict(base_dim=8, z_dim=4, dim_mult=[1, 2, 4, 4], num_res_blocks=1),
            True,
            4,
        ),
        "AutoencoderKLHunyuanVideo": (
            dict(
                block_out_channels=(8, 8, 16, 16),
                layers_per_block=1,
                latent_channels=4,
                norm_num_groups=8,
            ),
            True,
            4,
        ),
        "AutoencoderKLHunyuanVideo15": (
            dict(
                block_out_channels=(8, 8, 16, 16, 16),
                layers_per_block=1,
                latent_channels=4,
            ),
            True,
            4,
        ),
        "AutoencoderKLLTX2Video": (
            dict(
                block_out_channels=(8, 16, 32, 32),
                latent_channels=8,
                layers_per_block=(1, 1, 1, 1, 1),
                spatial_compression_ratio=32,
            ),
            True,
            8,
        ),
    }
    # Large enough that the output is several tiles across once the window is halved, since a
    # decode that fits in one tile would pass without tiling anything.
    LATENT_GRID = 16

    def test_a_halved_window_decodes_to_the_same_size(self):
        import torch

        for name, (kwargs, video, channels) in self.VAES.items():
            with self.subTest(vae=name):
                vae = _diffusers_vae(self, name, kwargs, require_tiling=True)

                grid = self.LATENT_GRID
                shape = (
                    (1, channels, 1, grid, grid) if video else (1, channels, grid, grid)
                )
                torch.manual_seed(0)
                latents = torch.randn(*shape)
                with torch.no_grad():
                    vae.disable_tiling()
                    expected = vae.decode(latents).sample.shape[-2:]

                # Same order as the caller: turn tiling on, then size its window.
                vae.enable_tiling()
                window = vae_tiling.tile_shape(vae)
                self.assertIsNotNone(
                    window, f"{name} tiles but exposes no window this can read"
                )
                shape = tuple(axis // 2 for axis in window)
                plan = vae_tiling.tile_shape_plan(vae, *shape)
                self.assertIsNotNone(
                    plan, f"{name} refused exact tile shape {shape}"
                )
                for attr, value in plan.items():
                    setattr(vae, attr, value)
                with torch.no_grad():
                    got = vae.decode(latents).sample.shape[-2:]
                self.assertEqual(
                    got, expected, f"{name} decoded at tile shape {shape}"
                )


class TestTileOverlap(unittest.TestCase):
    """The exact output-pixel overlap between neighbouring tiles."""

    def test_stride_and_overlap_factor_layouts_report_absolute_pixels(self):
        self.assertEqual(vae_tiling.tile_overlap(legacy_pair_vae()), (64, 64))
        self.assertEqual(vae_tiling.tile_overlap(stride_vae()), (64, 64))
        self.assertIsNone(vae_tiling.tile_overlap(StubVAE(tile_sample_min_size=256)))

    def test_reporting_a_step_is_not_knowing_what_moving_it_does(self):
        self.assertIsNotNone(vae_tiling.tile_overlap(stride_vae()))
        self.assertIsNone(vae_tiling.tile_overlap_plan(stride_vae(), 32, 32))
        self.assertIsNone(vae_tiling.tile_overlap_plan(overlap_hw_vae(), 32, 32))

    def test_a_column_strip_requires_zero_overlap_on_its_inactive_axis(self):
        vae = StubVAE(
            tile_sample_min_height=120,
            tile_sample_min_width=128,
            tile_latent_min_height=15,
            tile_latent_min_width=16,
            tile_overlap_factor=0.25,
            blend_v=lambda above, tile, extent: tile,
            blend_h=lambda left, tile, extent: tile,
        )
        self.assertEqual(
            vae_tiling.tile_overlap_plan(vae, 0, 16, sample_shape=(120, 512)),
            {
                "tile_overlap_factor_height": 0.0,
                "tile_overlap_factor_width": 0.125,
            },
        )
        self.assertIsNone(
            vae_tiling.tile_overlap_plan(vae, 8, 16, sample_shape=(120, 512))
        )

    def test_a_row_strip_accepts_a_distinct_height_overlap(self):
        vae = StubVAE(
            tile_sample_min_height=120,
            tile_sample_min_width=128,
            tile_latent_min_height=15,
            tile_latent_min_width=16,
            tile_overlap_factor=0.25,
            blend_v=lambda above, tile, extent: tile,
            blend_h=lambda left, tile, extent: tile,
        )
        self.assertEqual(
            vae_tiling.tile_overlap_plan(vae, 16, 0, sample_shape=(480, 128)),
            {
                "tile_overlap_factor_height": 2 / 15,
                "tile_overlap_factor_width": 0.0,
            },
        )

    def test_a_stride_walked_strip_sets_both_strides(self):
        cls = type("AutoencoderKLQwenImage", (StubVAE,), {})
        vae = cls(
            tile_sample_min_height=120,
            tile_sample_min_width=128,
            tile_sample_stride_height=96,
            tile_sample_stride_width=96,
            spatial_compression_ratio=8,
            blend_v=lambda above, tile, extent: tile,
            blend_h=lambda left, tile, extent: tile,
            decoder=lambda tile: tile,
            post_quant_conv=lambda tile: tile,
            clear_cache=lambda: None,
        )
        self.assertEqual(
            vae_tiling.tile_overlap_plan(vae, 0, 16, sample_shape=(120, 512)),
            {
                "tile_sample_stride_height": 120,
                "tile_sample_stride_width": 112,
            },
        )

    def test_a_single_tile_sets_zero_on_both_axes(self):
        self.assertEqual(
            vae_tiling.tile_overlap_plan(
                overlap_factor_vae(), 0, 0, sample_shape=(256, 256)
            ),
            {
                "tile_overlap_factor": 0.0,
                "tile_overlap_factor_height": 0.0,
                "tile_overlap_factor_width": 0.0,
            },
        )

    def test_exact_pixel_requests_keep_both_loop_truncations_agreeing(self):
        for build in (overlap_factor_vae, overlap_keyed_vae):
            for asked in (0, 16, 32, 64):
                with self.subTest(vae=build.__name__, overlap=asked):
                    vae = build()
                    plan = vae_tiling.tile_overlap_plan(vae, asked, asked)
                    self.assertIsNotNone(plan)
                    vae_tiling.apply_tile_plan(vae, plan)
                    factors = (
                        vae.tile_overlap_factor_height,
                        vae.tile_overlap_factor_width,
                    )
                    (down, across), (deep, wide) = vae_tiling.overlap_windows(vae)
                    for latent, pixel, factor in zip(
                        (down, across), (deep, wide), factors
                    ):
                        stride = int(latent * (1.0 - factor))
                        self.assertGreaterEqual(stride, 1)
                        self.assertEqual(
                            pixel - int(pixel * factor), stride * (pixel // latent)
                        )

    def test_unrepresentable_overlap_is_refused_without_rounding(self):
        self.assertIsNone(
            vae_tiling.tile_overlap_plan(overlap_factor_vae(), 1, 64)
        )

    def test_zero_overlap_is_a_step_of_the_whole_window(self):
        vae = overlap_factor_vae()
        vae_tiling.apply_tile_plan(vae, vae_tiling.tile_overlap_plan(vae, 0, 0))
        self.assertEqual(vae.tile_overlap_factor, 0.0)
        self.assertEqual(vae_tiling.tile_overlap(vae), (0, 0))

    def test_an_overlap_as_wide_as_the_window_is_refused(self):
        vae = overlap_factor_vae()
        self.assertIsNone(vae_tiling.tile_overlap_plan(vae, 256, 64))


class TestTiledDecode(unittest.TestCase):
    """Tests DistVAE's replacement for overlap-factor tiled-decode loops."""

    # These classes derive tile strides from overlap factors. HunyuanVideo 1.5 uses per-axis
    # window attributes and one shared overlap factor.
    FAMILY = ("AutoencoderKL", "AutoencoderKLFlux2", "AutoencoderKLHunyuanVideo15")
    # Three windows per axis exercise both full and clipped boundary tiles.
    WINDOWS_ACROSS = 3
    # Two frames exercise the video path without treating the frame axis as a singleton.
    FRAMES = 2

    def test_overlap_factor_detection_requires_a_supported_vae_class(self):
        # Stored-stride VAEs use a different replacement loop. CogVideoX is excluded because its
        # spatial loop also performs temporal tiling.
        self.assertTrue(vae_tiling.tiles_by_overlap_factor(overlap_factor_vae()))
        self.assertTrue(vae_tiling.tiles_by_overlap_factor(overlap_keyed_vae()))
        self.assertFalse(vae_tiling.tiles_by_overlap_factor(stride_vae()))
        self.assertFalse(vae_tiling.tiles_by_overlap_factor(overlap_hw_vae()))
        self.assertIsNone(vae_tiling.overlap_tiled_decode(stride_vae()))
        self.assertIsNotNone(vae_tiling.overlap_tiled_decode(overlap_factor_vae()))
        self.assertIsNotNone(vae_tiling.overlap_tiled_decode(overlap_keyed_vae()))

    def test_shared_and_per_axis_window_attributes_produce_the_same_pair(self):
        self.assertEqual(
            vae_tiling.overlap_windows(overlap_factor_vae()), ((32, 32), (256, 256))
        )
        self.assertEqual(
            vae_tiling.overlap_windows(overlap_keyed_vae()), ((16, 16), (256, 256))
        )
        self.assertIsNone(vae_tiling.overlap_windows(stride_vae()))

    def _sample(self, decoded):
        """Return the sample tensor from either supported tiled-decode return type."""
        return getattr(decoded, "sample", decoded)

    def _tiled_vae(self, name, batch=1):
        """Build a small tiled VAE and an input spanning several tiles."""
        import torch

        kwargs, video, channels = TestEverySupportedVAE.VAES[name]
        vae = _diffusers_vae(self, name, kwargs, require_tiling=True)
        vae.enable_tiling()

        window = vae_tiling.tile_shape(vae)
        shape = tuple(axis // 4 for axis in window)
        plan = vae_tiling.tile_shape_plan(vae, *shape)
        self.assertIsNotNone(
            plan, f"{name} refused exact tile shape {shape}"
        )
        vae_tiling.apply_tile_plan(vae, plan)
        self.assertTrue(
            vae_tiling.tiles_by_overlap_factor(vae),
            f"{name} was expected to tile by overlap fraction",
        )

        (latent_down, _), _ = vae_tiling.overlap_windows(vae)
        grid = latent_down * self.WINDOWS_ACROSS
        torch.manual_seed(0)
        shape = (
            (batch, channels, self.FRAMES, grid, grid)
            if video
            else (batch, channels, grid, grid)
        )
        return vae, torch.randn(*shape)

    def _counted(self, vae):
        """Replace the decoder with a wrapper that records each input shape."""
        import torch.nn as nn

        class CountingDecoder(nn.Module):
            def __init__(self, decoder):
                super().__init__()
                self.decoder = decoder
                self.shapes = []

            def forward(self, x):
                self.shapes.append(tuple(x.shape))
                return self.decoder(x)

            @property
            def rows(self):
                """Return the leading dimension of every decoder input."""
                return [shape[0] for shape in self.shapes]

        counted = CountingDecoder(vae.decoder)
        vae.decoder = counted
        return counted

    def test_it_decodes_a_tile_at_a_time_exactly_as_upstream_does(self):
        import torch

        # With no dispatcher, this loop must preserve the VAE's call boundaries and produce a
        # bit-identical sample. Each tile remains a separate decoder call because convolution
        # arithmetic depends on the rows grouped into that call.
        for name in self.FAMILY:
            with self.subTest(vae=name):
                vae, latents = self._tiled_vae(name)
                with torch.no_grad():
                    expected = self._sample(vae.tiled_decode(latents))
                    counted = self._counted(vae)
                    got = self._sample(vae_tiling.overlap_tiled_decode(vae)(latents))
                self.assertEqual(got.shape, expected.shape)
                torch.testing.assert_close(got, expected, rtol=0, atol=0)
                self.assertEqual(set(counted.rows), {1})

    def test_a_wider_step_decodes_fewer_tiles_to_the_same_image_size(self):
        import torch

        # Zero overlap increases the stride and reduces the tile count. Compare with the
        # upstream loop at the same settings to verify both output size and values.
        for name in self.FAMILY:
            with self.subTest(vae=name):
                vae, latents = self._tiled_vae(name)
                counted = self._counted(vae)
                with torch.no_grad():
                    before = self._sample(vae.tiled_decode(latents))
                at_own = len(counted.shapes)

                plan = vae_tiling.tile_overlap_plan(vae, 0, 0)
                self.assertIsNotNone(plan, f"{name} refused a step of its whole window")
                vae_tiling.apply_tile_plan(vae, plan)
                counted.shapes.clear()
                with torch.no_grad():
                    expected = self._sample(vae.tiled_decode(latents))
                at_zero = len(counted.shapes)
                counted.shapes.clear()
                with torch.no_grad():
                    got = self._sample(vae_tiling.overlap_tiled_decode(vae)(latents))

                self.assertGreater(
                    at_own, 0, f"{name} decoded nothing through its decoder"
                )
                self.assertLess(at_zero, at_own)
                self.assertEqual(len(counted.shapes), at_zero)
                self.assertEqual(got.shape, before.shape)
                torch.testing.assert_close(got, expected, rtol=0, atol=0)

    def test_the_replacement_preserves_the_upstream_return_type(self):
        import torch

        # Most tiled-decode methods return DecoderOutput when requested. HunyuanVideo 1.5 returns
        # a tensor directly. The replacement must preserve each class's return convention.
        for name in self.FAMILY:
            with self.subTest(vae=name):
                vae, latents = self._tiled_vae(name)
                wraps = vae_tiling._returns_decoder_output(vae)
                with torch.no_grad():
                    upstream = vae.tiled_decode(latents)
                    ours = vae_tiling.overlap_tiled_decode(vae)(latents)
                self.assertEqual(wraps, not isinstance(upstream, torch.Tensor))
                self.assertIs(type(ours), type(upstream))

    def test_a_latent_batch_is_decoded_as_it_stands(self):
        import torch

        # Each tile carries every sample in the batch, so a call already decodes as many rows as
        # there are samples and the decoder is handed the tensor upstream would have given it.
        vae, latents = self._tiled_vae("AutoencoderKL", batch=2)
        with torch.no_grad():
            expected = vae.tiled_decode(latents).sample
            counted = self._counted(vae)
            got = vae_tiling.overlap_tiled_decode(vae)(latents).sample
        self.assertEqual(set(counted.rows), {2})
        torch.testing.assert_close(got, expected, rtol=0, atol=0)

    def test_tile_parallel_support_requires_a_distvae_owned_loop(self):
        self.assertTrue(vae_tiling.supports_tile_parallel(overlap_factor_vae()))
        self.assertTrue(vae_tiling.supports_tile_parallel(overlap_keyed_vae()))
        self.assertFalse(vae_tiling.supports_tile_parallel(stride_vae()))
        self.assertFalse(vae_tiling.supports_tile_parallel(overlap_hw_vae()))

    def test_the_dispatcher_is_given_every_call_and_the_image_is_unchanged(self):
        import torch

        for name in self.FAMILY:
            with self.subTest(vae=name):
                vae, latents = self._tiled_vae(name)
                seen = []

                def dispatch(calls):
                    seen.append(len(calls))
                    return [call() for call in calls]

                with torch.no_grad():
                    expected = self._sample(vae.tiled_decode(latents))
                    counted = self._counted(vae)
                    got = self._sample(
                        vae_tiling.overlap_tiled_decode(vae, dispatch)(latents)
                    )
                # The replacement submits every tile in one dispatch call.
                self.assertEqual(seen, [len(counted.shapes)])
                torch.testing.assert_close(got, expected, rtol=0, atol=0)

    def test_the_calls_can_be_made_in_any_order(self):
        import torch

        # Independent tiles may execute in any order; assembly restores grid order.
        def backwards(calls):
            return list(reversed([call() for call in reversed(calls)]))

        for name in self.FAMILY:
            with self.subTest(vae=name):
                vae, latents = self._tiled_vae(name)
                with torch.no_grad():
                    expected = self._sample(vae.tiled_decode(latents))
                    got = self._sample(
                        vae_tiling.overlap_tiled_decode(vae, backwards)(latents)
                    )
                torch.testing.assert_close(got, expected, rtol=0, atol=0)

    def test_a_tiled_decode_that_fits_in_one_tile_still_works(self):
        import torch

        # A single tile means one call and no blending pass at all.
        vae, _ = self._tiled_vae("AutoencoderKL")
        latents = torch.randn(1, vae.config.latent_channels, 4, 4)
        with torch.no_grad():
            expected = vae.tiled_decode(latents).sample
            got = vae_tiling.overlap_tiled_decode(vae)(latents).sample
        torch.testing.assert_close(got, expected, rtol=0, atol=0)


class TestStrideTiledDecode(unittest.TestCase):
    """Tests DistVAE's replacement for stored-stride tiled-decode loops."""

    # Wan and Qwen-Image decode each tile frame by frame with a tile-local feature cache.
    # HunyuanVideo and LTX-2 decode each spatial tile in one call.
    FAMILY = (
        "AutoencoderKLWan",
        "AutoencoderKLQwenImage",
        "AutoencoderKLHunyuanVideo",
        "AutoencoderKLLTX2Video",
    )
    # LTX-2 passes a positional timestep embedding through tiled_decode.
    CONDITIONED = ("AutoencoderKLLTX2Video",)
    # The grid spans several tiles after halving the window. Two frames exercise cache reuse.
    LATENT_GRID = 16
    FRAMES = 2

    def _tiled_vae(self, name, **extra):
        """Build a small video VAE and an input spanning several tiles."""
        import torch

        kwargs, _, channels = TestEverySupportedVAE.VAES[name]
        vae = _diffusers_vae(self, name, {**kwargs, **extra}, require_tiling=True)
        vae.enable_tiling()

        window = vae_tiling.tile_shape(vae)
        shape = tuple(axis // 2 for axis in window)
        plan = vae_tiling.tile_shape_plan(vae, *shape)
        self.assertIsNotNone(
            plan, f"{name} refused exact tile shape {shape}"
        )
        vae_tiling.apply_tile_plan(vae, plan)
        self.assertTrue(
            vae_tiling.tiles_by_stored_stride(vae),
            f"{name} was expected to tile by a stride it stores",
        )

        torch.manual_seed(0)
        grid = self.LATENT_GRID
        return vae, torch.randn(1, channels, self.FRAMES, grid, grid)

    def _conditioning(self, vae):
        """Return positional conditioning arguments required by tiled_decode."""
        return (None,) if type(vae).__name__ in self.CONDITIONED else ()

    def test_stored_stride_detection_requires_a_supported_vae_class(self):
        # Matching stride attributes is insufficient; support is limited to known loop
        # implementations.
        self.assertFalse(vae_tiling.tiles_by_stored_stride(stride_vae()))
        self.assertFalse(vae_tiling.tiles_by_stored_stride(overlap_factor_vae()))
        # HunyuanVideo 1.5 derives its stride from an overlap factor.
        self.assertFalse(vae_tiling.tiles_by_stored_stride(overlap_keyed_vae()))

    def test_reimplemented_stride_tiling_matches_native_tiled_decode(self):
        import torch

        for name, extra in (
            ("AutoencoderKLWan", {}),
            # Wan 2.2 uses pixel unshuffle during decode. The channel count and spatial
            # compression ratio must include its patch size.
            (
                "AutoencoderKLWan",
                {
                    "patch_size": 2,
                    "in_channels": 12,
                    "out_channels": 12,
                    "scale_factor_spatial": 16,
                },
            ),
            ("AutoencoderKLQwenImage", {}),
            ("AutoencoderKLHunyuanVideo", {}),
            ("AutoencoderKLLTX2Video", {}),
        ):
            with self.subTest(vae=name, **extra):
                vae, latents = self._tiled_vae(name, **extra)
                args = self._conditioning(vae)
                with torch.no_grad():
                    expected = vae.tiled_decode(latents, *args).sample
                    got = vae_tiling.strided_tiled_decode(vae)(latents, *args).sample
                # Local execution must match the upstream loop exactly.
                self.assertEqual(got.shape, expected.shape)
                torch.testing.assert_close(got, expected, rtol=0, atol=0)

    def test_a_tile_is_a_call_and_they_can_be_made_in_any_order(self):
        import torch

        for name in self.FAMILY:
            with self.subTest(vae=name):
                vae, latents = self._tiled_vae(name)
                args = self._conditioning(vae)
                seen = []

                def backwards(calls):
                    seen.append(len(calls))
                    return list(reversed([call() for call in reversed(calls)]))

                with torch.no_grad():
                    expected = vae.tiled_decode(latents, *args).sample
                    got = vae_tiling.strided_tiled_decode(vae, backwards)(
                        latents, *args
                    ).sample
                # State may be shared between frames within one tile, but not between tiles.
                # Tile execution order therefore cannot affect the output.
                stride = vae.tile_sample_stride_height // vae.spatial_compression_ratio
                across = len(range(0, latents.shape[-1], stride))
                self.assertEqual(seen, [across * across])
                torch.testing.assert_close(got, expected, rtol=0, atol=0)

    def test_a_wider_step_decodes_fewer_tiles_to_the_same_image_size(self):
        import torch

        # Stored pixel strides are converted to latent-grid steps by the compression ratio and,
        # for pixel-unshuffle decoders, to crop steps by the patch size. Compare against the
        # upstream loop to catch inconsistent integer conversion.
        for name in self.FAMILY:
            with self.subTest(vae=name):
                vae, latents = self._tiled_vae(name)
                args = self._conditioning(vae)
                with torch.no_grad():
                    before = vae.tiled_decode(latents, *args).sample
                at_own = self._tiles_across(vae, latents)

                plan = vae_tiling.tile_overlap_plan(vae, 0, 0)
                self.assertIsNotNone(plan, f"{name} refused a step of its whole window")
                vae_tiling.apply_tile_plan(vae, plan)
                self.assertLess(self._tiles_across(vae, latents), at_own)

                with torch.no_grad():
                    expected = vae.tiled_decode(latents, *args).sample
                    got = vae_tiling.strided_tiled_decode(vae)(latents, *args).sample
                self.assertEqual(got.shape, before.shape)
                torch.testing.assert_close(got, expected, rtol=0, atol=0)

    def _tiles_across(self, vae, latents):
        """Return the number of tile columns at the VAE's current stride."""
        # Count grid positions because cached VAEs make several decoder calls per tile.
        stride = vae.tile_sample_stride_width // vae.spatial_compression_ratio
        return len(range(0, latents.shape[-1], stride))

    def test_temporal_chunking_reaches_the_installed_spatial_loop(self):
        import torch

        # HunyuanVideo's temporal loop calls the installed spatial loop once per frame chunk.
        vae, _ = self._tiled_vae("AutoencoderKLHunyuanVideo")
        ratio = vae.spatial_compression_ratio
        latent_stride = vae.tile_sample_stride_width // ratio
        chunk = vae.tile_sample_stride_num_frames // vae.temporal_compression_ratio
        # Exceed the spatial window by one latent pixel and provide two temporal chunks.
        grid = vae.tile_sample_min_width // ratio + 1
        torch.manual_seed(0)
        latents = torch.randn(1, vae.config.latent_channels, 2 * chunk, grid, grid)

        seen = []

        def counted(calls):
            seen.append(len(calls))
            return [call() for call in calls]

        with torch.no_grad():
            expected = vae.decode(latents).sample
            vae.tiled_decode = vae_tiling.strided_tiled_decode(vae, counted)
            got = vae.decode(latents).sample

        across = len(range(0, grid, latent_stride))
        self.assertGreater(across, 1, "the grid was too narrow to tile")
        self.assertEqual(
            seen,
            [across * across] * 2,
            "the temporal loop did not reach the installed loop",
        )
        torch.testing.assert_close(got, expected, rtol=0, atol=0)

    def test_the_native_stride_loop_is_kept_without_a_dispatcher(self):
        vae, _ = self._tiled_vae("AutoencoderKLWan")
        self.assertTrue(vae_tiling.supports_tile_parallel(vae))
        self.assertIsNone(vae_tiling.tiled_decode_for(vae))
        self.assertIsNotNone(vae_tiling.tiled_decode_for(vae, lambda calls: []))


if __name__ == "__main__":
    unittest.main()
