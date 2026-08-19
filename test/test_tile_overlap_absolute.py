from types import SimpleNamespace

import torch
import torch.nn.functional as functional

from distvae.vae import tiling


class StubVAE:
    def __init__(self, **attrs):
        for name, value in attrs.items():
            setattr(self, name, value)


def overlap_vae(height=256, width=256, latent_height=32, latent_width=32):
    return StubVAE(
        tile_sample_min_height=height,
        tile_sample_min_width=width,
        tile_latent_min_height=latent_height,
        tile_latent_min_width=latent_width,
        tile_overlap_factor=0.25,
        blend_v=lambda above, tile, extent: tile,
        blend_h=lambda left, tile, extent: tile,
    )


def stride_vae():
    cls = type("AutoencoderKLQwenImage", (StubVAE,), {})
    return cls(
        tile_sample_min_height=256,
        tile_sample_min_width=384,
        tile_sample_stride_height=192,
        tile_sample_stride_width=288,
        spatial_compression_ratio=8,
        config=SimpleNamespace(),
        blend_v=lambda above, tile, extent: tile,
        blend_h=lambda left, tile, extent: tile,
        decoder=lambda tile: tile,
        post_quant_conv=lambda tile: tile,
        clear_cache=lambda: None,
    )


def test_tile_overlap_reports_absolute_pixels_for_both_storage_families():
    assert tiling.tile_overlap(overlap_vae(height=240, width=320)) == (60, 80)
    assert tiling.tile_overlap(stride_vae()) == (64, 96)


def test_exact_per_axis_overlap_plans_keyed_factors_and_scalar_only_when_equal():
    vae = overlap_vae(height=240, width=320, latent_height=30, latent_width=40)

    plan = tiling.tile_overlap_plan(vae, 40, 64)

    assert plan == {
        "tile_overlap_factor_height": 1 / 6,
        "tile_overlap_factor_width": 0.2,
    }
    tiling.apply_tile_plan(vae, plan)
    assert tiling.tile_overlap(vae) == (40, 64)
    assert vae.tile_overlap_factor == 0.25

    square = overlap_vae()
    equal = tiling.tile_overlap_plan(square, 64, 64)
    assert equal["tile_overlap_factor"] == 0.25


def test_overlap_plan_is_exact_and_rejects_unrepresentable_requests():
    vae = overlap_vae(height=240, width=320, latent_height=30, latent_width=40)

    assert tiling.tile_overlap_plan(vae, 41, 64) is None
    assert tiling.tile_overlap_plan(vae, 240, 64) is None
    assert tiling.tile_overlap_plan(vae, True, 64) is None
    assert tiling.tile_overlap_plan(vae, -1, 64) is None


def test_sample_shape_requires_zero_on_inactive_axes_and_sets_both_factors():
    vae = overlap_vae(height=240, width=320, latent_height=30, latent_width=40)

    assert tiling.tile_overlap_plan(vae, 0, 64, sample_shape=(240, 640)) == {
        "tile_overlap_factor_height": 0.0,
        "tile_overlap_factor_width": 0.2,
    }
    assert tiling.tile_overlap_plan(vae, 40, 64, sample_shape=(240, 640)) is None
    assert tiling.tile_overlap_plan(vae, 0, 0, sample_shape=(240, 320)) == {
        "tile_overlap_factor": 0.0,
        "tile_overlap_factor_height": 0.0,
        "tile_overlap_factor_width": 0.0,
    }


def test_stored_stride_plan_sets_both_axes_and_requires_exact_granularity():
    vae = stride_vae()

    assert tiling.tile_overlap_plan(vae, 64, 128) == {
        "tile_sample_stride_height": 192,
        "tile_sample_stride_width": 256,
    }
    assert tiling.tile_overlap_plan(vae, 63, 128) is None
    assert tiling.tile_overlap_plan(vae, 0, 128, sample_shape=(256, 768)) == {
        "tile_sample_stride_height": 256,
        "tile_sample_stride_width": 256,
    }


def test_replacement_decode_uses_rectangular_keyed_overlap_factors():
    vae = overlap_vae(height=16, width=24, latent_height=2, latent_width=3)
    vae.decoder = lambda tile: functional.interpolate(tile, scale_factor=8, mode="nearest")
    plan = tiling.tile_overlap_plan(vae, 8, 8)
    assert plan is not None
    tiling.apply_tile_plan(vae, plan)

    decode = tiling.tiled_decode_for(vae)
    sample = decode(torch.randn(1, 4, 4, 6)).sample

    assert sample.shape == (1, 4, 32, 48)
