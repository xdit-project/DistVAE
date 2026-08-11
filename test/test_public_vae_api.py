from packaging.version import Version

from distvae.__version__ import __version__
from distvae import vae


PUBLIC_VAE_API_VERSION = Version("0.0.0beta9")
PUBLIC_VAE_FUNCTIONS = {
    "ParallelContext",
    "apply_tile_plan",
    "context_of",
    "decoder_adapter_name",
    "encoder_adapter_name",
    "encoder_scale_factor",
    "is_tile_padding_error",
    "parallelize_decoder",
    "parallelize_encoder",
    "require_vae_support",
    "sharing",
    "supports_tile_parallel",
    "tile_overlap",
    "tile_overlap_plan",
    "tile_shape",
    "tile_shape_plan",
    "tiled_decode_for",
}


def test_package_version_identifies_the_public_vae_api():
    assert Version(__version__) == PUBLIC_VAE_API_VERSION


def test_public_vae_api_exports_xdit_orchestration_functions():
    assert set(vae.__all__) == PUBLIC_VAE_FUNCTIONS
    assert all(callable(getattr(vae, name)) for name in PUBLIC_VAE_FUNCTIONS)


def test_removed_vae_facade_names_are_absent():
    removed = {
        "Blend",
        "assemble_here",
        "assemble_in_runs",
        "dispatch_over",
        "group_of",
        "in_order",
        "latent_rows",
        "local_tiled_decode_for",
        "mark",
        "narrowest_useful_window",
        "overlap_tiled_decode",
        "overlap_windows",
        "runs",
        "shares",
        "smallest_tile_window",
        "snap_tile_window",
        "spatial_ratio",
        "strided_tiled_decode",
        "tile_plan",
        "tile_window",
        "tiles_by_overlap_factor",
        "tiles_by_stored_stride",
        "widest_tile_overlap",
    }
    assert removed.isdisjoint(vae.__all__)
    assert all(not hasattr(vae, name) for name in removed)
