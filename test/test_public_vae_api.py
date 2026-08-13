from packaging.version import Version

from distvae.__version__ import __version__
from distvae import vae


MINIMUM_PUBLIC_VAE_API_VERSION = Version("0.0.0beta9")
PUBLIC_VAE_FUNCTIONS = {
    "ParallelContext",
    "VAERowSplitError",
    "apply_tile_plan",
    "context_of",
    "decoder_adapter_name",
    "encoder_adapter_name",
    "encoder_scale_factor",
    "is_tile_padding_error",
    "latent_rows",
    "mark",
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


def test_package_version_meets_the_public_vae_api_minimum():
    assert Version(__version__) >= MINIMUM_PUBLIC_VAE_API_VERSION


def test_public_vae_api_exports_xdit_orchestration_functions():
    assert set(vae.__all__) == PUBLIC_VAE_FUNCTIONS
    assert all(callable(getattr(vae, name)) for name in PUBLIC_VAE_FUNCTIONS)
