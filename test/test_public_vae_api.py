from packaging.version import Version

from distvae.__version__ import __version__
from distvae import vae


PUBLIC_VAE_API_VERSION = Version("0.0.0beta7")
PUBLIC_VAE_FUNCTIONS = {
    "ParallelContext",
    "apply_tile_plan",
    "context_of",
    "local_tiled_decode_for",
    "parallelize_decoder",
    "parallelize_encoder",
    "sharing",
    "snap_tile_window",
    "tile_overlap_plan",
    "tile_shape",
    "tile_shape_plan",
    "tile_window",
    "tiled_decode_for",
}


def test_package_version_identifies_the_public_vae_api():
    assert Version(__version__) >= PUBLIC_VAE_API_VERSION


def test_public_vae_api_exports_xdit_orchestration_functions():
    assert PUBLIC_VAE_FUNCTIONS <= set(vae.__all__)
    assert all(callable(getattr(vae, name)) for name in PUBLIC_VAE_FUNCTIONS)
