"""Public VAE orchestration APIs for DistVAE."""

from distvae.utils import ParallelContext

from .parallel import (
    decoder_adapter_name,
    encoder_adapter_name,
    encoder_scale_factor,
    parallelize_decoder,
    parallelize_encoder,
)
from .tile_parallel import (
    context_of,
    mark,
    sharing,
)
from .tiling import (
    apply_tile_plan,
    is_tile_padding_error,
    latent_rows,
    require_vae_support,
    supports_tile_parallel,
    tile_overlap,
    tile_overlap_plan,
    tile_shape,
    tile_shape_plan,
    tiled_decode_for,
)

__all__ = [
    "ParallelContext",
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
]
