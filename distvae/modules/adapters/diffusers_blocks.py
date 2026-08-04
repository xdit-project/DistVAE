"""The diffusers VAE block classes the adapters are written against, resolved optionally.

DistVAE adapts several VAE families whose blocks arrived across a range of diffusers releases,
and an install new enough for one need not carry another. Importing them all eagerly would let
one missing family break every other family's adapter at import time, so each is resolved to
None instead and the adapter that wanted it says which class the installed diffusers is short
of, at the point someone tries to use it.
"""

import importlib
from typing import Optional, Tuple

WAN = "diffusers.models.autoencoders.autoencoder_kl_wan"
QWEN_IMAGE = "diffusers.models.autoencoders.autoencoder_kl_qwenimage"
HUNYUAN_VIDEO = "diffusers.models.autoencoders.autoencoder_kl_hunyuan_video"
HUNYUAN_VIDEO_15 = "diffusers.models.autoencoders.autoencoder_kl_hunyuanvideo15"
LTX2_VIDEO = "diffusers.models.autoencoders.autoencoder_kl_ltx2"


def block(module: str, name: str) -> Optional[type]:
    """The named class from a diffusers module, or None where this release has neither"""
    try:
        found = getattr(importlib.import_module(module), name, None)
    except ImportError:
        return None
    return found if isinstance(found, type) else None


def resolved(*blocks: Optional[type]) -> Tuple[type, ...]:
    """The blocks that were found, for an isinstance check the rest simply cannot pass"""
    return tuple(found for found in blocks if found is not None)


def require(supported: Tuple[type, ...], adapter: str, requires: str) -> None:
    """Refuse an adapter whose diffusers classes are not in this release

    Without this the isinstance check against an empty tuple would report that the block passed
    in was the wrong type, when the truth is that the right type does not exist here.
    """
    if not supported:
        import diffusers

        raise ImportError(
            f"{adapter} needs {requires}, which diffusers {diffusers.__version__} does not "
            f"provide. A newer diffusers is required to shard this VAE."
        )
