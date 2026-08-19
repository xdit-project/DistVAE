"""VAE family specifications, construction, sampling, and adapter descriptions."""

from contextlib import nullcontext

import torch

# `shapes` is the family's canonical matrix, as (height, width, frames), and `--matrix` runs it.
# It lives here beside the architecture rather than in a caller's script so that pinning a commit
# pins the shapes too: two runs of the same SHA measured the same thing, on whatever machine, and
# a result that cannot say what it measured is one nobody can reproduce.
#
# Frames are carried even where they are ignored, so every entry reads the same. A family with no
# temporal axis discards them in `sample_for`; one with a temporal axis needs 1 plus a multiple of
# it. Qwen-Image is a 3D VAE that ships as a single-image model, which is why it asks for one
# frame rather than the video-shaped default.
FAMILIES = {
    "flux2": {
        "cls": "AutoencoderKLFlux2",
        "config": {
            "in_channels": 3,
            "out_channels": 3,
            "latent_channels": 32,
            "block_out_channels": [128, 256, 512, 512],
            "layers_per_block": 2,
            "norm_num_groups": 32,
            "down_block_types": ["DownEncoderBlock2D"] * 4,
            "up_block_types": ["UpDecoderBlock2D"] * 4,
            "patch_size": [2, 2],
            "mid_block_add_attention": True,
            "use_quant_conv": True,
            "use_post_quant_conv": True,
        },
        "latent_channels": 32,
        "spatial": 8,
        "temporal": None,
        "shapes": ((1024, 1024, 1), (2048, 2048, 1)),
        "note": "FLUX.2 checkpoints",
    },
    "kl": {
        "cls": "AutoencoderKL",
        "config": {
            "in_channels": 3,
            "out_channels": 3,
            "latent_channels": 16,
            "block_out_channels": [128, 256, 512, 512],
            "layers_per_block": 2,
            "norm_num_groups": 32,
            "down_block_types": ["DownEncoderBlock2D"] * 4,
            "up_block_types": ["UpDecoderBlock2D"] * 4,
            "sample_size": 1024,
        },
        "latent_channels": 16,
        "spatial": 8,
        "temporal": None,
        "shapes": ((1024, 1024, 1), (2048, 2048, 1)),
        "note": "plain 2D KL autoencoders",
    },
    "wan": {
        "cls": "AutoencoderKLWan",
        "config": {
            "base_dim": 96,
            "z_dim": 16,
            "dim_mult": [1, 2, 4, 4],
            "num_res_blocks": 2,
            "attn_scales": [],
            "temperal_downsample": [False, True, True],
        },
        "latent_channels": 16,
        "spatial": 8,
        "temporal": 4,
        # Portrait 480p and 720p at the production length. 81 frames is 21 latent ones, which is
        # enough that the unsharded case may not fit at 720p. That failure is recorded per cell,
        # and the tiled configurations continue.
        "shapes": ((832, 480, 81), (1280, 720, 81)),
        "note": "Wan video autoencoders",
    },
    "qwen_image": {
        "cls": "AutoencoderKLQwenImage",
        "config": {
            "base_dim": 96,
            "z_dim": 16,
            "dim_mult": [1, 2, 4, 4],
            "num_res_blocks": 2,
            "attn_scales": [],
            "temperal_downsample": [False, True, True],
        },
        "latent_channels": 16,
        "spatial": 8,
        "temporal": 4,
        "shapes": ((1024, 1024, 1), (2048, 2048, 1)),
        "note": "Qwen Image autoencoders",
    },
    "hunyuan_video": {
        "cls": "AutoencoderKLHunyuanVideo",
        "config": {
            "in_channels": 3,
            "out_channels": 3,
            "latent_channels": 16,
            "block_out_channels": [128, 256, 512, 512],
            "layers_per_block": 2,
            "norm_num_groups": 32,
            "mid_block_add_attention": True,
            "spatial_compression_ratio": 8,
            "temporal_compression_ratio": 4,
        },
        "latent_channels": 16,
        "spatial": 8,
        "temporal": 4,
        "shapes": ((832, 480, 129), (1280, 720, 129)),
        "note": "Hunyuan Video autoencoders",
    },
    "hunyuan_video_15": {
        "cls": "AutoencoderKLHunyuanVideo15",
        "config": {
            "in_channels": 3,
            "out_channels": 3,
            "latent_channels": 32,
            "block_out_channels": [128, 256, 512, 1024, 1024],
            "layers_per_block": 2,
            "downsample_match_channel": True,
            "upsample_match_channel": True,
            "spatial_compression_ratio": 16,
            "temporal_compression_ratio": 4,
        },
        "latent_channels": 32,
        "spatial": 16,
        "temporal": 4,
        "shapes": ((832, 480, 129), (1280, 720, 129)),
        "note": "Hunyuan Video 1.5 autoencoders",
    },
    "ltx2": {
        "cls": "AutoencoderKLLTX2Video",
        "config": {
            "in_channels": 3,
            "out_channels": 3,
            "latent_channels": 128,
            "block_out_channels": [256, 512, 1024, 2048],
            "decoder_block_out_channels": [256, 512, 1024],
            "layers_per_block": [4, 6, 6, 2, 2],
            "decoder_layers_per_block": [5, 5, 5, 5],
            "spatio_temporal_scaling": [True, True, True, True],
            "decoder_spatio_temporal_scaling": [True, True, True],
            "decoder_inject_noise": [False, False, False, False],
            "downsample_type": [
                "spatial",
                "temporal",
                "spatiotemporal",
                "spatiotemporal",
            ],
            "upsample_factor": [2, 2, 2],
            "upsample_residual": [True, True, True],
            "encoder_causal": True,
            "decoder_causal": False,
            "encoder_spatial_padding_mode": "zeros",
            "decoder_spatial_padding_mode": "reflect",
            "patch_size": 4,
            "patch_size_t": 1,
            "resnet_norm_eps": 1e-6,
            "spatial_compression_ratio": 32,
            "temporal_compression_ratio": 8,
        },
        "latent_channels": 128,
        "spatial": 32,
        "temporal": 8,
        "shapes": ((1536, 1024, 121), (1920, 1280, 121)),
        "note": "LTX-2 autoencoders",
    },
}


def _dtype(value):
    return getattr(torch, value) if isinstance(value, str) else value


def matrix_for(family):
    """Return a family's canonical shapes, checked against what its VAE can accept.

    Checked here rather than left to `sample_for` because a matrix is meant to be run unattended
    across machines: an axis that does not divide, or a frame count the temporal ratio rejects,
    should fail while the pod is still starting rather than partway through the third shape.
    """
    spec = FAMILIES[family]
    shapes = spec.get("shapes")
    if not shapes:
        raise ValueError(
            f"--family {family} has no canonical shapes; ask for --shape explicitly"
        )
    ratio, temporal = spec["spatial"], spec["temporal"]
    for height, width, frames in shapes:
        if height % ratio or width % ratio:
            raise ValueError(
                f"{family} shape {height}x{width} is not divisible by "
                f"compression ratio {ratio}"
            )
        if temporal and (frames - 1) % temporal:
            raise ValueError(
                f"{family} shape {height}x{width}x{frames} needs 1 plus a multiple "
                f"of {temporal} frames"
            )
    return tuple(shapes)


def build_vae(family, dtype, device):
    """Build a deterministic architecture with random weights."""
    import diffusers

    spec = FAMILIES[family]
    cls = getattr(diffusers, spec["cls"], None)
    if cls is None:
        raise ValueError(
            f"diffusers {diffusers.__version__} does not provide {spec['cls']} "
            f"required by --family {family}"
        )
    device = torch.device(device)
    torch.manual_seed(0)
    context = torch.device("meta") if device.type == "meta" else nullcontext()
    with context:
        vae = cls(**spec["config"]).eval()
    if device.type != "meta":
        vae = vae.to(device=device, dtype=_dtype(dtype))
    return vae


def sample_for(spec, half, height, width, dtype, device, batch=1, frames=1):
    """Create the input tensor for one encoder or decoder call."""
    ratio = spec["spatial"]
    if height % ratio or width % ratio:
        raise ValueError(
            f"{height}x{width} is not divisible by compression ratio {ratio}"
        )
    temporal = spec["temporal"]
    if temporal and (frames - 1) % temporal:
        raise ValueError(f"--frames {frames} must be 1 plus a multiple of {temporal}")
    if half == "decoder":
        channels = spec["latent_channels"]
        rows, columns = height // ratio, width // ratio
        depth = 1 + (frames - 1) // temporal if temporal else None
    else:
        channels = spec["config"].get("in_channels", 3)
        rows, columns = height, width
        depth = frames if temporal else None
    shape = (batch, channels, rows, columns)
    if depth is not None:
        shape = (batch, channels, depth, rows, columns)
    torch.manual_seed(1)
    return torch.randn(*shape, dtype=_dtype(dtype), device=device)


def run_half(vae, half, sample):
    """Run one VAE half and return the comparable tensor."""
    if half == "decoder":
        return vae.decode(sample).sample
    encoded = vae.encode(sample)
    distribution = getattr(encoded, "latent_dist", None)
    return distribution.mean if distribution is not None else encoded.latent


def describe_vae(vae, half):
    """Describe the intact half and DistVAE's selected public adapter."""
    from distvae import vae as vae_api

    part = getattr(vae, half)
    blocks = tuple(
        getattr(part, "up_blocks" if half == "decoder" else "down_blocks", None) or ()
    )

    def named(obj):
        cls = type(obj)
        return f"{cls.__module__}.{cls.__name__}"

    choose = (
        vae_api.decoder_adapter_name
        if half == "decoder"
        else vae_api.encoder_adapter_name
    )
    return {
        "class": named(part),
        "blocks": sorted({named(block) for block in blocks}),
        "mid_block": named(getattr(part, "mid_block", None)),
        "conv_norm_out": named(getattr(part, "conv_norm_out", None)),
        "adapter": choose(vae),
    }
