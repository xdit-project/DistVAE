"""VAE family specifications, construction, sampling, and adapter descriptions."""

from contextlib import nullcontext

import torch

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
        "note": "LTX-2 autoencoders",
    },
}


def _dtype(value):
    return getattr(torch, value) if isinstance(value, str) else value


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
