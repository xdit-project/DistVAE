"""Build every family in the bench's table on the meta device, without weights or a GPU.

A config key that the installed diffusers does not take, or a shape the class refuses, is a
wasted pod otherwise: the bench only finds out after the image pulls and the ranks line up.
Run it anywhere diffusers imports:

  python bench/smoke_families.py
"""

import sys

import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])

from distvae_bench import FAMILIES, sample_for  # noqa: E402


def main():
    import diffusers

    print(f"diffusers {diffusers.__version__}")
    failures = 0
    for name, spec in sorted(FAMILIES.items()):
        cls = getattr(diffusers, spec["cls"], None)
        if cls is None:
            print(f"  {name:<18} SKIP  {spec['cls']} is not in this diffusers")
            continue
        try:
            with torch.device("meta"):
                vae = cls(**spec["config"]).eval()
            latent = sample_for(spec, "decoder", 512, 512, torch.bfloat16, "meta", frames=17)
            pixels = sample_for(spec, "encoder", 512, 512, torch.bfloat16, "meta", frames=17)
            params = sum(p.numel() for p in vae.parameters())
            print(
                f"  {name:<18} OK    {params / 1e6:>7.1f}M params  "
                f"latent {tuple(latent.shape)}  input {tuple(pixels.shape)}"
            )
        except Exception as error:
            failures += 1
            print(f"  {name:<18} FAIL  {type(error).__name__}: {error}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
