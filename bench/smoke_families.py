"""Build every catalog family on the meta device without weights or an accelerator."""

import torch

if __package__:
    from .harness.catalog import FAMILIES, sample_for
else:
    from harness.catalog import FAMILIES, sample_for


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
            latent = sample_for(
                spec, "decoder", 512, 512, torch.bfloat16, "meta", frames=17
            )
            pixels = sample_for(
                spec, "encoder", 512, 512, torch.bfloat16, "meta", frames=17
            )
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
