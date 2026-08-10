"""Every benchmark family builds on the meta device with representative samples."""

import pytest
import torch

from bench.harness.catalog import FAMILIES, sample_for

diffusers = pytest.importorskip("diffusers")


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_benchmark_family_builds_without_weights_or_an_accelerator(family):
    spec = FAMILIES[family]
    cls = getattr(diffusers, spec["cls"], None)
    if cls is None:
        pytest.skip(f"{spec['cls']} is not in this diffusers")

    with torch.device("meta"):
        vae = cls(**spec["config"]).eval()
    latent = sample_for(
        spec, "decoder", 512, 512, torch.bfloat16, "meta", frames=17
    )
    pixels = sample_for(
        spec, "encoder", 512, 512, torch.bfloat16, "meta", frames=17
    )

    assert sum(parameter.numel() for parameter in vae.parameters()) > 0
    assert latent.device.type == "meta"
    assert pixels.device.type == "meta"
