import ast
import inspect
import subprocess
import sys
from pathlib import Path

import pytest

from distvae.modules.adapters import midblock_adapters
from distvae.modules.adapters.vae import decoder_adapters, encoder_adapters


ROOT = Path(__file__).parents[1]


def test_legacy_family_specific_model_copies_are_absent():
    stale_paths = [
        ROOT / "distvae/models/layers/wan/__init__.py",
        ROOT / "distvae/models/unets/unet_2d_blocks.py",
        ROOT / "distvae/models/upsampling.py",
    ]

    assert [path.relative_to(ROOT) for path in stale_paths if path.exists()] == []


def test_adapter_packages_do_not_import_implementations_eagerly():
    script = """
import sys
import distvae.modules.adapters
import distvae.modules.adapters.vae

loaded = set(sys.modules)
forbidden = {
    "distvae.modules.adapters.downsampling_adapters",
    "distvae.modules.adapters.upsampling_adapters",
    "distvae.modules.adapters.vae.decoder_adapters",
    "distvae.modules.adapters.vae.encoder_adapters",
}
assert loaded.isdisjoint(forbidden), sorted(loaded & forbidden)
"""

    subprocess.run([sys.executable, "-c", script], cwd=ROOT, check=True)


def test_family_specific_diffusers_classes_are_resolved_lazily():
    offenders = []
    adapter_root = ROOT / "distvae/modules/adapters"
    for path in adapter_root.rglob("*.py"):
        if path.name == "diffusers_blocks.py":
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.ImportFrom) or node.module is None:
                continue
            if node.module.startswith(
                "diffusers.models.autoencoders.autoencoder_kl_"
            ):
                offenders.append((path.relative_to(ROOT), node.lineno, node.module))

    assert offenders == []


def test_runtime_code_does_not_import_private_torch_symbols():
    offenders = []
    for path in (ROOT / "distvae").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.ImportFrom) or node.module is None:
                continue
            if not node.module.startswith("torch"):
                continue
            for alias in node.names:
                if alias.name.startswith("_"):
                    offenders.append((path.relative_to(ROOT), node.lineno, alias.name))

    assert offenders == []


def test_package_metadata_has_only_runtime_dependencies():
    setup = (ROOT / "setup.py").read_text()

    assert 'install_requires=["torch>=2.2", "diffusers>=0.30.3"]' in setup
    assert '"pipeline": ["transformers"]' in setup
    assert 'python_requires=">=3.10"' in setup


def test_readme_quickstart_selects_the_model_at_launch_time():
    readme = (ROOT / "README.md").read_text()
    prose = " ".join(readme.split())

    assert 'os.environ["MODEL_ID"]' in readme
    assert "stabilityai/stable-diffusion-xl-base-1.0" not in readme
    assert "Individual VAE families may require a newer Diffusers release." in prose


def test_ci_checks_minimum_and_latest_supported_dependencies():
    workflow = (ROOT / ".github/workflows/test.yml").read_text()

    assert 'python-version: "3.10"' in workflow
    assert "torch==2.2.*" in workflow
    assert "diffusers==0.30.3" in workflow
    assert 'python-version: "3.12"' in workflow
    assert "minimum-dependencies" in workflow
    assert "latest-dependencies" in workflow


def test_causal_vae_halves_share_the_same_setup_primitive():
    assert (
        encoder_adapters._CausalEncoderAdapter._setup_type
        is decoder_adapters._CausalDecoderAdapter._setup_type
    )


def test_hunyuan15_mid_block_reuses_the_configured_causal_base():
    assert issubclass(
        midblock_adapters.HunyuanVideo15MidBlockAdapter,
        midblock_adapters._CausalMidBlockAdapter,
    )


def test_hunyuan_and_ltx_resamplers_use_the_shared_child_conv_replacement():
    sources = [
        ROOT / "distvae/modules/adapters/upsampling_adapters.py",
        ROOT / "distvae/modules/adapters/downsampling_adapters.py",
    ]
    for source in sources:
        text = source.read_text()
        assert "replace_child_convolution" in text


def test_decoder_adapters_have_no_benchmark_side_effect_implementation():
    source = inspect.getsource(decoder_adapters)
    forbidden = (
        "torch.profiler",
        "ProfilerActivity",
        "tensorboard_trace_handler",
        "export_memory_timeline",
        "_record_memory_history",
        "get_peak_memory",
        "time.time",
        "print(",
    )
    assert all(token not in source for token in forbidden)


@pytest.mark.parametrize("option", ["use_profiler", "verbose"])
def test_removed_decoder_instrumentation_has_a_clear_migration_error(option):
    signature = inspect.signature(decoder_adapters.DecoderAdapter.__init__)
    assert option in signature.parameters

    with pytest.raises(ValueError, match="bench"):
        decoder_adapters.DecoderAdapter(object(), **{option: True})
