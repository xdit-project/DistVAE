import inspect
from pathlib import Path

import pytest

from distvae.modules.adapters import midblock_adapters
from distvae.modules.adapters.vae import decoder_adapters, encoder_adapters


ROOT = Path(__file__).parents[1]


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
