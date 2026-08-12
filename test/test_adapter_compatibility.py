import subprocess
import sys
from pathlib import Path

import pytest

from distvae.modules.adapters.vae import decoder_adapters


ROOT = Path(__file__).parents[1]


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


@pytest.mark.parametrize("option", ["use_profiler", "verbose"])
def test_decoder_instrumentation_options_point_to_the_benchmark_harness(option):
    with pytest.raises(ValueError, match="bench"):
        decoder_adapters.DecoderAdapter(object(), **{option: True})
