from argparse import Namespace
import importlib.util
from pathlib import Path


SPEC = importlib.util.spec_from_file_location(
    "distvae_bench", Path(__file__).parents[1] / "bench" / "distvae_bench.py"
)
distvae_bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(distvae_bench)


def test_describe_only_returns_a_report_that_needs_no_measurement_formatting(monkeypatch):
    description = {"class": "WanDecoder3d", "adapter": "WanDecoderAdapter"}
    monkeypatch.setattr(distvae_bench, "build_vae", lambda *args: object())
    monkeypatch.setattr(
        distvae_bench, "sample_for", lambda *args: distvae_bench.torch.empty(1)
    )
    monkeypatch.setattr(distvae_bench, "describe", lambda *args, **kwargs: description)
    args = Namespace(family="wan", half="decoder", batch=1, describe_only=True)
    cell = {"name": "single", "height": 256, "width": 256, "frames": 1, "parallel_vae": True}

    report = distvae_bench.measure_cell(
        args,
        spec={},
        cell=cell,
        device=object(),
        dtype=object(),
        group=object(),
        world_size=1,
        rank=0,
        say=lambda *parts: None,
        references={},
    )

    assert report == {
        "arm": "single",
        "family": "wan",
        "half": "decoder",
        "description": description,
    }
    distvae_bench.print_report(report, args.half)
