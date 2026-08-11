import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from bench.harness import (
    arms,
    catalog,
    cli,
    distributed,
    measure,
    profile,
    report,
    shape_costs,
)


def test_harness_has_no_optional_runner_dependency():
    root = Path(__file__).parents[1] / "bench"
    forbidden = ("x" + "fuser", "x" + "dit")
    for path in root.rglob("*.py"):
        text = path.read_text().lower()
        assert all(word not in text for word in forbidden), path


def test_smoke_families_imports_catalog_without_path_mutation():
    source = (Path(__file__).parents[1] / "bench" / "smoke_families.py").read_text()
    assert "sys.path" not in source
    assert "harness.catalog" in source


def test_describe_only_runs_on_cpu_without_distributed_environment(
    tmp_path, monkeypatch
):
    for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(
        measure.torch.cuda,
        "set_device",
        lambda *args: pytest.fail("describe-only touched CUDA"),
    )
    output = tmp_path / "description.json"

    status = cli.main(
        ["--describe-only", "--family", "kl", "--half", "decoder", "--out", str(output)]
    )

    assert status == 0
    record = json.loads(output.read_text())
    assert record["schema_version"] == report.SCHEMA_VERSION
    assert record["measurement"]["description"]["adapter"] == "DecoderAdapter"
    assert record["composition"]["execution"] == "describe-only"
    assert record["runtime"] == {"dtype": "bfloat16", "world_size": 1}


def test_catalog_samples_decoder_and_encoder_on_meta():
    spec = catalog.FAMILIES["kl"]
    latent = catalog.sample_for(spec, "decoder", 512, 256, "float32", "meta")
    image = catalog.sample_for(spec, "encoder", 512, 256, "float32", "meta")
    assert tuple(latent.shape) == (1, 16, 64, 32)
    assert tuple(image.shape) == (1, 3, 512, 256)


def test_arm_axes_expand_orthogonally():
    cells = arms.expand_grid(
        arm_names="none,pvae,tile-nopvae,tile-dist",
        shapes="512x256",
        default_frames=1,
        overlaps="0x0,64x32",
    )
    base = {(cell["sharding"], cell["tiling"]) for cell in cells}
    assert ("unsharded", None) in base
    assert ("row", None) in base
    assert ("unsharded", "native") in base
    assert any(
        cell["sharding"] == "unsharded"
        and cell["tiling"] == "native"
        and cell["tile_distribution"] == "runs"
        for cell in cells
    )
    assert all(cell["overlap"] is None for cell in cells if cell["tiling"] is None)
    assert {cell["overlap"] for cell in cells if cell["tiling"]} == {
        None,
        (0, 0),
        (64, 32),
    }


def test_overlap_grid_labels_explicit_pixel_pairs():
    cells = arms.expand_grid("tile", "512x256", 1, "64x32")

    assert [cell["overlap"] for cell in cells] == [None, (64, 32)]
    assert cells[1]["name"] == "tile-ov64x32"


def test_parser_exposes_independent_composition_axes():
    args = cli.parser().parse_args(
        [
            "--sharding",
            "unsharded",
            "--tile-window",
            "256",
            "--tile-overlap",
            "64x32",
            "--tile-distribution",
            "runs",
        ]
    )

    [cell] = arms.cells_from_args(args)

    assert cell["sharding"] == "unsharded"
    assert cell["tiling"] == 256
    assert cell["overlap"] == (64, 32)
    assert cell["tile_distribution"] == "runs"


def test_non_grid_rejects_multiple_overlap_pairs():
    args = cli.parser().parse_args(
        ["--enable-tiling", "--tile-overlap", "64x32,32x16"]
    )

    with pytest.raises(ValueError, match=r"multiple.*--grid-arms"):
        arms.cells_from_args(args)


@pytest.mark.parametrize(
    ("legacy", "sharding", "distribution"),
    [
        ("tiles", "unsharded", "runs"),
        ("scattered", "unsharded", "scattered"),
        ("rows", "row", None),
    ],
)
def test_legacy_tile_split_selects_a_complete_composition(
    legacy, sharding, distribution
):
    args = cli.parser().parse_args(["--enable-tiling", "--tile-split", legacy])

    [cell] = arms.cells_from_args(args)

    assert (cell["sharding"], cell["tile_distribution"]) == (
        sharding,
        distribution,
    )


@pytest.mark.parametrize(
    "arguments",
    [
        ["--enable-tiling", "--tile-split", "tiles", "--sharding", "row"],
        ["--enable-tiling", "--tile-split", "rows", "--tile-distribution", "runs"],
        ["--enable-tiling", "--tile-split", "rows", "--no-parallel-vae"],
    ],
)
def test_legacy_tile_split_rejects_conflicting_explicit_axes(arguments):
    args = cli.parser().parse_args(arguments)

    with pytest.raises(ValueError, match="conflicts"):
        arms.cells_from_args(args)


def test_unknown_arms_are_rejected_without_expanding_supported_choices():
    assert set(arms.ARM_ALIASES) == {
        "none",
        "pvae",
        "tile",
        "tile-half",
        "tile-quarter",
        "tile-nopvae",
        "tile-dist",
        "tile-dist-half",
        "tile-dist-quarter",
    }
    for name in ("removed-arm", "legacy-comparison"):
        with pytest.raises(ValueError, match="unknown arm"):
            arms.expand_grid(name, "512x512", 1, None)


def test_parser_exposes_tile_shape_cost_controls():
    args = cli.parser().parse_args(
        [
            "--tile-shape-costs",
            "--tile-shape-batch",
            "4",
            "--tile-shape-sides",
            "8,16",
        ]
    )

    assert args.tile_shape_costs is True
    assert args.tile_shape_batch == 4
    assert args.tile_shape_sides == "8,16"


def test_tile_shape_cost_mode_bypasses_ordinary_cell_normalization(monkeypatch):
    runtime = SimpleNamespace(rank=0, world_size=1, group=object())
    runtime.close = lambda: None
    monkeypatch.setattr(
        arms,
        "cells_from_args",
        lambda args: pytest.fail("shape-cost mode normalized ordinary cells"),
    )
    monkeypatch.setattr(cli.Runtime, "start", lambda timeout: runtime)
    monkeypatch.setattr(cli, "_measure", lambda *args, **kwargs: [])
    monkeypatch.setattr(report, "report_status", lambda records: 0)
    monkeypatch.setattr(
        cli.dist,
        "all_gather_object",
        lambda values, value, **kwargs: values.__setitem__(0, value),
    )

    assert cli.main(["--tile-shape-costs", "--tile-overlap", "irrelevant"]) == 0


def test_tile_shape_cost_mode_rejects_describe_only():
    with pytest.raises(SystemExit):
        cli.main(["--tile-shape-costs", "--describe-only"])


def test_invocation_provenance_is_collected_once_and_reused(monkeypatch):
    calls = []
    provenance_data = {"versions": {}, "provenance": {"recorded_at": "once"}}
    monkeypatch.setattr(
        report,
        "provenance",
        lambda: calls.append("provenance") or provenance_data,
    )
    monkeypatch.setattr(
        cli,
        "_describe",
        lambda args, cells, shared: [
            {"measurement": {}, **shared},
            {"measurement": {}, **shared},
        ],
    )
    monkeypatch.setattr(report, "render", lambda *args: None)

    assert cli.main(["--describe-only", "--grid-arms", "none,pvae"]) == 0
    assert calls == ["provenance"]


@pytest.mark.parametrize(
    "axis",
    [
        ["--sharding", "row"],
        ["--no-parallel-vae"],
        ["--enable-tiling"],
        ["--tile-window", "256"],
        ["--vae-tile-size", "256"],
        ["--tile-distribution", "runs"],
        ["--tile-split", "rows"],
    ],
)
def test_grid_arms_reject_ambiguous_explicit_composition_axes(axis):
    args = cli.parser().parse_args(["--grid-arms", "none", *axis])

    with pytest.raises(ValueError, match="--grid-arms"):
        arms.cells_from_args(args)


def test_rank_error_helpers_preserve_original_rank_and_type(monkeypatch):
    peer = {"type": "ValueError", "message": "peer", "rank": 1}
    runtime = SimpleNamespace(rank=0, world_size=2, group=object())
    monkeypatch.setattr(
        distributed.dist,
        "all_gather_object",
        lambda values, value, **kwargs: values.__setitem__(slice(None), [value, peer]),
    )

    failures = distributed.gather_rank_errors(
        distributed.exception_record(RuntimeError("local"), runtime.rank), runtime
    )
    aggregate = distributed.aggregate_rank_errors(failures)

    assert aggregate["type"] == "RuntimeError"
    assert aggregate["rank"] == 0
    assert aggregate["failed_ranks"] == [0, 1]
    assert aggregate["failures"] == [
        {"type": "RuntimeError", "message": "local", "rank": 0},
        peer,
    ]


def test_legacy_arm_names_are_normalized_at_the_cli_boundary():
    args = cli.parser().parse_args(["--grid-arms", "none,pvae,tile-dist"])

    arms.normalize_legacy_args(args)

    assert args.grid_arms == "unsharded,row,tile-runs"


def test_extracted_benchmark_modules_own_shape_costs_and_profiling():
    assert not hasattr(measure, "tile_shape_costs")
    assert not hasattr(measure, "profile_once")
    assert callable(shape_costs.tile_shape_costs)
    assert callable(profile.profile_once)


def test_parser_exposes_harness_owned_profiler_controls(tmp_path):
    args = cli.parser().parse_args(
        [
            "--profile",
            "--profile-trace",
            "--profile-memory",
            "--profile-dir",
            str(tmp_path),
        ]
    )

    assert args.profile is True
    assert args.profile_trace is True
    assert args.profile_memory is True
    assert args.profile_dir == str(tmp_path)


def test_disabled_profiler_has_no_runtime_overhead(monkeypatch):
    monkeypatch.setattr(
        profile.torch.profiler,
        "profile",
        lambda **kwargs: pytest.fail("disabled profiling touched torch.profiler"),
    )
    args = SimpleNamespace(
        profile=False,
        profile_trace=False,
        profile_memory=False,
    )

    assert (
        profile.profile_once(lambda: pytest.fail("disabled profiling ran"), args)
        is None
    )


def test_profile_without_exports_returns_a_bounded_summary(monkeypatch):
    table_calls = []

    class Averages:
        def table(self, **options):
            table_calls.append(options)
            return "x" * (profile.PROFILE_SUMMARY_LIMIT + 100)

    class FakeProfile:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def key_averages(self):
            return Averages()

    monkeypatch.setattr(
        profile.torch.profiler,
        "ProfilerActivity",
        SimpleNamespace(CPU="cpu", CUDA="cuda"),
    )
    monkeypatch.setattr(
        profile.torch.profiler, "profile", lambda **kwargs: FakeProfile()
    )
    args = SimpleNamespace(
        profile=True,
        profile_trace=False,
        profile_memory=False,
        profile_dir="unused",
        family="kl",
        half="encoder",
    )

    result = profile.profile_once(
        lambda: object(),
        args,
        cell={"name": "single", "height": 256, "width": 128, "frames": 1},
        runtime=SimpleNamespace(rank=0, device=SimpleNamespace(type="cuda")),
    )

    assert result["artifacts"] == {}
    assert len(result["summary"]) == profile.PROFILE_SUMMARY_LIMIT
    assert table_calls == [{"sort_by": "self_cuda_time_total", "row_limit": 20}]


def test_profiler_setup_failure_on_a_peer_cleans_up_before_run(monkeypatch):
    events = []
    history = []
    peer_failure = {
        "type": "RuntimeError",
        "message": "profiler enter failed",
        "rank": 1,
    }

    class FakeProfile:
        def __enter__(self):
            events.append("enter")
            return self

        def __exit__(self, *args):
            events.append("exit")
            return False

    def gather(values, value, **kwargs):
        values[:] = [value, peer_failure]

    monkeypatch.setattr(
        profile.torch.profiler,
        "ProfilerActivity",
        SimpleNamespace(CPU="cpu", CUDA="cuda"),
    )
    monkeypatch.setattr(
        profile.torch.cuda.memory,
        "_record_memory_history",
        lambda enabled=None: history.append(enabled),
    )
    monkeypatch.setattr(profile.torch.profiler, "profile", lambda **kwargs: FakeProfile())
    monkeypatch.setattr(profile.torch.distributed, "all_gather_object", gather)
    args = SimpleNamespace(
        profile=True,
        profile_trace=False,
        profile_memory=True,
        profile_dir="unused",
        family="kl",
        half="decoder",
    )
    runtime = SimpleNamespace(
        rank=0,
        world_size=2,
        group=object(),
        device=SimpleNamespace(type="cuda"),
    )

    with pytest.raises(distributed.RankError) as caught:
        profile.profile_once(
            lambda: events.append("run"),
            args,
            cell={"name": "single", "height": 16, "width": 16, "frames": 1},
            runtime=runtime,
        )

    assert caught.value.rank_error["rank"] == 1
    assert caught.value.rank_error["type"] == "RuntimeError"
    assert caught.value.rank_error["failures"] == [peer_failure]
    assert events == ["enter", "exit"]
    assert history == ["all", None]


@pytest.mark.parametrize("failure_step", ["directory", "memory", "enter"])
def test_profiler_local_setup_failure_is_synchronized_with_peers(
    monkeypatch, failure_step
):
    history = []
    events = []

    class FakeProfile:
        def __enter__(self):
            events.append("enter")
            if failure_step == "enter":
                raise ValueError("context setup")
            return self

        def __exit__(self, *args):
            events.append("exit")
            return False

    def record_memory(enabled=None):
        history.append(enabled)
        if failure_step == "memory" and enabled == "all":
            raise RuntimeError("memory setup")

    def gather(values, value, **kwargs):
        values[:] = [value, None]

    monkeypatch.setattr(
        profile.torch.profiler,
        "ProfilerActivity",
        SimpleNamespace(CPU="cpu", CUDA="cuda"),
    )
    monkeypatch.setattr(
        profile.torch.cuda.memory, "_record_memory_history", record_memory
    )
    monkeypatch.setattr(profile.torch.profiler, "profile", lambda **kwargs: FakeProfile())
    monkeypatch.setattr(profile.torch.distributed, "all_gather_object", gather)
    if failure_step == "directory":
        monkeypatch.setattr(
            profile.Path,
            "mkdir",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                OSError("artifact directory setup")
            ),
        )
    args = SimpleNamespace(
        profile=True,
        profile_trace=False,
        profile_memory=True,
        profile_dir="unused",
        family="kl",
        half="decoder",
    )
    runtime = SimpleNamespace(
        rank=0,
        world_size=2,
        group=object(),
        device=SimpleNamespace(type="cuda"),
    )

    with pytest.raises(distributed.RankError) as caught:
        profile.profile_once(
            lambda: pytest.fail("run began after setup failure"),
            args,
            cell={"name": "single", "height": 16, "width": 16, "frames": 1},
            runtime=runtime,
        )

    expected_type = {
        "directory": "OSError",
        "memory": "RuntimeError",
        "enter": "ValueError",
    }[failure_step]
    assert caught.value.rank_error["rank"] == 0
    assert caught.value.rank_error["type"] == expected_type
    assert caught.value.rank_error["failed_ranks"] == [0]
    if failure_step == "enter":
        assert history == ["all", None]
        assert events == ["enter"]


def test_profiler_exports_harness_named_trace_and_memory_artifacts(
    tmp_path, monkeypatch
):
    exports = {}
    history = []

    class FakeProfile:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def export_chrome_trace(self, path):
            exports["trace"] = Path(path)

        def export_memory_timeline(self, path):
            exports["memory"] = Path(path)

        def key_averages(self):
            return SimpleNamespace(table=lambda **kwargs: "cuda summary")

    monkeypatch.setattr(
        profile.torch.profiler,
        "ProfilerActivity",
        SimpleNamespace(CPU="cpu", CUDA="cuda"),
    )
    monkeypatch.setattr(
        profile.torch.cuda.memory,
        "_record_memory_history",
        lambda enabled=None: history.append(enabled),
    )
    monkeypatch.setattr(
        profile.importlib,
        "import_module",
        lambda name: pytest.fail(f"CUDA profiling imported {name}"),
    )
    monkeypatch.setattr(
        profile.torch.profiler,
        "profile",
        lambda **kwargs: exports.update(options=kwargs) or FakeProfile(),
    )
    args = SimpleNamespace(
        profile=True,
        profile_trace=True,
        profile_memory=True,
        profile_dir=str(tmp_path),
        family="wan",
        half="decoder",
    )
    runtime = SimpleNamespace(rank=2, device=SimpleNamespace(type="cuda"))
    result = profile.profile_once(
        lambda: object(),
        args,
        cell={"name": "tile-half", "height": 512, "width": 256, "frames": 17},
        runtime=runtime,
    )

    assert result == {
        "summary": "cuda summary",
        "artifacts": {
            "trace": str(
                tmp_path / "wan-decoder-tile-half-512x256x17-rank2.trace.json"
            ),
            "memory": str(
                tmp_path / "wan-decoder-tile-half-512x256x17-rank2.memory.html"
            ),
        },
    }
    assert exports["trace"] == Path(result["artifacts"]["trace"])
    assert exports["memory"] == Path(result["artifacts"]["memory"])
    assert exports["options"]["activities"] == ["cpu", "cuda"]
    assert exports["options"]["profile_memory"] is True
    assert exports["options"]["record_shapes"] is True
    assert exports["options"]["with_stack"] is True
    assert history == ["all", None]


def test_profiler_avoids_overwriting_existing_artifacts(tmp_path):
    existing = tmp_path / "kl-decoder-single-16x16x1-rank0.trace.json"
    existing.write_text("old")

    stem = profile._artifact_stem(
        tmp_path, "kl-decoder-single-16x16x1-rank0", ["trace.json"]
    )

    assert stem == "kl-decoder-single-16x16x1-rank0-2"


def test_musa_profiler_is_loaded_lazily_and_uses_musa_memory_history(
    tmp_path, monkeypatch
):
    exports = {}
    history = []
    imported = []

    class FakeProfile:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def key_averages(self):
            return SimpleNamespace(table=lambda **kwargs: "musa summary")

        def export_memory_timeline(self, path):
            exports["memory"] = Path(path)

    musa = SimpleNamespace(
        memory=SimpleNamespace(
            _record_memory_history=lambda enabled=None: history.append(enabled)
        )
    )
    monkeypatch.setattr(profile.torch, "musa", musa, raising=False)
    monkeypatch.setattr(
        profile.importlib,
        "import_module",
        lambda name: imported.append(name) or object(),
    )
    monkeypatch.setattr(
        profile.torch.profiler,
        "ProfilerActivity",
        SimpleNamespace(CPU="cpu", MUSA="musa"),
    )
    monkeypatch.setattr(
        profile.torch.profiler,
        "profile",
        lambda **kwargs: exports.update(options=kwargs) or FakeProfile(),
    )
    args = SimpleNamespace(
        profile=True,
        profile_trace=False,
        profile_memory=True,
        profile_dir=str(tmp_path),
        family="wan",
        half="decoder",
    )

    result = profile.profile_once(
        lambda: object(),
        args,
        cell={"name": "single", "height": 64, "width": 64, "frames": 5},
        runtime=SimpleNamespace(rank=1, device=SimpleNamespace(type="musa")),
    )

    assert imported == ["torch_musa"]
    assert exports["options"]["activities"] == ["cpu", "musa"]
    assert exports["memory"] == Path(result["artifacts"]["memory"])
    assert history == ["all", None]
    assert result["summary"] == "musa summary"


def test_runtime_selects_cuda_without_importing_musa(monkeypatch):
    monkeypatch.setattr(distributed.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        distributed.importlib,
        "import_module",
        lambda name: pytest.fail(f"CUDA runtime imported {name}"),
    )

    name, api, backend = distributed.accelerator_backend()

    assert (name, api, backend) == ("cuda", distributed.torch.cuda, "nccl")


def test_runtime_loads_musa_lazily_and_selects_mccl(monkeypatch):
    imported = []
    musa = SimpleNamespace(is_available=lambda: True)
    monkeypatch.setattr(distributed.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(distributed.torch, "musa", musa, raising=False)
    monkeypatch.setattr(
        distributed.importlib,
        "import_module",
        lambda name: imported.append(name) or object(),
    )

    name, api, backend = distributed.accelerator_backend()

    assert imported == ["torch_musa"]
    assert (name, api, backend) == ("musa", musa, "mccl")


def test_profile_summary_is_embedded_in_measurement(monkeypatch):
    sample = measure.torch.zeros(1, 4, 2, 2)
    profile = {"summary": "bounded profiler table", "artifacts": {}}
    execution_order = []
    monkeypatch.setattr(measure.catalog, "build_vae", lambda *args: object())
    monkeypatch.setattr(measure.catalog, "sample_for", lambda *args: sample)
    monkeypatch.setattr(
        measure.catalog, "describe_vae", lambda *args: {"adapter": "Adapter"}
    )
    monkeypatch.setattr(measure.catalog, "run_half", lambda *args: sample)
    monkeypatch.setattr(measure, "configure_sharding", lambda *args: "Adapter")
    monkeypatch.setattr(measure, "configure_tiling", lambda *args: {"enabled": False})
    monkeypatch.setattr(
        measure.profile,
        "profile_once",
        lambda *args: execution_order.append("profile") or profile,
    )
    monkeypatch.setattr(measure, "across_ranks", lambda *args: {})
    monkeypatch.setattr(
        measure,
        "timed",
        lambda *args: execution_order.append("timed") or {"median_s": 0.0},
    )
    monkeypatch.setattr(measure.torch.cuda, "synchronize", lambda *args: None)
    monkeypatch.setattr(
        measure.torch.cuda, "reset_peak_memory_stats", lambda *args: None
    )
    monkeypatch.setattr(measure.torch.cuda, "max_memory_allocated", lambda *args: 0)

    class Log:
        enabled = False
        by_call = {}

        def reset(self):
            pass

        def report(self):
            return {}

    args = SimpleNamespace(
        family="kl",
        half="decoder",
        dtype="float32",
        batch=1,
        skip_reference=True,
        reference_max_latent_elems=0,
        phase_timing=False,
        profile=True,
        profile_trace=False,
        profile_memory=False,
        warmup=0,
        iters=1,
        max_rel=None,
    )
    cell = {
        "name": "single",
        "height": 16,
        "width": 16,
        "frames": 1,
        "sharding": "unsharded",
    }
    runtime = SimpleNamespace(
        device=SimpleNamespace(type="cuda"),
        device_api=measure.torch.cuda,
        rank=0,
        world_size=1,
        group=object(),
        log=Log(),
    )

    _, measurement = measure.measure_cell(
        args,
        {"spatial": 8, "temporal": None},
        cell,
        runtime,
        {},
        lambda *args: None,
    )

    assert measurement["profile"] == profile
    assert execution_order == ["timed", "profile"]


def test_tile_shape_costs_measure_latency_memory_and_batch_scaling(monkeypatch):
    vae = object()
    calls = []
    monkeypatch.setattr(shape_costs.catalog, "build_vae", lambda *args: vae)
    monkeypatch.setattr(shape_costs.vae_api, "tile_shape", lambda value: (64, 64))
    monkeypatch.setattr(
        shape_costs.catalog,
        "run_half",
        lambda value, half, sample: calls.append(tuple(sample.shape)) or sample,
    )

    monkeypatch.setattr(shape_costs.dist, "barrier", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        shape_costs.dist,
        "all_gather_object",
        lambda values, value, **kwargs: values.__setitem__(0, value),
    )
    monkeypatch.setattr(shape_costs.torch.cuda, "synchronize", lambda *args: None)
    monkeypatch.setattr(
        shape_costs.torch.cuda, "reset_peak_memory_stats", lambda *args: None
    )
    monkeypatch.setattr(
        shape_costs.torch.cuda,
        "max_memory_allocated",
        lambda *args: 10 * 1024 * 1024,
    )
    monkeypatch.setattr(shape_costs.torch.cuda, "empty_cache", lambda: None)
    args = SimpleNamespace(
        family="kl",
        dtype="float32",
        frames=1,
        iters=1,
        tile_shape_batch=2,
        tile_shape_sides="8,4",
        warmup=0,
    )
    spec = {"latent_channels": 16, "spatial": 8, "temporal": None}

    result = shape_costs.tile_shape_costs(
        args,
        spec,
        SimpleNamespace(
            device="cpu",
            device_api=shape_costs.torch.cuda,
            group=object(),
            rank=0,
            world_size=1,
        ),
        lambda *parts: None,
    )

    assert [entry["tiles_in_the_call"] for entry in result["shapes"]] == [1, 2, 1, 2]
    assert all(entry["median_ms"] >= 0 for entry in result["shapes"])
    assert all(entry["peak_vram_mb"] == 10.0 for entry in result["shapes"])
    assert result["analysis"]["highest_area_cost"]["rows"] in {4, 8}
    assert result["analysis"]["worst_batch_scaling"]["tiles_in_the_call"] == 2
    assert result["latent_window"] == (8, 8)
    assert result["frames"] is None
    assert calls == [(1, 16, 8, 8), (2, 16, 8, 8), (1, 16, 4, 4), (2, 16, 4, 4)]


def test_explicit_shape_sides_accept_an_asymmetric_native_window(monkeypatch):
    monkeypatch.setattr(shape_costs.catalog, "build_vae", lambda *args: object())
    monkeypatch.setattr(shape_costs.vae_api, "tile_shape", lambda value: (64, 32))
    monkeypatch.setattr(
        shape_costs.dist,
        "all_gather_object",
        lambda values, value, **kwargs: values.__setitem__(0, value),
    )
    monkeypatch.setattr(shape_costs.torch, "randn", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        shape_costs, "_shape_iterations", lambda *args: ([0.001], None)
    )
    device_api = SimpleNamespace(
        reset_peak_memory_stats=lambda *args: None,
        max_memory_allocated=lambda *args: 0,
        empty_cache=lambda: None,
    )
    args = SimpleNamespace(
        family="kl",
        dtype="float32",
        frames=1,
        iters=1,
        tile_shape_batch=1,
        tile_shape_sides="8",
        warmup=0,
    )

    result = shape_costs.tile_shape_costs(
        args,
        {"latent_channels": 16, "spatial": 8, "temporal": None},
        SimpleNamespace(
            device="cpu",
            device_api=device_api,
            group=object(),
            rank=0,
            world_size=1,
        ),
        lambda *parts: None,
    )

    assert result["latent_window"] == (8, 4)
    assert [(entry["rows"], entry["columns"]) for entry in result["shapes"]] == [
        (8, 8)
    ]


@pytest.mark.parametrize("failure_phase", ["allocation", "decode"])
def test_tile_shape_oom_is_synchronized_before_the_next_case(
    monkeypatch, failure_phase
):
    vae = object()
    monkeypatch.setattr(shape_costs.catalog, "build_vae", lambda *args: vae)
    monkeypatch.setattr(shape_costs.vae_api, "tile_shape", lambda value: (64, 64))
    monkeypatch.setattr(shape_costs.dist, "barrier", lambda *args, **kwargs: None)
    gathered = []

    def gather(values, value, **kwargs):
        gathered.append(value)
        values[:] = [value, None]

    monkeypatch.setattr(shape_costs.dist, "all_gather_object", gather)
    monkeypatch.setattr(shape_costs.torch.cuda, "synchronize", lambda *args: None)
    monkeypatch.setattr(
        shape_costs.torch.cuda, "reset_peak_memory_stats", lambda *args: None
    )
    monkeypatch.setattr(shape_costs.torch.cuda, "empty_cache", lambda: None)
    if failure_phase == "allocation":
        monkeypatch.setattr(
            shape_costs.torch,
            "randn",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                shape_costs.torch.OutOfMemoryError("allocation")
            ),
        )
        monkeypatch.setattr(
            shape_costs.catalog,
            "run_half",
            lambda *args: pytest.fail("decode ran after allocation failed"),
        )
    else:
        monkeypatch.setattr(
            shape_costs.catalog,
            "run_half",
            lambda *args: (_ for _ in ()).throw(
                shape_costs.torch.OutOfMemoryError("decode")
            ),
        )
    args = SimpleNamespace(
        family="kl",
        dtype="float32",
        frames=17,
        iters=1,
        tile_shape_batch=1,
        tile_shape_sides="8",
        warmup=1,
    )

    result = shape_costs.tile_shape_costs(
        args,
        {"latent_channels": 16, "spatial": 8, "temporal": None},
        SimpleNamespace(
            device="cpu",
            device_api=shape_costs.torch.cuda,
            group=object(),
            rank=0,
            world_size=2,
        ),
        lambda *parts: None,
    )

    assert result["shapes"][0]["out_of_memory"] is True
    assert result["shapes"][0]["failed_ranks"] == [0]
    assert any(
        failure and failure["type"] == "OutOfMemoryError" for failure in gathered
    )


@pytest.mark.parametrize("failure_phase", ["allocation", "decode"])
def test_tile_shape_mixed_rank_failure_is_not_treated_as_oom(
    monkeypatch, failure_phase
):
    monkeypatch.setattr(shape_costs.catalog, "build_vae", lambda *args: object())
    monkeypatch.setattr(shape_costs.vae_api, "tile_shape", lambda value: (64, 64))
    monkeypatch.setattr(shape_costs.dist, "barrier", lambda *args, **kwargs: None)
    peer_failure = {"type": "RuntimeError", "message": "fatal peer", "rank": 1}

    def gather(values, value, **kwargs):
        values[:] = [value, peer_failure if value is not None else None]

    monkeypatch.setattr(shape_costs.dist, "all_gather_object", gather)
    device_api = SimpleNamespace(
        synchronize=lambda *args: None,
        reset_peak_memory_stats=lambda *args: None,
        empty_cache=lambda: None,
    )
    if failure_phase == "allocation":
        monkeypatch.setattr(
            shape_costs.torch,
            "randn",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                shape_costs.torch.OutOfMemoryError("local oom")
            ),
        )
    else:
        monkeypatch.setattr(shape_costs.torch, "randn", lambda *args, **kwargs: object())
        monkeypatch.setattr(
            shape_costs.catalog,
            "run_half",
            lambda *args: (_ for _ in ()).throw(
                shape_costs.torch.OutOfMemoryError("local oom")
            ),
        )
    args = SimpleNamespace(
        family="kl",
        dtype="float32",
        frames=1,
        iters=1,
        tile_shape_batch=1,
        tile_shape_sides="8",
        warmup=1,
    )

    with pytest.raises(distributed.RankError) as caught:
        shape_costs.tile_shape_costs(
            args,
            {"latent_channels": 16, "spatial": 8, "temporal": None},
            SimpleNamespace(
                device="cpu",
                device_api=device_api,
                group=object(),
                rank=0,
                world_size=2,
            ),
            lambda *parts: None,
        )

    assert caught.value.rank_error["failed_ranks"] == [0, 1]
    assert [failure["type"] for failure in caught.value.rank_error["failures"]] == [
        "OutOfMemoryError",
        "RuntimeError",
    ]


def test_tile_shape_setup_failure_is_synchronized_before_cases(monkeypatch):
    monkeypatch.setattr(shape_costs.catalog, "build_vae", lambda *args: object())
    monkeypatch.setattr(shape_costs.vae_api, "tile_shape", lambda value: (64, 64))
    peer_failure = {"type": "RuntimeError", "message": "setup failed", "rank": 1}
    gathered = []

    def gather(values, value, **kwargs):
        gathered.append(value)
        values[:] = [value, peer_failure]

    monkeypatch.setattr(shape_costs.dist, "all_gather_object", gather)
    monkeypatch.setattr(
        shape_costs.torch,
        "randn",
        lambda *args, **kwargs: pytest.fail("case allocation began before setup vote"),
    )
    args = SimpleNamespace(
        family="kl",
        dtype="float32",
        frames=1,
        iters=1,
        tile_shape_batch=1,
        tile_shape_sides="8",
        warmup=0,
    )

    with pytest.raises(RuntimeError, match="setup failed on rank 1"):
        shape_costs.tile_shape_costs(
            args,
            {"latent_channels": 16, "spatial": 8, "temporal": None},
            SimpleNamespace(
                device="cpu",
                device_api=shape_costs.torch.cuda,
                group=object(),
                rank=0,
                world_size=2,
            ),
            lambda *parts: None,
        )

    assert gathered == [None]


@pytest.mark.parametrize("is_shape_costs", [False, True])
def test_measurement_records_effective_dtype_and_world_size(
    monkeypatch, is_shape_costs
):
    args = SimpleNamespace(
        family="kl",
        half="decoder",
        dtype="float16",
        frames=1,
        tile_shape_costs=is_shape_costs,
    )
    runtime = SimpleNamespace(
        rank=0, world_size=3, group=object(), device_api=measure.torch.cuda
    )
    cell = {
        "name": "single",
        "height": 512,
        "width": 512,
        "frames": 1,
    }
    monkeypatch.setattr(report, "render", lambda *args: None)
    monkeypatch.setattr(cli.dist, "all_gather_object", lambda *args, **kwargs: None)
    if is_shape_costs:
        monkeypatch.setattr(
            shape_costs,
            "tile_shape_costs",
            lambda *args: {
                "latent_window": 64,
                "frames": None,
                "analysis": {},
                "shapes": [],
            },
        )
    else:
        monkeypatch.setattr(
            measure,
            "measure_cell",
            lambda *args: ({"sharding": "row"}, {"timing": {}}),
        )

    [record] = cli._measure(args, [cell], runtime)

    assert record["runtime"] == {"dtype": "float16", "world_size": 3}
    if is_shape_costs:
        assert record["shape"]["frames"] is None
        assert record["measurement"]["tile_shape_costs"]["frames"] is None


@pytest.mark.parametrize("is_shape_costs", [False, True])
def test_distributed_cell_errors_preserve_per_rank_details(
    monkeypatch, capsys, is_shape_costs
):
    args = SimpleNamespace(
        family="kl",
        half="decoder",
        dtype="float32",
        frames=1,
        tile_shape_costs=is_shape_costs,
    )
    runtime = SimpleNamespace(
        rank=0,
        world_size=2,
        group=object(),
        device_api=SimpleNamespace(empty_cache=lambda: None),
    )
    cell = {
        "name": "failing-cell",
        "height": 512,
        "width": 512,
        "frames": 1,
    }
    peer_error = {"type": "ValueError", "message": "peer failure", "rank": 1}

    def gather(values, value, **kwargs):
        values[:] = [value, peer_error]

    monkeypatch.setattr(cli.dist, "all_gather_object", gather)
    if is_shape_costs:
        monkeypatch.setattr(
            shape_costs,
            "tile_shape_costs",
            lambda *args: (_ for _ in ()).throw(RuntimeError("local failure")),
        )
    else:
        monkeypatch.setattr(
            measure,
            "measure_cell",
            lambda *args: (_ for _ in ()).throw(RuntimeError("local failure")),
        )

    [record] = cli._measure(args, [cell], runtime)

    assert record["error"]["rank"] == 0
    assert record["error"]["failed_ranks"] == [0, 1]
    assert record["error"]["failures"] == [
        {"type": "RuntimeError", "message": "local failure", "rank": 0},
        peer_error,
    ]
    assert report.report_status([record]) == 1
    assert "RuntimeError: local failure" in capsys.readouterr().out


def test_tiled_numerical_disagreement_keeps_raw_verdict_without_enforcement():
    agreement = measure.agreement_with(
        measure.torch.tensor([2.0]),
        measure.torch.tensor([1.0]),
        "float32",
        max_rel=0.1,
    )

    assert agreement["disagreement_type"] == "numerical"
    assert agreement["ok"] is False
    assert "enforced" not in agreement
    report.set_agreement_policy(agreement, tiling_enabled=True)
    assert agreement["enforced"] is False
    assert report.report_status([{"measurement": {"agreement": agreement}}]) == 0


def test_tiled_shape_mismatch_is_enforced():
    agreement = measure.agreement_with(
        measure.torch.zeros(1, 2),
        measure.torch.zeros(1, 3),
        "float32",
        max_rel=None,
    )

    assert agreement["disagreement_type"] == "shape"
    assert agreement["ok"] is False
    assert "enforced" not in agreement
    report.set_agreement_policy(agreement, tiling_enabled=True)
    assert agreement["enforced"] is True
    assert report.report_status([{"measurement": {"agreement": agreement}}]) == 1


def test_enforced_agreement_and_execution_errors_fail():
    mismatch = {"measurement": {"agreement": {"ok": False, "enforced": True}}}
    assert report.report_status([mismatch]) == 1
    assert report.report_status([{}, mismatch]) == 1
    assert (
        report.report_status([{"error": {"type": "RuntimeError", "message": "boom"}}])
        == 1
    )
    assert report.report_status([{}, {"error": {"type": "RuntimeError"}}]) == 1


def test_square_tile_shape_and_stride_are_applied_through_distvae_plans(monkeypatch):
    class Vae:
        tile_sample_min_size = 512
        overlap = (128, 128)

        def enable_tiling(self):
            pass

    vae = Vae()
    calls = []

    monkeypatch.setattr(measure.vae_api, "require_vae_support", lambda *args: None)
    monkeypatch.setattr(
        measure.vae_api,
        "tile_shape",
        lambda value: (value.tile_sample_min_size,) * 2,
    )
    monkeypatch.setattr(measure.vae_api, "tile_overlap", lambda value: value.overlap)
    monkeypatch.setattr(measure, "latent_rows", lambda value, plan=None: 32)
    monkeypatch.setattr(
        measure.vae_api,
        "tile_shape_plan",
        lambda value, height, width: (
            calls.append(("tile_shape_plan", (height, width)))
            or {"tile_sample_min_size": height}
        ),
    )
    monkeypatch.setattr(
        measure.vae_api,
        "tile_overlap_plan",
        lambda value, height, width, sample_shape=None: (
            calls.append(("tile_overlap_plan", (height, width), sample_shape))
            or {"tile_sample_stride_height": 192}
        ),
    )
    monkeypatch.setattr(
        measure.vae_api,
        "tiled_decode_for",
        lambda value: calls.append(("tiled_decode_for",)) or (lambda sample: sample),
    )

    def apply(value, plan):
        calls.append(("apply_tile_plan", dict(plan)))
        for name, setting in plan.items():
            setattr(value, name, setting)
        if "tile_sample_min_size" in plan:
            value.overlap = (64, 64)
        if "tile_sample_stride_height" in plan:
            value.overlap = (64, 32)

    monkeypatch.setattr(measure.vae_api, "apply_tile_plan", apply)
    cell = {
        "sharding": "unsharded",
        "tiling": 256,
        "height": 2048,
        "width": 2048,
        "overlap": (64, 32),
        "tile_distribution": None,
    }

    facts = measure.configure_tiling(
        vae,
        cell,
        SimpleNamespace(world_size=2, group=object()),
        "decoder",
        lambda *parts: None,
    )

    assert calls == [
        ("tile_shape_plan", (256, 256)),
        ("apply_tile_plan", {"tile_sample_min_size": 256}),
        ("tile_overlap_plan", (64, 32), (2048, 2048)),
        ("apply_tile_plan", {"tile_sample_stride_height": 192}),
        ("tiled_decode_for",),
    ]
    assert facts["native_window_px"] == (512, 512)
    assert facts["window_px"] == (256, 256)
    assert facts["native_overlap_px"] == (128, 128)
    assert facts["overlap"] == (64, 32)
    assert "default_overlap" not in facts
    assert "narrowest_useful_window_px" not in facts
    assert "below_useful_floor" not in facts


def test_an_invalid_exact_tile_shape_is_not_silently_snapped(monkeypatch):
    class Vae:
        def enable_tiling(self):
            pass

    monkeypatch.setattr(measure.vae_api, "require_vae_support", lambda *args: None)
    monkeypatch.setattr(measure.vae_api, "tile_shape", lambda value: (512, 512))
    monkeypatch.setattr(measure.vae_api, "tile_overlap", lambda value: (128, 128))
    monkeypatch.setattr(measure.vae_api, "tile_shape_plan", lambda *args: None)

    with pytest.raises(ValueError, match=r"tile shape \(255, 255\) is invalid for Vae"):
        measure.configure_tiling(
            Vae(),
            {
                "sharding": "unsharded",
                "tiling": 255,
                "overlap": None,
                "tile_distribution": None,
            },
            SimpleNamespace(world_size=1, group=object()),
            "decoder",
            lambda *parts: None,
        )


def test_native_tile_window_enables_tiling_without_replanning(monkeypatch):
    class Vae:
        def __init__(self):
            self.enabled = False

        def enable_tiling(self):
            self.enabled = True

    vae = Vae()
    monkeypatch.setattr(measure.vae_api, "require_vae_support", lambda *args: None)
    monkeypatch.setattr(measure.vae_api, "tile_shape", lambda value: (512, 512))
    monkeypatch.setattr(measure.vae_api, "tile_overlap", lambda value: (128, 128))
    monkeypatch.setattr(measure, "latent_rows", lambda value: 64)
    monkeypatch.setattr(
        measure.vae_api,
        "tile_shape_plan",
        lambda *args: pytest.fail("native tiling must not create a replacement plan"),
    )

    facts = measure.configure_tiling(
        vae,
        {
            "sharding": "unsharded",
            "tiling": "native",
            "overlap": None,
            "tile_distribution": None,
        },
        SimpleNamespace(world_size=1, group=object()),
        "decoder",
        lambda *parts: None,
    )

    assert vae.enabled is True
    assert facts["requested_window"] == "native"
    assert facts["window_px"] == (512, 512)


def test_custom_overlap_installs_the_per_axis_replacement(monkeypatch):
    class Vae:
        def enable_tiling(self):
            pass

    vae = Vae()
    applied = []
    monkeypatch.setattr(measure.vae_api, "require_vae_support", lambda *args: None)
    monkeypatch.setattr(measure.vae_api, "tile_shape", lambda value: (512, 512))
    monkeypatch.setattr(measure.vae_api, "tile_overlap", lambda value: (64, 32))
    monkeypatch.setattr(measure, "latent_rows", lambda value: 64)
    monkeypatch.setattr(
        measure.vae_api,
        "tile_overlap_plan",
        lambda value, height, width, sample_shape=None: {
            "overlap": (height, width),
            "sample_shape": sample_shape,
        },
    )
    monkeypatch.setattr(
        measure.vae_api,
        "apply_tile_plan",
        lambda value, plan: applied.append(plan),
    )
    replacement = object()
    monkeypatch.setattr(
        measure.vae_api, "tiled_decode_for", lambda value: replacement
    )

    facts = measure.configure_tiling(
        vae,
        {
            "sharding": "unsharded",
            "tiling": "native",
            "height": 2048,
            "width": 2048,
            "overlap": (64, 32),
            "tile_distribution": None,
        },
        SimpleNamespace(world_size=1, group=object()),
        "decoder",
        lambda *parts: None,
    )

    assert applied == [
        {"overlap": (64, 32), "sample_shape": (2048, 2048)}
    ]
    assert vae.tiled_decode is replacement


def test_report_schema_contains_provenance_and_effective_composition():
    record = report.make_record(
        family="kl",
        half="decoder",
        shape={"height": 512, "width": 512, "frames": 1},
        composition={
            "sharding": "row",
            "tiling": "native",
            "overlap": (64, 32),
            "tile_distribution": None,
        },
        measurement={"timing": {"median_s": 1.0}},
        dtype="float32",
        world_size=4,
    )

    assert report.SCHEMA_VERSION == 5
    assert record["schema_version"] == 5
    assert set(record["versions"]) >= {"torch", "diffusers", "distvae"}
    assert "distvae_git_revision" in record["provenance"]
    assert record["composition"]["sharding"] == "row"
    assert record["measurement"]["timing"]["median_s"] == 1.0
    assert record["runtime"] == {"dtype": "float32", "world_size": 4}
    assert any(
        path.endswith("bench/harness/measure.py")
        for path in record["provenance"]["benchmark"]["implementation"]
    )


def test_measured_record_with_description_renders_metrics(capsys):
    record = {
        "composition": {"execution": "measurement"},
        "measurement": {
            "description": {"adapter": "DecoderAdapter"},
            "collectives": {
                "by_call": {},
                "by_call_max": {},
            },
            "timing": {"median_s": 0.125},
            "peak_vram_mb": 64,
            "agreement": None,
        },
    }

    report.render(record, "decoder")

    output = capsys.readouterr().out
    assert "median 125.0 ms" in output
    assert '"adapter"' not in output
