import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from bench.harness import arms, catalog, cli, measure, report


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
        cli.torch.cuda,
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
        overlaps="0,0.25",
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
    assert {cell["overlap"] for cell in cells if cell["tiling"]} == {None, 0.0, 0.25}


def test_parser_exposes_independent_composition_axes():
    args = cli.parser().parse_args(
        [
            "--sharding",
            "unsharded",
            "--tile-window",
            "256",
            "--tile-overlap",
            "0.25",
            "--tile-distribution",
            "runs",
        ]
    )

    [cell] = arms.cells_from_args(args)

    assert cell["sharding"] == "unsharded"
    assert cell["tiling"] == 256
    assert cell["overlap"] == 0.25
    assert cell["tile_distribution"] == "runs"


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


def test_tile_shape_costs_measure_latency_memory_and_batch_scaling(monkeypatch):
    vae = object()
    calls = []
    monkeypatch.setattr(measure.catalog, "build_vae", lambda *args: vae)
    monkeypatch.setattr(measure.vae_api, "tile_window", lambda value: 64)
    monkeypatch.setattr(
        measure.catalog,
        "run_half",
        lambda value, half, sample: calls.append(tuple(sample.shape)) or sample,
    )

    monkeypatch.setattr(measure.dist, "barrier", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        measure.dist,
        "all_gather_object",
        lambda values, value, **kwargs: values.__setitem__(0, value),
    )
    monkeypatch.setattr(measure.torch.cuda, "synchronize", lambda *args: None)
    monkeypatch.setattr(
        measure.torch.cuda, "reset_peak_memory_stats", lambda *args: None
    )
    monkeypatch.setattr(
        measure.torch.cuda, "max_memory_allocated", lambda *args: 10 * 1024 * 1024
    )
    monkeypatch.setattr(measure.torch.cuda, "empty_cache", lambda: None)
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

    result = measure.tile_shape_costs(
        args,
        spec,
        SimpleNamespace(device="cpu", group=object(), rank=0, world_size=1),
        lambda *parts: None,
    )

    assert [entry["tiles_in_the_call"] for entry in result["shapes"]] == [1, 2, 1, 2]
    assert all(entry["median_ms"] >= 0 for entry in result["shapes"])
    assert all(entry["peak_vram_mb"] == 10.0 for entry in result["shapes"])
    assert result["analysis"]["highest_area_cost"]["rows"] in {4, 8}
    assert result["analysis"]["worst_batch_scaling"]["tiles_in_the_call"] == 2
    assert result["frames"] is None
    assert calls == [(1, 16, 8, 8), (2, 16, 8, 8), (1, 16, 4, 4), (2, 16, 4, 4)]


@pytest.mark.parametrize("failure_phase", ["allocation", "decode"])
def test_tile_shape_oom_is_synchronized_before_the_next_case(
    monkeypatch, failure_phase
):
    vae = object()
    monkeypatch.setattr(measure.catalog, "build_vae", lambda *args: vae)
    monkeypatch.setattr(measure.vae_api, "tile_window", lambda value: 64)
    monkeypatch.setattr(measure.dist, "barrier", lambda *args, **kwargs: None)
    gathered = []

    def gather(values, value, **kwargs):
        gathered.append(value)
        values[:] = [value, None]

    monkeypatch.setattr(measure.dist, "all_gather_object", gather)
    monkeypatch.setattr(measure.torch.cuda, "synchronize", lambda *args: None)
    monkeypatch.setattr(
        measure.torch.cuda, "reset_peak_memory_stats", lambda *args: None
    )
    monkeypatch.setattr(measure.torch.cuda, "empty_cache", lambda: None)
    if failure_phase == "allocation":
        monkeypatch.setattr(
            measure.torch,
            "randn",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                measure.torch.OutOfMemoryError("allocation")
            ),
        )
        monkeypatch.setattr(
            measure.catalog,
            "run_half",
            lambda *args: pytest.fail("decode ran after allocation failed"),
        )
    else:
        monkeypatch.setattr(
            measure.catalog,
            "run_half",
            lambda *args: (_ for _ in ()).throw(
                measure.torch.OutOfMemoryError("decode")
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

    result = measure.tile_shape_costs(
        args,
        {"latent_channels": 16, "spatial": 8, "temporal": None},
        SimpleNamespace(device="cpu", group=object(), rank=0, world_size=2),
        lambda *parts: None,
    )

    assert result["shapes"][0]["out_of_memory"] is True
    assert result["shapes"][0]["failed_ranks"] == [0]
    assert any(
        failure and failure["type"] == "OutOfMemoryError" for failure in gathered
    )


def test_tile_shape_setup_failure_is_synchronized_before_cases(monkeypatch):
    monkeypatch.setattr(measure.catalog, "build_vae", lambda *args: object())
    monkeypatch.setattr(measure.vae_api, "tile_window", lambda value: 64)
    peer_failure = {"type": "RuntimeError", "message": "setup failed", "rank": 1}
    gathered = []

    def gather(values, value, **kwargs):
        gathered.append(value)
        values[:] = [value, peer_failure]

    monkeypatch.setattr(measure.dist, "all_gather_object", gather)
    monkeypatch.setattr(
        measure.torch,
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
        measure.tile_shape_costs(
            args,
            {"latent_channels": 16, "spatial": 8, "temporal": None},
            SimpleNamespace(device="cpu", group=object(), rank=0, world_size=2),
            lambda *parts: None,
        )

    assert gathered == [None]


@pytest.mark.parametrize("shape_costs", [False, True])
def test_measurement_records_effective_dtype_and_world_size(monkeypatch, shape_costs):
    args = SimpleNamespace(
        family="kl",
        half="decoder",
        dtype="float16",
        frames=1,
        tile_shape_costs=shape_costs,
    )
    runtime = SimpleNamespace(rank=0, world_size=3, group=object())
    cell = {
        "name": "single",
        "height": 512,
        "width": 512,
        "frames": 1,
    }
    monkeypatch.setattr(report, "render", lambda *args: None)
    monkeypatch.setattr(cli.dist, "all_gather_object", lambda *args, **kwargs: None)
    if shape_costs:
        monkeypatch.setattr(
            measure,
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
    if shape_costs:
        assert record["shape"]["frames"] is None
        assert record["measurement"]["tile_shape_costs"]["frames"] is None


def test_tiled_agreement_keeps_raw_verdict_without_enforcement():
    agreement = {"ok": False, "max_rel_to_scale": 0.2}

    report.set_agreement_policy(agreement, tiling_enabled=True)

    assert agreement["ok"] is False
    assert agreement["enforced"] is False
    assert report.report_status([{"measurement": {"agreement": agreement}}]) == 0


def test_enforced_agreement_and_execution_errors_fail():
    mismatch = {"measurement": {"agreement": {"ok": False, "enforced": True}}}
    assert report.report_status([mismatch]) == 1
    assert report.report_status([{}, mismatch]) == 1
    assert (
        report.report_status([{"error": {"type": "RuntimeError", "message": "boom"}}])
        == 1
    )
    assert report.report_status([{}, {"error": {"type": "RuntimeError"}}]) == 1


def test_tile_window_and_stride_are_applied_through_distvae_plans(monkeypatch):
    class Vae:
        tile_sample_min_size = 512

        def enable_tiling(self):
            pass

    vae = Vae()
    calls = []

    monkeypatch.setattr(measure.vae_api, "require_vae_support", lambda *args: None)
    monkeypatch.setattr(
        measure.vae_api, "tile_window", lambda value: value.tile_sample_min_size
    )
    monkeypatch.setattr(measure.vae_api, "narrowest_useful_window", lambda value: 128)
    monkeypatch.setattr(measure.vae_api, "tile_overlap", lambda value: (0.25, 0.25))
    monkeypatch.setattr(measure.vae_api, "latent_rows", lambda value, plan=None: 32)
    monkeypatch.setattr(
        measure.vae_api,
        "tile_plan",
        lambda value, pixels: calls.append(("tile_plan", pixels))
        or {"tile_sample_min_size": pixels},
    )
    monkeypatch.setattr(
        measure.vae_api,
        "tile_overlap_plan",
        lambda value, overlap: calls.append(("tile_overlap_plan", overlap))
        or {"tile_sample_stride_height": 192},
    )

    def apply(value, plan):
        calls.append(("apply_tile_plan", dict(plan)))
        for name, setting in plan.items():
            setattr(value, name, setting)

    monkeypatch.setattr(measure.vae_api, "apply_tile_plan", apply)
    cell = {
        "sharding": "unsharded",
        "tiling": 256,
        "overlap": 0.25,
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
        ("tile_plan", 256),
        ("apply_tile_plan", {"tile_sample_min_size": 256}),
        ("tile_overlap_plan", 0.25),
        ("apply_tile_plan", {"tile_sample_stride_height": 192}),
    ]
    assert facts["window_px"] == 256
    assert facts["overlap"] == (0.25, 0.25)


def test_report_schema_contains_provenance_and_effective_composition():
    record = report.make_record(
        family="kl",
        half="decoder",
        shape={"height": 512, "width": 512, "frames": 1},
        composition={
            "sharding": "row",
            "tiling": "native",
            "overlap": 0.25,
            "tile_distribution": None,
        },
        measurement={"timing": {"median_s": 1.0}},
        dtype="float32",
        world_size=4,
    )

    assert record["schema_version"] == report.SCHEMA_VERSION
    assert set(record["versions"]) >= {"torch", "diffusers", "distvae"}
    assert "distvae_git_revision" in record["provenance"]
    assert record["composition"]["sharding"] == "row"
    assert record["measurement"]["timing"]["median_s"] == 1.0
    assert record["runtime"] == {"dtype": "float32", "world_size": 4}


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
