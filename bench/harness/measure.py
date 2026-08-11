"""Benchmark execution, timing, memory, phase timing, and output agreement."""

import importlib
import time
from collections import Counter
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn

from distvae import vae as vae_api

from . import catalog
from .distributed import across_ranks
from .report import set_agreement_policy

MAX_REL = {"float32": 1e-4, "float16": 2e-2, "bfloat16": 5e-2}
PROFILE_SUMMARY_LIMIT = 16_000


def _device_api(runtime):
    return runtime.device_api


def _profiler_backend(device_type):
    activity_name = device_type.upper()
    if device_type == "musa":
        try:
            importlib.import_module("torch_musa")
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "MUSA profiling requires the optional torch_musa package"
            ) from error
    activity = getattr(torch.profiler.ProfilerActivity, activity_name, None)
    if device_type != "cpu" and activity is None:
        raise RuntimeError(f"torch.profiler has no {activity_name} activity")
    device_api = getattr(torch, device_type, None)
    memory = getattr(device_api, "memory", None)
    recorder = getattr(memory, "_record_memory_history", None)
    sort_by = f"self_{device_type}_time_total"
    return activity, recorder, sort_by


def profile_once(run, args, cell=None, runtime=None):
    """Profile one VAE-half call and export only explicitly requested artifacts."""
    enabled = args.profile or args.profile_trace or args.profile_memory
    if not enabled:
        return None

    output_dir = Path(args.profile_dir)
    shape = f"{cell['height']}x{cell['width']}x{cell['frames']}"
    stem = (
        f"{args.family}-{args.half}-{cell['name']}-{shape}-rank{runtime.rank}"
    )
    artifacts = {}
    if args.profile_trace:
        artifacts["trace"] = str(output_dir / f"{stem}.trace.json")
    if args.profile_memory:
        artifacts["memory"] = str(output_dir / f"{stem}.memory.html")
    if artifacts:
        output_dir.mkdir(parents=True, exist_ok=True)

    accelerator, memory_recorder, sort_by = _profiler_backend(runtime.device.type)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if accelerator is not None:
        activities.append(accelerator)

    if args.profile_memory and memory_recorder is not None:
        memory_recorder(enabled="all")
    try:
        with torch.profiler.profile(
            activities=activities,
            profile_memory=args.profile_memory,
            record_shapes=args.profile_memory,
            with_stack=args.profile_memory,
        ) as profiler:
            run()
        summary = profiler.key_averages().table(
            sort_by=sort_by, row_limit=20
        )[:PROFILE_SUMMARY_LIMIT]
        if args.profile_trace:
            profiler.export_chrome_trace(artifacts["trace"])
        if args.profile_memory:
            profiler.export_memory_timeline(artifacts["memory"])
    finally:
        if args.profile_memory and memory_recorder is not None:
            memory_recorder(enabled=None)
    return {"summary": summary, "artifacts": artifacts}


def _tile_latent_area(vae):
    sizes = [
        getattr(vae, name, None)
        for name in (
            "tile_latent_min_size",
            "tile_latent_min_height",
            "tile_latent_min_width",
        )
    ]
    sizes = [value for value in sizes if isinstance(value, int) and value > 0]
    if not sizes:
        return None
    return sizes[0] * (sizes[-1] if len(sizes) > 1 else sizes[0])


def configure_tiling(vae, cell, runtime, half, say):
    """Apply the requested tile window, overlap, and whole-tile distribution."""
    if cell["tiling"] is None:
        return {"enabled": False}
    if half != "decoder":
        raise ValueError("tiling is a decode-side feature and requires --half decoder")

    vae_api.require_vae_support(vae, "tiling", "--enable-tiling")
    vae.enable_tiling()
    native = vae_api.tile_window(vae)
    floor = vae_api.narrowest_useful_window(vae)
    facts = {
        "enabled": True,
        "requested_window": cell["tiling"],
        "native_window_px": native,
        "window_px": native,
        "narrowest_useful_window_px": floor,
        "default_overlap": vae_api.tile_overlap(vae),
    }

    requested = cell["tiling"]
    if requested in ("half", "quarter"):
        if native is None:
            raise ValueError(
                f"{requested} needs a single native window for {type(vae).__name__}"
            )
        requested = native // (2 if requested == "half" else 4)
    elif requested != "native":
        requested = int(requested)

    if requested != "native":
        pixels = requested
        plan = vae_api.tile_plan(vae, requested)
        if plan is None:
            pixels, plan = vae_api.snap_tile_window(vae, requested)
        if plan is None:
            raise ValueError(
                f"no workable tile window at or below {requested}px for "
                f"{type(vae).__name__}"
            )
        rows = vae_api.latent_rows(vae, plan)
        if cell["sharding"] == "row" and rows is not None and rows < runtime.world_size:
            raise ValueError(
                f"a {pixels}px tile has {rows} latent rows for "
                f"{runtime.world_size} row shards"
            )
        vae_api.apply_tile_plan(vae, plan)
        facts.update(window_px=pixels, tile_latent_rows=rows)
        if pixels != requested:
            say(f"tile window snapped {requested} -> {pixels}px")
    elif cell["sharding"] == "row":
        rows = vae_api.latent_rows(vae)
        if rows is not None and rows < runtime.world_size:
            raise ValueError(
                f"native tile has {rows} latent rows for "
                f"{runtime.world_size} row shards"
            )
        facts["tile_latent_rows"] = rows

    overlap = cell.get("overlap")
    if overlap is not None:
        if overlap == "half":
            native_overlap = facts["default_overlap"]
            if native_overlap is None:
                raise ValueError(
                    f"half needs a native overlap for {type(vae).__name__}"
                )
            values = (
                native_overlap
                if isinstance(native_overlap, (tuple, list))
                else (native_overlap,)
            )
            overlap = min(values) / 2
            facts["requested_overlap"] = "half"
            facts["native_overlap_min"] = min(values)
            say(
                f"tile overlap half of the VAE's native {min(values):.1%}, "
                f"using {overlap:.1%}"
            )
        plan = vae_api.tile_overlap_plan(vae, overlap)
        if plan is None:
            widest = vae_api.widest_tile_overlap(vae)
            hint = f"; widest supported is {widest}" if widest is not None else ""
            raise ValueError(
                f"tile overlap {overlap} is unavailable for {type(vae).__name__}{hint}"
            )
        vae_api.apply_tile_plan(vae, plan)
    facts.update(
        overlap=vae_api.tile_overlap(vae),
        tile_latent_area=_tile_latent_area(vae),
        below_useful_floor=bool(
            floor is not None
            and facts["window_px"] is not None
            and facts["window_px"] < floor
        ),
    )

    if cell["tile_distribution"] is not None:
        if not vae_api.supports_tile_parallel(vae):
            raise ValueError(
                f"{type(vae).__name__} does not support whole-tile distribution"
            )
        if cell["tile_distribution"] == "scattered":
            dispatch, assemble = vae_api.dispatch_over(runtime.group), None
        else:
            dispatch, assemble = vae_api.sharing(runtime.group)
        tiled_decode = vae_api.tiled_decode_for(vae, dispatch, assemble)
        if tiled_decode is None:
            raise ValueError(f"{type(vae).__name__} has no distributable tiled decode")
        vae.tiled_decode = tiled_decode
        facts["distribution"] = cell["tile_distribution"]
    else:
        facts["distribution"] = None
    return facts


def configure_sharding(vae, cell, runtime, half):
    """Install row sharding or leave the decoder whole."""
    if cell["sharding"] != "row":
        return None
    install = (
        vae_api.parallelize_decoder
        if half == "decoder"
        else vae_api.parallelize_encoder
    )
    return install(vae, runtime.group)


class PhaseTimer(nn.Module):
    """Measure decoder calls separately from the full tiled decode."""

    def __init__(self, decoder, runtime, counters):
        super().__init__()
        self.decoder = decoder
        self.runtime = runtime
        self.counters = counters

    def forward(self, *args, **kwargs):
        _device_api(self.runtime).synchronize(self.runtime.device)
        start = time.perf_counter()
        output = self.decoder(*args, **kwargs)
        _device_api(self.runtime).synchronize(self.runtime.device)
        self.counters["decoder_s"] += time.perf_counter() - start
        self.counters["calls"] += 1
        return output


def install_phase_timing(vae, runtime):
    """Wrap decoder and tiled decode calls for optional phase accounting."""
    counters = Counter()
    vae.decoder = PhaseTimer(vae.decoder, runtime, counters)
    tiled_decode = vae.tiled_decode

    def timed_decode(*args, **kwargs):
        _device_api(runtime).synchronize(runtime.device)
        start = time.perf_counter()
        output = tiled_decode(*args, **kwargs)
        _device_api(runtime).synchronize(runtime.device)
        counters["total_s"] += time.perf_counter() - start
        counters["decodes"] += 1
        return output

    vae.tiled_decode = timed_decode
    return counters


def phase_report(counters, runtime):
    """Summarize phase time per decode and load spread across ranks."""
    decodes = counters.get("decodes", 0)
    if not decodes:
        return None
    total = counters["total_s"] / decodes
    decoder = counters["decoder_s"] / decodes
    result = {
        "total_ms": total * 1e3,
        "decoder_ms": decoder * 1e3,
        "rest_ms": (total - decoder) * 1e3,
        "calls_per_decode": counters["calls"] / decodes,
    }
    if runtime.world_size > 1:
        gathered = [None] * runtime.world_size
        dist.all_gather_object(
            gathered,
            (decoder, counters["calls"] / decodes),
            group=runtime.group,
        )
        result["decoder_ms_by_rank"] = [value * 1e3 for value, _ in gathered]
        result["calls_by_rank"] = [calls for _, calls in gathered]
        slowest = max(value for value, _ in gathered)
        result["idle_share"] = sum(slowest - value for value, _ in gathered) / (
            runtime.world_size * slowest or 1
        )
    return result


def timed(run, iters, runtime):
    """Measure synchronized latency samples."""
    samples = []
    for _ in range(iters):
        dist.barrier(group=runtime.group)
        _device_api(runtime).synchronize(runtime.device)
        start = time.perf_counter()
        run()
        _device_api(runtime).synchronize(runtime.device)
        samples.append(time.perf_counter() - start)
    return _timing_report(samples)


def _timing_report(samples):
    """Summarize a non-empty sequence of latency samples."""
    samples.sort()
    return {
        "median_s": samples[len(samples) // 2],
        "mean_s": sum(samples) / len(samples),
        "min_s": samples[0],
        "max_s": samples[-1],
        "samples_s": samples,
    }


def _error_record(error, rank):
    return {"type": type(error).__name__, "message": str(error), "rank": rank}


def _synchronize_failure(local_error, runtime):
    """Share a rank-local case failure before any rank enters the next case."""
    failures = [None] * runtime.world_size
    dist.all_gather_object(failures, local_error, group=runtime.group)
    failed_ranks = [rank for rank, failure in enumerate(failures) if failure]
    first = next((failure for failure in failures if failure), None)
    return first, failed_ranks


def _shape_iterations(run, iterations, runtime):
    """Run unsharded decoder iterations with a verdict exchange after each call.

    A decoder call here must not contain distributed collectives. A rank that fails
    inside an unmatched collective cannot reach the verdict exchange and cannot be
    recovered by benchmark orchestration.
    """
    samples = []
    for _ in range(iterations):
        dist.barrier(group=runtime.group)
        local_error = None
        elapsed = None
        try:
            _device_api(runtime).synchronize(runtime.device)
            start = time.perf_counter()
            run()
            _device_api(runtime).synchronize(runtime.device)
            elapsed = time.perf_counter() - start
        except Exception as error:
            local_error = _error_record(error, runtime.rank)
        failure, failed_ranks = _synchronize_failure(local_error, runtime)
        if failure is not None:
            return None, failure, failed_ranks
        samples.append(elapsed)
    return samples, None, []


def tile_shape_costs(args, spec, runtime, say):
    """Measure decoder cost across representative tile shapes and batch sizes."""
    local_error = None
    try:
        vae = catalog.build_vae(args.family, args.dtype, runtime.device)
        window = vae_api.tile_window(vae)
        if window is None:
            raise ValueError(
                f"{type(vae).__name__} has no single tile window for shape analysis"
            )
        side = window // spec["spatial"]
        depth = 1 + (args.frames - 1) // spec["temporal"] if spec["temporal"] else None

        if args.tile_shape_sides:
            sides = [int(value) for value in args.tile_shape_sides.split(",")]
            if any(value <= 0 for value in sides):
                raise ValueError("--tile-shape-sides values must be positive")
            shapes = [(value, value) for value in sides]
        else:
            shapes = []
            for down in (1, 2, 4):
                for across in (1, 2, 4):
                    shape = (side // down, side // across)
                    if min(shape) >= 8 and shape not in shapes:
                        shapes.append(shape)
        if not shapes:
            raise ValueError(
                f"tile window produces no representative shapes at {side}px"
            )
        if args.tile_shape_batch < 1:
            raise ValueError("--tile-shape-batch must be positive")

        counts = []
        count = 1
        while count <= args.tile_shape_batch:
            counts.append(count)
            count *= 2
        dtype = getattr(torch, args.dtype)
        effective_frames = args.frames if spec["temporal"] else None
    except (Exception, SystemExit) as error:
        local_error = _error_record(error, runtime.rank)
    failure, _ = _synchronize_failure(local_error, runtime)
    if failure is not None:
        raise RuntimeError(
            f"tile shape setup failed on rank {failure['rank']}: "
            f"{failure['type']}: {failure['message']}"
        )

    measured = []
    baseline = None
    alone = {}
    for rows, columns in shapes:
        for count in counts:
            shape = (count, spec["latent_channels"], rows, columns)
            if depth is not None:
                shape = (count, spec["latent_channels"], depth, rows, columns)
            latent = None
            local_error = None
            try:
                torch.manual_seed(1)
                latent = torch.randn(*shape, dtype=dtype, device=runtime.device)
                _device_api(runtime).reset_peak_memory_stats(runtime.device)
            except Exception as error:
                local_error = _error_record(error, runtime.rank)
            failure, failed_ranks = _synchronize_failure(local_error, runtime)
            if failure is not None:
                if failure["type"] != "OutOfMemoryError":
                    raise RuntimeError(
                        f"tile shape setup failed on rank {failure['rank']}: "
                        f"{failure['type']}: {failure['message']}"
                    )
                measured.append(
                    {
                        "rows": rows,
                        "columns": columns,
                        "tiles_in_the_call": count,
                        "latent_area": rows * columns,
                        "out_of_memory": True,
                        "failure_phase": "allocation",
                        "failed_ranks": failed_ranks,
                    }
                )
                say(f"{rows}x{columns} x{count}: out of memory during allocation")
                latent = None
                _device_api(runtime).empty_cache()
                break

            def once():
                with torch.no_grad():
                    return catalog.run_half(vae, "decoder", latent)

            _, failure, failed_ranks = _shape_iterations(once, args.warmup, runtime)
            failure_phase = "warmup"
            if failure is None:
                samples, failure, failed_ranks = _shape_iterations(
                    once, args.iters, runtime
                )
                failure_phase = "measurement"
            if failure is not None:
                if failure["type"] != "OutOfMemoryError":
                    raise RuntimeError(
                        f"tile shape {failure_phase} failed on rank "
                        f"{failure['rank']}: {failure['type']}: {failure['message']}"
                    )
                measured.append(
                    {
                        "rows": rows,
                        "columns": columns,
                        "tiles_in_the_call": count,
                        "latent_area": rows * columns,
                        "out_of_memory": True,
                        "failure_phase": failure_phase,
                        "failed_ranks": failed_ranks,
                    }
                )
                say(
                    f"{rows}x{columns} x{count}: out of memory during "
                    f"{failure_phase}"
                )
                latent = None
                _device_api(runtime).empty_cache()
                break
            timing = _timing_report(samples)

            area = rows * columns
            median_ms = timing["median_s"] * 1000
            per_tile_ms = median_ms / count
            if count == 1:
                alone[(rows, columns)] = per_tile_ms
                if baseline is None:
                    baseline = (per_tile_ms, area)
            predicted_ms = baseline[0] * area / baseline[1]
            entry = {
                "rows": rows,
                "columns": columns,
                "tiles_in_the_call": count,
                "latent_area": area,
                "timing": timing,
                "median_ms": median_ms,
                "ms_per_tile": per_tile_ms,
                "peak_vram_mb": _device_api(runtime).max_memory_allocated(
                    runtime.device
                )
                / (1024 * 1024),
                "ms_per_1k_latent_area": per_tile_ms / area * 1000,
                "against_area_prediction": per_tile_ms / predicted_ms,
                "against_single_tile": per_tile_ms / alone[(rows, columns)],
            }
            measured.append(entry)
            say(
                f"{rows}x{columns} x{count}: {median_ms:.1f} ms, "
                f"{entry['peak_vram_mb']:.0f} MB"
            )
            latent = None
            _device_api(runtime).empty_cache()

    fitted = [entry for entry in measured if not entry.get("out_of_memory")]
    batched = [entry for entry in fitted if entry["tiles_in_the_call"] > 1]
    analysis = {
        "highest_area_cost": (
            max(fitted, key=lambda entry: entry["against_area_prediction"])
            if fitted
            else None
        ),
        "worst_batch_scaling": (
            max(batched, key=lambda entry: entry["against_single_tile"])
            if batched
            else None
        ),
    }
    return {
        "family": args.family,
        "latent_window": side,
        "frames": effective_frames,
        "shapes": measured,
        "analysis": analysis,
    }


def agreement_with(actual, reference, dtype, max_rel, tiled):
    """Measure raw error against an unsharded, untiled reference."""
    if tuple(actual.shape) != tuple(reference.shape):
        agreement = {
            "ok": False,
            "disagreement_type": "shape",
            "why": f"shape {tuple(actual.shape)} != {tuple(reference.shape)}",
        }
    else:
        diff = (actual.float().cpu() - reference).abs()
        scale = reference.abs().max().item()
        tolerance = max_rel if max_rel is not None else MAX_REL[dtype]
        relative = diff.max().item() / scale if scale else 0.0
        agreement = {
            "ok": bool(relative <= tolerance),
            "disagreement_type": "numerical",
            "max_abs": diff.max().item(),
            "mean_abs": diff.mean().item(),
            "reference_max_abs": scale,
            "max_rel_to_scale": relative,
            "mean_rel_to_scale": diff.mean().item() / scale if scale else 0.0,
            "share_off_by_1pc": (
                (diff > 0.01 * scale).float().mean().item() if scale else 0.0
            ),
            "max_rel_allowed": tolerance,
        }
    set_agreement_policy(agreement, tiled)
    return agreement


def measure_cell(args, spec, cell, runtime, references, say):
    """Build and measure one normalized composition cell."""
    vae = catalog.build_vae(args.family, args.dtype, runtime.device)
    sample = catalog.sample_for(
        spec,
        args.half,
        cell["height"],
        cell["width"],
        args.dtype,
        runtime.device,
        args.batch,
        cell["frames"],
    )
    description = catalog.describe_vae(vae, args.half)
    if cell["sharding"] == "row" and description["adapter"] is None:
        raise ValueError(f"DistVAE has no adapter for {type(vae).__name__} {args.half}")

    latent_area = sample.shape[0] * sample.shape[-2] * sample.shape[-1]
    if sample.ndim == 5:
        latent_area *= sample.shape[2]
    if args.half == "encoder":
        latent_area //= spec["spatial"] ** 2
    key = (cell["height"], cell["width"], cell["frames"], args.half)
    take_reference = (
        not args.skip_reference and latent_area <= args.reference_max_latent_elems
    )
    if take_reference and key not in references:
        with torch.no_grad():
            references[key] = catalog.run_half(vae, args.half, sample).float().cpu()

    adapter = configure_sharding(vae, cell, runtime, args.half)
    tiling = configure_tiling(vae, cell, runtime, args.half, say)
    counters = (
        install_phase_timing(vae, runtime)
        if args.phase_timing and tiling["enabled"]
        else Counter()
    )

    def once():
        with torch.no_grad():
            return catalog.run_half(vae, args.half, sample)

    for _ in range(args.warmup):
        once()
    _device_api(runtime).synchronize(runtime.device)

    runtime.log.reset()
    runtime.log.enabled = True
    try:
        output = once()
    finally:
        runtime.log.enabled = False
    collectives = runtime.log.report()
    collectives.update(
        across_ranks(runtime.log.by_call, runtime.world_size, runtime.group)
    )
    profile = None
    if args.profile or args.profile_trace or args.profile_memory:
        profile = profile_once(once, args, cell, runtime)

    counters.clear()
    _device_api(runtime).reset_peak_memory_stats(runtime.device)
    timing = timed(once, args.iters, runtime)
    peak_mb = _device_api(runtime).max_memory_allocated(runtime.device) / (
        1024 * 1024
    )
    phases = phase_report(counters, runtime)
    reference = references.get(key)
    agreement = (
        agreement_with(
            output,
            reference,
            args.dtype,
            args.max_rel,
            tiling["enabled"],
        )
        if reference is not None
        else None
    )
    composition = {
        **cell,
        "execution": "measurement",
        "adapter": adapter,
        "tiling_effective": tiling,
    }
    measurement = {
        "description": description,
        "latent_shape": list(sample.shape),
        "collectives": collectives,
        "timing": timing,
        "phases": phases,
        "profile": profile,
        "peak_vram_mb": peak_mb,
        "agreement": agreement,
    }
    return composition, measurement
