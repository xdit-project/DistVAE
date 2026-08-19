"""Decoder cost measurements across tile shapes and batch sizes."""

import time

import torch
import torch.distributed as dist

from distvae import vae as vae_api

from . import cases, catalog
from .distributed import (
    RankError,
    aggregate_rank_errors,
    exception_record,
    gather_rank_errors,
)


def _device_api(runtime):
    return runtime.device_api


def _timing_report(samples):
    samples.sort()
    return {
        "median_s": samples[len(samples) // 2],
        "mean_s": sum(samples) / len(samples),
        "min_s": samples[0],
        "max_s": samples[-1],
        "samples_s": samples,
    }


def _synchronize_failure(local_error, runtime):
    """Share a rank-local case failure before any rank enters the next case."""
    failures = gather_rank_errors(local_error, runtime)
    return aggregate_rank_errors(failures)


def _all_out_of_memory(failure):
    """Return whether every underlying rank failure is an OOM."""
    failures = failure.get("failures", [failure])
    return all(item["type"] == "OutOfMemoryError" for item in failures)


def _shape_iterations(run, iterations, runtime):
    """Run decoder iterations with a rank verdict exchange after each call."""
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
            local_error = exception_record(error, runtime.rank)
        failure = _synchronize_failure(local_error, runtime)
        if failure is not None:
            return None, failure
        samples.append(elapsed)
    return samples, None


def tile_shape_costs(args, spec, runtime, say):
    """Measure decoder cost across representative tile shapes and batch sizes."""
    local_error = None
    try:
        vae = catalog.build_vae(args.family, args.dtype, runtime.device)
        window = vae_api.tile_shape(vae)
        if window is None:
            raise ValueError(
                f"{type(vae).__name__} has no native tile shape for shape analysis"
            )
        latent_window = tuple(value // spec["spatial"] for value in window)
        depth = 1 + (args.frames - 1) // spec["temporal"] if spec["temporal"] else None

        if args.tile_shape_windows:
            shapes = [
                cases.parse_pair(value, "latent tile window")
                for value in args.tile_shape_windows.split(",")
            ]
            if any(min(shape) <= 0 for shape in shapes):
                raise ValueError("--tile-shape-windows axes must be positive")
        else:
            plans = cases.plans_for_vae(
                vae, args.height, args.width, runtime.world_size
            )
            shapes = [
                tuple(axis // spec["spatial"] for axis in plan["window"])
                for plan in plans
            ]
        if not shapes:
            raise ValueError(
                f"tile window produces no representative shapes at {latent_window}"
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
        local_error = exception_record(error, runtime.rank)
    failure = _synchronize_failure(local_error, runtime)
    if failure is not None:
        raise RankError(failure, "tile shape setup")

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
                local_error = exception_record(error, runtime.rank)
            failure = _synchronize_failure(local_error, runtime)
            if failure is not None:
                if not _all_out_of_memory(failure):
                    raise RankError(failure, "tile shape setup")
                measured.append(
                    {
                        "rows": rows,
                        "columns": columns,
                        "tiles_in_the_call": count,
                        "latent_area": rows * columns,
                        "out_of_memory": True,
                        "failure_phase": "allocation",
                        "failed_ranks": failure["failed_ranks"],
                    }
                )
                say(f"{rows}x{columns} x{count}: out of memory during allocation")
                latent = None
                _device_api(runtime).empty_cache()
                break

            def once():
                with torch.no_grad():
                    return catalog.run_half(vae, "decoder", latent)

            _, failure = _shape_iterations(once, args.warmup, runtime)
            failure_phase = "warmup"
            if failure is None:
                samples, failure = _shape_iterations(once, args.iters, runtime)
                failure_phase = "measurement"
            if failure is not None:
                if not _all_out_of_memory(failure):
                    raise RankError(failure, f"tile shape {failure_phase}")
                measured.append(
                    {
                        "rows": rows,
                        "columns": columns,
                        "tiles_in_the_call": count,
                        "latent_area": rows * columns,
                        "out_of_memory": True,
                        "failure_phase": failure_phase,
                        "failed_ranks": failure["failed_ranks"],
                    }
                )
                say(f"{rows}x{columns} x{count}: out of memory during {failure_phase}")
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
    return {
        "family": args.family,
        "latent_window": latent_window,
        "frames": effective_frames,
        "shapes": measured,
        "analysis": {
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
        },
    }
