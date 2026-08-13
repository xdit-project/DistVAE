"""Benchmark execution, timing, memory, phase timing, and output agreement."""

import time
from collections import Counter

import torch
import torch.distributed as dist
import torch.nn as nn

from distvae import vae as vae_api
from distvae.vae.tiling import _latent_shape, latent_rows

from . import catalog, profile
from .distributed import across_ranks
from .report import set_agreement_policy

MAX_REL = {"float32": 1e-4, "float16": 2e-2, "bfloat16": 5e-2}


def _device_api(runtime):
    return runtime.device_api


def _tile_latent_area(vae):
    shape = _latent_shape(vae)
    return shape[0] * shape[1] if shape is not None else None


def configure_tiling(vae, cell, runtime, half, say):
    """Apply the requested tile window, overlap, and whole-tile distribution."""
    if cell["window"] is None:
        return {"enabled": False}
    if half != "decoder":
        raise ValueError("tiling is a decode-side feature and requires --half decoder")

    vae_api.require_vae_support(vae, "tiling", "--case")
    vae.enable_tiling()
    native = vae_api.tile_shape(vae)
    native_window = tuple(native) if native is not None else None
    native_overlap = vae_api.tile_overlap(vae)
    facts = {
        "enabled": True,
        "requested_window_px": tuple(cell["window"]),
        "native_window_px": native_window,
        "window_px": tuple(cell["window"]),
        "native_overlap_px": native_overlap,
    }

    requested = tuple(cell["window"])
    plan = vae_api.tile_shape_plan(vae, *requested)
    if plan is None:
        raise ValueError(
            f"tile shape {requested} is invalid for {type(vae).__name__}"
        )
    rows = latent_rows(vae, plan)
    if cell["sharding"] == "row" and rows is not None and rows < runtime.world_size:
        raise ValueError(
            f"a {requested[0]}x{requested[1]}px tile has {rows} latent rows "
            f"for {runtime.world_size} row shards"
        )
    vae_api.apply_tile_plan(vae, plan)
    facts["tile_latent_rows"] = rows

    overlap = cell.get("overlap")
    if overlap is not None:
        plan = vae_api.tile_overlap_plan(
            vae,
            *overlap,
            sample_shape=(cell["height"], cell["width"]),
        )
        if plan is None:
            raise ValueError(
                f"tile overlap {overlap} is unavailable for {type(vae).__name__}"
            )
        vae_api.apply_tile_plan(vae, plan)
    tiled_decode = vae_api.tiled_decode_for(vae)
    if tiled_decode is not None:
        vae.tiled_decode = tiled_decode
    facts.update(
        overlap=vae_api.tile_overlap(vae),
        tile_latent_area=_tile_latent_area(vae),
    )

    if cell["tile_distribution"] is not None:
        if not vae_api.supports_tile_parallel(vae):
            raise ValueError(
                f"{type(vae).__name__} does not support whole-tile distribution"
            )
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


def agreement_with(actual, reference, dtype, max_rel):
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
    counters.clear()
    _device_api(runtime).reset_peak_memory_stats(runtime.device)
    timing = timed(once, args.iters, runtime)
    peak_mb = _device_api(runtime).max_memory_allocated(runtime.device) / (1024 * 1024)
    phases = phase_report(counters, runtime)
    profile_result = profile.profile_once(once, args, cell, runtime)
    reference = references.get(key)
    agreement = (
        agreement_with(
            output,
            reference,
            args.dtype,
            args.max_rel,
        )
        if reference is not None
        else None
    )
    if agreement is not None:
        set_agreement_policy(agreement, tiling["enabled"])
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
        "profile": profile_result,
        "peak_vram_mb": peak_mb,
        "agreement": agreement,
    }
    return composition, measurement
