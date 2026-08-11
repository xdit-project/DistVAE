"""Bounded benchmark cases and deterministic rectangular tile-plan selection."""

import math

from distvae import vae as vae_api
from distvae.vae.tile_parallel import shares
from distvae.vae.tiling import latent_rows


PROFILES = ("throughput", "balanced", "memory")
MODES = ("unsharded", "row", "local", "tile-runs", "row-tiled")


def parse_pair(value, label):
    """Parse an exact HEIGHTxWIDTH integer pair."""
    parts = value.lower().split("x")
    if len(parts) != 2:
        raise ValueError(f"{label} must be HEIGHTxWIDTH, not {value!r}")
    try:
        pair = tuple(int(part) for part in parts)
    except ValueError:
        raise ValueError(f"{label} must be HEIGHTxWIDTH, not {value!r}") from None
    if any(axis < 0 for axis in pair):
        raise ValueError(f"{label} axes must be non-negative")
    return pair


def _cell(name, mode, height, width, frames, window=None, overlap=None, **facts):
    sharding = "row" if mode in ("row", "row-tiled") else "unsharded"
    distribution = "runs" if mode == "tile-runs" else None
    return {
        "name": name,
        "mode": mode,
        "sharding": sharding,
        "window": window,
        "overlap": overlap,
        "tile_distribution": distribution,
        "height": height,
        "width": width,
        "frames": frames,
        **facts,
    }


def parse_case(value, height, width, frames):
    """Parse MODE or tiled MODE:HEIGHTxWIDTH@HEIGHTxWIDTH."""
    if value in ("unsharded", "row"):
        return _cell(value, value, height, width, frames)
    try:
        mode, plan = value.split(":", 1)
        window_text, overlap_text = plan.split("@", 1)
    except ValueError:
        raise ValueError(
            "case must be unsharded, row, or "
            "MODE:WINDOW_HEIGHTxWINDOW_WIDTH@OVERLAP_HEIGHTxOVERLAP_WIDTH"
        ) from None
    if mode not in ("local", "tile-runs", "row-tiled"):
        raise ValueError(f"unknown tiled case mode {mode!r}")
    window = parse_pair(window_text, "tile window")
    overlap = parse_pair(overlap_text, "tile overlap")
    if any(axis <= 0 for axis in window):
        raise ValueError("tile window axes must be positive")
    if any(overlap_axis >= window_axis for overlap_axis, window_axis in zip(overlap, window)):
        raise ValueError("tile overlap must be smaller than its window")
    label = f"{mode}-{window[0]}x{window[1]}-ov{overlap[0]}x{overlap[1]}"
    return _cell(label, mode, height, width, frames, window, overlap)


def cells_from_args(args):
    """Return exact requested cases; an empty list requests the default suite."""
    if args.case and args.shape:
        raise ValueError("--shape cannot be combined with exact --case values")
    return [
        parse_case(value, args.height, args.width, args.frames)
        for value in (args.case or ())
    ]


def shapes_from_args(args):
    """Return explicitly requested sample shapes or the single global shape."""
    if not args.shape:
        return [(args.height, args.width, args.frames)]
    shapes = []
    for value in args.shape:
        parts = value.lower().split("x")
        if len(parts) not in (2, 3):
            raise ValueError(f"--shape must be HxW or HxWxFRAMES, not {value!r}")
        try:
            height, width = (int(axis) for axis in parts[:2])
            frames = int(parts[2]) if len(parts) == 3 else args.frames
        except ValueError:
            raise ValueError(
                f"--shape must be HxW or HxWxFRAMES, not {value!r}"
            ) from None
        if min(height, width, frames) <= 0:
            raise ValueError("--shape axes and frames must be positive")
        shapes.append((height, width, frames))
    return shapes


def _axis_window(length, overlap, count):
    if count == 1:
        return length
    return math.ceil(length / count) + overlap


def topology_objectives(window, overlap, sample_shape, world_size):
    """Price actual clipped tile areas and deterministic scheduler imbalance."""
    axis_sizes = []
    for length, size, blend in zip(sample_shape, window, overlap):
        stride = size - blend
        axis_sizes.append(
            [min(size, length - start) for start in range(0, length, stride)]
        )
    weights = [
        height * width for height in axis_sizes[0] for width in axis_sizes[1]
    ]
    tile_count = len(weights)
    owners = shares(weights, world_size)
    loads = [
        sum(weight for weight, owner in zip(weights, owners) if owner == rank)
        for rank in range(world_size)
    ]
    average = sum(loads) / world_size
    return {
        "window_area": window[0] * window[1],
        "decoded_area": sum(weights),
        "tile_count": tile_count,
        "max_rank_area": max(loads),
        "rank_imbalance": max(loads) / average - 1,
        "tile_grid": tuple(len(sizes) for sizes in axis_sizes),
    }


def _dominates(left, right):
    keys = ("window_area", "decoded_area", "rank_imbalance")
    return all(left[key] <= right[key] for key in keys) and any(
        left[key] < right[key] for key in keys
    )


def pareto_frontier(candidates):
    """Return candidates not dominated on memory, work, and rank imbalance."""
    return [
        candidate
        for candidate in candidates
        if not any(
            other is not candidate
            and _dominates(other["objectives"], candidate["objectives"])
            for other in candidates
        )
    ]


def _balanced_key(candidate, frontier):
    objectives = candidate["objectives"]
    keys = ("window_area", "decoded_area", "rank_imbalance")
    distances = []
    for key in keys:
        values = [entry["objectives"][key] for entry in frontier]
        low, high = min(values), max(values)
        distances.append(0.0 if high == low else (objectives[key] - low) / (high - low))
    return max(distances), sum(distances), candidate["window"]


def select_plans(sample_shape, native_overlap, world_size, normalize):
    """Select throughput, knee, and memory representatives from a bounded frontier."""
    if world_size < 1:
        raise ValueError("world size must be positive")
    max_tiles = max(4, 4 * world_size)
    min_tiles = max(2, world_size)
    candidates = {}
    for down in range(1, max_tiles + 1):
        for across in range(1, max_tiles + 1):
            requested_tiles = down * across
            if not min_tiles <= requested_tiles <= max_tiles:
                continue
            overlap = (
                0 if down == 1 else native_overlap[0],
                0 if across == 1 else native_overlap[1],
            )
            window = (
                _axis_window(sample_shape[0], overlap[0], down),
                _axis_window(sample_shape[1], overlap[1], across),
            )
            normalized = normalize(window, overlap)
            if normalized is None:
                continue
            window, overlap = normalized
            if any(blend >= size for blend, size in zip(overlap, window)):
                continue
            objectives = topology_objectives(
                window, overlap, sample_shape, world_size
            )
            if not min_tiles <= objectives["tile_count"] <= max_tiles:
                continue
            candidates[(window, overlap)] = {
                "window": tuple(window),
                "overlap": tuple(overlap),
                "objectives": objectives,
            }
    frontier = pareto_frontier(list(candidates.values()))
    if len(frontier) < 3:
        raise ValueError(
            f"sample {sample_shape} produces only {len(frontier)} useful tile plans"
        )
    throughput = min(
        frontier,
        key=lambda item: (
            item["objectives"]["decoded_area"],
            item["objectives"]["rank_imbalance"],
            -item["objectives"]["window_area"],
            item["window"],
        ),
    )
    memory = min(
        (item for item in frontier if item is not throughput),
        key=lambda item: (
            item["objectives"]["window_area"],
            item["objectives"]["decoded_area"],
            item["objectives"]["rank_imbalance"],
            item["window"],
        ),
    )
    balanced = min(
        (item for item in frontier if item not in (throughput, memory)),
        key=lambda item: _balanced_key(item, frontier),
    )
    selected = (throughput, balanced, memory)
    return [
        {
            **plan,
            "profile": profile,
            "selection": {
                "pareto_optimal": True,
                "frontier_size": len(frontier),
                "candidate_limit": max_tiles,
            },
        }
        for profile, plan in zip(PROFILES, selected)
    ]


def normalizer_for_vae(vae, sample_shape, world_size):
    """Return a candidate normalizer backed by DistVAE's exact planners."""
    native = vae_api.tile_shape(vae)
    if native is None:
        raise ValueError(f"{type(vae).__name__} has no tile window")

    def normalize(window, overlap):
        height_options = [
            value
            for value in range(window[0], window[0] + native[0] + 1)
            if vae_api.tile_shape_plan(vae, value, native[1]) is not None
        ][:8]
        width_options = [
            value
            for value in range(window[1], window[1] + native[1] + 1)
            if vae_api.tile_shape_plan(vae, native[0], value) is not None
        ][:8]
        for height in height_options:
            for width in width_options:
                shape_plan = vae_api.tile_shape_plan(vae, height, width)
                if shape_plan is None:
                    continue
                rows = latent_rows(vae, shape_plan)
                if rows is not None and rows < world_size:
                    continue
                original = {}
                missing = []
                for name, planned in shape_plan.items():
                    if hasattr(vae, name):
                        original[name] = getattr(vae, name)
                    else:
                        missing.append(name)
                    setattr(vae, name, planned)
                try:
                    overlap_plan = vae_api.tile_overlap_plan(
                        vae, *overlap, sample_shape=sample_shape
                    )
                finally:
                    for name in missing:
                        delattr(vae, name)
                    for name, value in original.items():
                        setattr(vae, name, value)
                if overlap_plan is not None:
                    return (height, width), overlap
        return None

    return normalize


def plans_for_vae(vae, height, width, world_size):
    """Select the default plans for a concrete VAE."""
    overlap = vae_api.tile_overlap(vae)
    if overlap is None:
        raise ValueError(f"{type(vae).__name__} has no tile overlap")
    sample_shape = (height, width)
    return select_plans(
        sample_shape,
        overlap,
        world_size,
        normalizer_for_vae(vae, sample_shape, world_size),
    )


def default_suite(plans, height, width, frames):
    """Build the bounded nine-case suite from three selected tile plans."""
    suite = baseline_suite(height, width, frames)
    for mode in ("local", "tile-runs"):
        for plan in plans:
            window, overlap, profile = (
                plan["window"],
                plan["overlap"],
                plan["profile"],
            )
            suite.append(
                _cell(
                    f"{mode}-{profile}",
                    mode,
                    height,
                    width,
                    frames,
                    window,
                    overlap,
                    profile=profile,
                    plan_selection=plan,
                )
            )
    memory = next(plan for plan in plans if plan["profile"] == "memory")
    suite.append(
        _cell(
            "row-tiled-memory",
            "row-tiled",
            height,
            width,
            frames,
            memory["window"],
            memory["overlap"],
            profile="memory",
            plan_selection=memory,
        )
    )
    return suite


def baseline_suite(height, width, frames):
    """Build the two untiled cases supported by encoders and decoders."""
    return [
        _cell("unsharded", "unsharded", height, width, frames),
        _cell("row", "row", height, width, frames),
    ]
