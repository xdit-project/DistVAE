"""Bounded benchmark cases and deterministic rectangular tile-plan selection."""

import math

from distvae import vae as vae_api
from distvae.vae.tile_parallel import shares
from distvae.vae.tiling import latent_rows

from . import catalog

PROFILES = ("coarse", "balanced", "fine")
MODES = ("unsharded", "row", "local", "tile-runs", "row-tiled")
MIN_TILE_LATENT_EXTENT = 16


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
        if getattr(args, "matrix", False):
            return list(catalog.matrix_for(args.family))
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


def _overlap_options(length, count, native_overlap):
    """Return bounded overlap candidates, widest first."""
    if count == 1:
        return (0,)
    pitch = math.ceil(length / count)
    floor = math.ceil(pitch / 3)
    options = {native_overlap}
    options.update(math.ceil(pitch / divisor) for divisor in (2, 3))
    return tuple(sorted((value for value in options if value >= floor), reverse=True))


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
    row_shard_area = math.ceil(sample_shape[0] / world_size) * sample_shape[1]
    return {
        "window_area": window[0] * window[1],
        "decoded_area": sum(weights),
        "tile_count": tile_count,
        "tile_columns": len(axis_sizes[1]),
        "max_rank_area": max(loads),
        "rank_imbalance": max(loads) / average - 1,
        "tile_grid": tuple(len(sizes) for sizes in axis_sizes),
        "row_shard_area": row_shard_area,
        "beats_row_sharding": window[0] * window[1] < row_shard_area,
    }


def _dominates(left, right):
    keys = ("window_area", "decoded_area", "rank_imbalance", "tile_columns")
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
    keys = ("window_area", "decoded_area", "rank_imbalance", "tile_columns")
    distances = []
    for key in keys:
        values = [entry["objectives"][key] for entry in frontier]
        low, high = min(values), max(values)
        distances.append(0.0 if high == low else (objectives[key] - low) / (high - low))
    return max(distances), sum(distances), candidate["window"]


def select_plans(sample_shape, native_overlap, world_size, normalize):
    """Select coarse, knee, and fine representatives from a bounded frontier."""
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
            down_overlaps = _overlap_options(
                sample_shape[0], down, native_overlap[0]
            )
            across_overlaps = _overlap_options(
                sample_shape[1], across, native_overlap[1]
            )
            for overlap_down in down_overlaps:
                for overlap_across in across_overlaps:
                    overlap = (overlap_down, overlap_across)
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
                    if any(
                        blend and blend * 4 < size
                        for blend, size in zip(overlap, window)
                    ):
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
    if len(frontier) < 2:
        raise ValueError(
            f"sample {sample_shape} produces only {len(frontier)} useful tile plans"
        )
    coarse = min(
        frontier,
        key=lambda item: (
            item["objectives"]["tile_count"],
            item["objectives"]["tile_columns"],
            item["objectives"]["rank_imbalance"],
            item["objectives"]["decoded_area"],
            -item["objectives"]["window_area"],
            item["window"],
        ),
    )
    fine_candidates = [
        item
        for item in frontier
        if item["window"] != coarse["window"]
        and item["objectives"]["window_area"] < coarse["objectives"]["window_area"]
        and item["window"] != tuple(reversed(coarse["window"]))
    ]
    if not fine_candidates:
        fine_candidates = [
            item
            for item in frontier
            if item["window"] != coarse["window"]
            and item["objectives"]["window_area"] < coarse["objectives"]["window_area"]
        ]
    fine = min(
        fine_candidates,
        key=lambda item: (
            item["objectives"]["window_area"],
            -item["objectives"]["tile_count"],
            item["objectives"]["tile_columns"],
            item["objectives"]["decoded_area"],
            item["objectives"]["rank_imbalance"],
            item["window"],
        ),
    )
    middle = [
        item
        for item in frontier
        if item not in (coarse, fine)
        and item["window"] not in (coarse["window"], fine["window"])
    ]
    selected = [coarse]
    if middle:
        selected.append(min(middle, key=lambda item: _balanced_key(item, frontier)))
    selected.append(fine)
    profiles = (
        ("coarse", "fine")
        if len(selected) == 2
        else PROFILES
    )
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
        for profile, plan in zip(profiles, selected)
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
                if rows is not None and rows < max(world_size, MIN_TILE_LATENT_EXTENT):
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


def default_suite(plans, height, width, frames, diagnostics=False):
    """Build the selectable suite, optionally including diagnostic compositions."""
    suite = baseline_suite(height, width, frames)
    modes = ("local", "tile-runs") if diagnostics else ("tile-runs",)
    for mode in modes:
        for plan in plans:
            profile = plan["profile"]
            suite.append(
                _cell(
                    f"{mode}-{profile}",
                    mode,
                    height,
                    width,
                    frames,
                    plan["window"],
                    plan["overlap"],
                    profile=profile,
                    plan_selection=plan,
                )
            )
    if diagnostics:
        lightest = min(plans, key=lambda plan: plan["objectives"]["window_area"])
        suite.append(
            _cell(
                f"row-tiled-{lightest['profile']}",
                "row-tiled",
                height,
                width,
                frames,
                lightest["window"],
                lightest["overlap"],
                profile=lightest["profile"],
                plan_selection=lightest,
            )
        )
    return suite


def baseline_suite(height, width, frames):
    """Build the two untiled cases supported by encoders and decoders."""
    return [
        _cell("unsharded", "unsharded", height, width, frames),
        _cell("row", "row", height, width, frames),
    ]
