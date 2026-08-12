"""Bounded benchmark cases and deterministic rectangular tile-plan selection."""

import math

from distvae import vae as vae_api
from distvae.vae.tile_parallel import shares
from distvae.vae.tiling import latent_rows


# Named for tile count, which is a fact about the plan, rather than for an outcome, which is a
# claim about a device - see select_plans.
PROFILES = ("coarse", "balanced", "fine")
MODES = ("unsharded", "row", "local", "tile-runs", "row-tiled")

# The smallest latent extent a tile may have on its narrower axis. Below roughly this, a tile
# normalizes over content too unrepresentative of the image and comes out at a different tone
# from its neighbours. The blend then ramps that difference across the overlap rather than
# stepping at the join, so it reads as banding and no seam metric detects it: the join is smooth,
# the tone is wrong.
#
# In latent units rather than pixels, deliberately, because that is what carries across families
# - a scale-16 VAE reaches the same bound at twice the pixel height a scale-8 one does. A
# fraction of the VAE's native window would NOT carry: FLUX.2's native tile is 128 latent and
# Wan's is 16, so one percentage would mean an eight-fold difference in strictness between them.
# A fraction of the sample would be wrong in a different way, making an identical tile legal at
# one canvas size and illegal at another when the tile's own statistics do not depend on the
# canvas it was cut from.
#
# 16 is where two unrelated families agree. Measured on FLUX.2 at 1024x1024 on four ranks, a
# 96px window is 12 latent and bands visibly while a 128px window is 16 and does not, which
# brackets the threshold at (12, 16]; and Wan's own native tile is exactly 16 latent, so raising
# this bound would reject a vendor default. The bracket has not been narrowed further - 13, 14
# and 15 are untested - so treat 16 as the conservative end of a measurement, not a precise edge.
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
    """Return the requested sample shapes, most explicit request first.

    `--shape` beats `--matrix` beats the single `--height/--width/--frames`, so asking for one
    shape by hand always overrides the family's matrix rather than being appended to it.
    """
    if not args.shape:
        if getattr(args, "matrix", False):
            # Imported here rather than at module scope because catalog builds VAEs and so pulls
            # in diffusers; nothing else in this module needs it, and the planner is exercised
            # without a model.
            from . import catalog

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


def _overlap_options(length, count, native):
    """Overlap candidates for one axis, in output pixels, widest first.

    An inactive axis blends nothing, as before. On an active axis the pitch - the un-overlapped
    share each tile advances by - is the only scale a blend means anything against, so the ladder
    is a fraction of the pitch rather than one fixed pixel count.

    Using the native overlap for every candidate imposes a lower bound on each active window
    axis because `window = pitch + overlap`. With a 256px native overlap, the smallest window for
    a 1024x1024 sample on four ranks is 512x512. Its area equals the 262144 pixels assigned to one
    row-sharded rank, so it cannot reduce memory relative to row sharding. Allowing smaller
    overlaps makes 272x272 windows reachable on the same sample.

    The ladder stops at a quarter of the window, which is a measured bound and not a margin.
    Tile size decides how far a tile's tone drifts from its neighbours'; overlap decides how far
    that drift is ramped out, and so whether the eye reads a gradient or a band. On FLUX.2 at
    1024x1024 on four ranks, a 128px window blended 32px - a quarter - is clean, while the same
    window blended 16px bands. The difference between those decodes is concentrated near tile
    boundaries spaced 112px apart. Since window is pitch + overlap, a quarter of the window is a
    third of the pitch.

    `native` stays in the set so the previous behaviour remains reachable and comparable, but it
    is dropped where it would fall under that quarter.
    """
    if count == 1:
        return (0,)
    pitch = math.ceil(length / count)
    options = {native, pitch // 2, math.ceil(pitch / 3)}
    return tuple(sorted(
        (option for option in options if option > 0 and option * 3 >= pitch),
        reverse=True,
    ))


def topology_objectives(window, overlap, sample_shape, world_size):
    """Compute clipped tile areas and deterministic scheduler imbalance."""
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
    window_area = window[0] * window[1]
    # What a rank holds under plain row sharding, which is the baseline every tiled plan is
    # really competing with - not the unsharded decode. Recording it makes "is this plan a
    # memory win at all?" answerable from the report instead of by hand.
    row_shard_area = math.ceil(sample_shape[0] / world_size) * sample_shape[1]
    return {
        "window_area": window_area,
        "decoded_area": sum(weights),
        "tile_count": tile_count,
        "max_rank_area": max(loads),
        "rank_imbalance": max(loads) / average - 1,
        "tile_grid": tuple(len(sizes) for sizes in axis_sizes),
        "row_shard_area": row_shard_area,
        "beats_row_sharding": window_area < row_shard_area,
        # How many tile columns the grid has, which is the one thing separating a plan from its
        # transpose. Area, work and imbalance are all symmetric under transpose, so without this
        # the model cannot tell a full-WIDTH strip from a full-HEIGHT one - and the hardware very
        # much can. Measured on FLUX.2 at 1024x1024 on four ranks, at identical window area and
        # tile count: 128x1024 costs 966 MB against 1024x128's 1126 MB under tile-runs, and
        # 651 MB against 812 MB under local. A wide tile is a few long contiguous spans and a
        # tall one is a row of short ones, so fewer columns is cheaper at the same area.
        "tile_columns": len(axis_sizes[1]),
    }


def _dominates(left, right):
    keys = ("window_area", "decoded_area", "rank_imbalance", "tile_columns")
    return all(left[key] <= right[key] for key in keys) and any(
        left[key] < right[key] for key in keys
    )


def pareto_frontier(candidates):
    """Return candidates not dominated on memory, work, imbalance, and tile columns."""
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
    """Bracket the tile axis with a coarse, a knee, and a fine plan.

    Coarse has the fewest tiles, fine has the most, and balanced is the Pareto knee between them.
    Coarse minimizes blend boundaries; fine minimizes window area. Return two plans when no
    distinct balanced plan exists.
    """
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
            # Overlap is a search dimension, not a constant. Reducing it shrinks the window
            # without changing the grid - stride stays at the pitch either way - so it is the
            # cheapest axis the planner has, and holding it fixed forfeited the whole region
            # where tiling beats row sharding. See _overlap_options.
            for down_overlap in _overlap_options(
                sample_shape[0], down, native_overlap[0]
            ):
                for across_overlap in _overlap_options(
                    sample_shape[1], across, native_overlap[1]
                ):
                    overlap = (down_overlap, across_overlap)
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
                    # Re-check the quarter-of-window bound against the window actually used.
                    # normalize() may enlarge the requested window. blend_for_window() increases
                    # the overlap when necessary, and this check validates the resulting window
                    # and overlap. A VAE may normalize the requested overlap in turn.
                    # An inactive axis blends nothing and is exempt.
                    if any(
                        0 < blend * 4 < size for blend, size in zip(overlap, window)
                    ):
                        continue
                    objectives = topology_objectives(
                        window, overlap, sample_shape, world_size
                    )
                    if not min_tiles <= objectives["tile_count"] <= max_tiles:
                        continue
                    candidates[(tuple(window), tuple(overlap))] = {
                        "window": tuple(window),
                        "overlap": tuple(overlap),
                        "objectives": objectives,
                    }
    frontier = pareto_frontier(list(candidates.values()))
    if len(frontier) < 2:
        raise ValueError(
            f"sample {sample_shape} produces only {len(frontier)} useful tile plans"
        )
    # Select the largest- and smallest-window Pareto candidates without predicting which is
    # faster. The plan space is primarily ordered by window size, equivalently tile count. Its
    # limits are the minimum window that avoids banding and the maximum useful window.
    # Performance within those limits depends on hardware and must be measured.
    #
    # So the profiles name geometry, not predicted outcome. An earlier pair named throughput and
    # memory scored plans by decoded_area, least total work, which reliably chose the widest
    # window because it overlaps its neighbours fewer times. On gfx1201 those configurations
    # were slower and used more memory than row sharding: 5034 MB versus 3526 MB at 2048x2048 on
    # four ranks. Geometry-based profile names remain accurate across devices.
    coarse = max(
        frontier,
        key=lambda item: (
            item["objectives"]["window_area"],
            -item["objectives"]["tile_columns"],
            item["window"],
        ),
    )
    fine = min(
        (item for item in frontier if item is not coarse),
        key=lambda item: (
            item["objectives"]["window_area"],
            item["objectives"]["decoded_area"],
            item["objectives"]["rank_imbalance"],
            item["objectives"]["tile_columns"],
            item["window"],
        ),
    )
    # The fine candidate must have a smaller window area than the coarse candidate. A transposed
    # window can have identical modeled area, work, and imbalance while using more measured
    # memory; 1024x128 used 17% more than 128x1024 in the measured configuration.
    if fine["objectives"]["window_area"] >= coarse["objectives"]["window_area"]:
        fine = None
    # Distinct by WINDOW, not by identity. Two frontier points can share a window and differ only
    # in blend, and 832x128 blended 36px against the same window blended 34px is not two profiles
    # worth two cases each.
    taken = {plan["window"] for plan in (coarse, fine) if plan is not None}
    remaining = [item for item in frontier if item["window"] not in taken]
    balanced = (
        min(remaining, key=lambda item: _balanced_key(item, frontier))
        if remaining
        else None
    )
    selected = [
        (profile, plan)
        for profile, plan in zip(PROFILES, (coarse, balanced, fine))
        if plan is not None
    ]
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
        for profile, plan in selected
    ]


def blend_for_window(overlap, window):
    """Widen a blend that snapping to a legal window has left under a quarter of it.

    Some VAEs round requested windows upward without changing the requested overlap. Increase
    each active overlap to at least one quarter of the normalized window before select_plans()
    validates the result.

    This adjustment matters most for VAEs with coarse window increments. LTX-2 uses 256px
    increments; without the adjustment, a 1088x1920 sample on eight ranks produces no candidates.
    Increasing the overlap restores three plans at its native 16-latent tile size.

    An inactive axis blends nothing and stays at zero.
    """
    return tuple(
        blend if blend == 0 or blend * 4 >= size else -(-size // 4)
        for blend, size in zip(overlap, window)
    )


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
                # `latent_rows` reports the SMALLER of the tile's two latent extents, so this
                # bounds the narrow axis whichever one it is. A tile needs enough of it both to
                # shard across the ranks and to normalize over something representative; the
                # second is the binding constraint at every world size we run. Without it the
                # widened overlap search reaches genuinely small windows for the first time and
                # the memory profile selects them - it picked 9 latent rows on FLUX.2 at 1024.
                extent = latent_rows(vae, shape_plan)
                if extent is not None and extent < max(world_size, MIN_TILE_LATENT_EXTENT):
                    continue
                blend = blend_for_window(overlap, (height, width))
                if any(size <= value for value, size in zip(blend, (height, width))):
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
                        vae, *blend, sample_shape=sample_shape
                    )
                finally:
                    for name in missing:
                        delattr(vae, name)
                    for name, value in original.items():
                        setattr(vae, name, value)
                if overlap_plan is not None:
                    return (height, width), blend
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
    """Build the bounded suite from the selected tile plans.

    By default only the compositions an orchestrator can actually select: the two untiled
    baselines and whole-tile distribution at each plan. `local` tiles without distributing and
    `row-tiled` shards rows beneath the tiling, and callers reach neither - xFuser, for one,
    branches straight between marking a VAE for tile parallelism and parallelizing its decoder,
    with nothing in between. They are also the slow ones, together about 60% of the suite's
    compute at 1024x1024 on four ranks, which is a poor trade for a number nobody can act on.

    `diagnostics` puts them back. They earn it when characterising a new geometry rather than
    comparing plans: `local` is the only case with no collectives at all, so it separates what
    tiling does to the decode from what the collectives cost, and its peak is the true floor for
    a window - 651 MB against tile-runs' 806 MB on that sample, the difference being assembly
    rather than tile.
    """
    suite = baseline_suite(height, width, frames)
    modes = ("local", "tile-runs") if diagnostics else ("tile-runs",)
    for mode in modes:
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
    if diagnostics:
        # Row sharding beneath the tiling, at the finest plan the sample offers. Lightest by
        # predicted window area, which is a model's opinion rather than a measurement, and one
        # more reason this belongs with the diagnostics.
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
