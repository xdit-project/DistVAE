"""Orthogonal sharding, tile-window, overlap, and distribution configurations."""

from itertools import product

ARM_ALIASES = {
    "none": {"sharding": "unsharded", "tiling": None},
    "pvae": {"sharding": "row", "tiling": None},
    "tile": {"sharding": "row", "tiling": "native"},
    "tile-half": {"sharding": "row", "tiling": "half"},
    "tile-quarter": {"sharding": "row", "tiling": "quarter"},
    "tile-nopvae": {"sharding": "unsharded", "tiling": "native"},
    "tile-dist": {
        "sharding": "unsharded",
        "tiling": "native",
        "tile_distribution": "runs",
    },
    "tile-dist-half": {
        "sharding": "unsharded",
        "tiling": "half",
        "tile_distribution": "runs",
    },
    "tile-dist-quarter": {
        "sharding": "unsharded",
        "tiling": "quarter",
        "tile_distribution": "runs",
    },
}


def parse_shapes(text, default_frames):
    """Parse comma-separated HxW and HxWxFRAMES shapes."""
    shapes = []
    for value in text.split(","):
        parts = value.strip().lower().split("x")
        if len(parts) not in (2, 3):
            raise ValueError(f"--grid-shapes takes HxW or HxWxFRAMES, not {value!r}")
        shapes.append(
            {
                "height": int(parts[0]),
                "width": int(parts[1]),
                "frames": int(parts[2]) if len(parts) == 3 else default_frames,
            }
        )
    return shapes


def _overlaps(text):
    return [None] if not text else [None, *(float(value) for value in text.split(","))]


def _arm(name):
    if name not in ARM_ALIASES:
        raise ValueError(f"unknown arm {name!r}; choose from {sorted(ARM_ALIASES)}")
    return ARM_ALIASES[name]


def expand_grid(arm_names, shapes, default_frames, overlaps):
    """Expand arm, shape, and overlap axes into independent cells."""
    names = [name.strip() for name in arm_names.split(",")]
    cells = []
    for shape, name in product(parse_shapes(shapes, default_frames), names):
        arm = _arm(name)
        for overlap in _overlaps(overlaps):
            if overlap is not None and arm["tiling"] is None:
                continue
            cells.append(
                {
                    "name": name if overlap is None else f"{name}-ov{overlap:g}",
                    **arm,
                    **shape,
                    "overlap": overlap,
                    "tile_distribution": arm.get("tile_distribution"),
                }
            )
    return cells


def parse_tile_window(value):
    """Normalize a native, relative, or pixel tile window."""
    if value in (None, "native", "half", "quarter"):
        return value
    pixels = int(value)
    if pixels <= 0:
        raise ValueError("tile window must be a positive pixel count")
    return pixels


def cells_from_args(args):
    """Normalize a single invocation or a requested grid."""
    shapes = args.grid_shapes or f"{args.height}x{args.width}x{args.frames}"
    if args.grid_arms:
        return expand_grid(args.grid_arms, shapes, args.frames, args.tile_overlap)

    tiling = args.tile_window
    if tiling is None and args.enable_tiling:
        tiling = "native"
    if args.vae_tile_size is not None:
        tiling = parse_tile_window(args.vae_tile_size)
    explicit_sharding = args.sharding
    if (
        explicit_sharding is not None
        and args.no_parallel_vae
        and explicit_sharding != "unsharded"
    ):
        raise ValueError("--sharding conflicts with --no-parallel-vae")
    sharding = explicit_sharding
    if sharding is None:
        sharding = "unsharded" if args.no_parallel_vae else "row"
    distribution = args.tile_distribution
    if args.tile_split is not None:
        legacy = {
            "tiles": ("unsharded", "runs"),
            "scattered": ("unsharded", "scattered"),
            "rows": ("row", None),
        }
        legacy_sharding, legacy_distribution = legacy[args.tile_split]
        if explicit_sharding is not None and explicit_sharding != legacy_sharding:
            raise ValueError("--tile-split conflicts with --sharding")
        if args.no_parallel_vae and legacy_sharding != "unsharded":
            raise ValueError("--tile-split conflicts with --no-parallel-vae")
        if (
            args.tile_distribution is not None
            and args.tile_distribution != legacy_distribution
        ):
            raise ValueError("--tile-split conflicts with --tile-distribution")
        sharding, distribution = legacy_sharding, legacy_distribution
    if distribution is not None and tiling is None:
        raise ValueError("tile distribution requires a tile window")
    if distribution is not None and sharding == "row":
        raise ValueError(
            "row sharding and whole-tile distribution are alternative execution modes"
        )
    overlap = float(args.tile_overlap.split(",")[0]) if args.tile_overlap else None
    if overlap is not None and tiling is None:
        raise ValueError("tile overlap requires a tile window")
    return [
        {
            "name": "single",
            "sharding": sharding,
            "tiling": tiling,
            "height": args.height,
            "width": args.width,
            "frames": args.frames,
            "overlap": overlap,
            "tile_distribution": distribution,
        }
    ]
