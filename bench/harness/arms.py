"""Orthogonal sharding, tile-window, overlap, and distribution configurations."""

from itertools import product

PRESETS = {
    "unsharded": {"sharding": "unsharded", "tiling": None},
    "row": {"sharding": "row", "tiling": None},
    "row-tiled": {"sharding": "row", "tiling": "native"},
    "row-tiled-half": {"sharding": "row", "tiling": "half"},
    "row-tiled-quarter": {"sharding": "row", "tiling": "quarter"},
    "tiled": {"sharding": "unsharded", "tiling": "native"},
    "tile-runs": {
        "sharding": "unsharded",
        "tiling": "native",
        "tile_distribution": "runs",
    },
    "tile-runs-half": {
        "sharding": "unsharded",
        "tiling": "half",
        "tile_distribution": "runs",
    },
    "tile-runs-quarter": {
        "sharding": "unsharded",
        "tiling": "quarter",
        "tile_distribution": "runs",
    },
}

LEGACY_ARM_NAMES = {
    "none": "unsharded",
    "pvae": "row",
    "tile": "row-tiled",
    "tile-half": "row-tiled-half",
    "tile-quarter": "row-tiled-quarter",
    "tile-nopvae": "tiled",
    "tile-dist": "tile-runs",
    "tile-dist-half": "tile-runs-half",
    "tile-dist-quarter": "tile-runs-quarter",
}

# Public compatibility table retained for callers that enumerate legacy arms.
ARM_ALIASES = {name: PRESETS[preset] for name, preset in LEGACY_ARM_NAMES.items()}


def normalize_legacy_args(args):
    """Normalize compatibility preset names once at the CLI boundary."""
    if args.grid_arms:
        args.grid_arms = ",".join(
            LEGACY_ARM_NAMES.get(name.strip(), name.strip())
            for name in args.grid_arms.split(",")
        )


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


def parse_overlap(value):
    """Parse an explicit HEIGHTxWIDTH output-pixel overlap."""
    value = value.strip().lower()
    parts = value.split("x")
    if len(parts) != 2:
        raise ValueError(
            f"tile overlap must be an absolute HEIGHTxWIDTH pixel pair, not {value!r}"
        )
    try:
        overlap = tuple(int(part) for part in parts)
    except ValueError:
        raise ValueError(
            f"tile overlap must be an absolute HEIGHTxWIDTH pixel pair, not {value!r}"
        ) from None
    if any(axis < 0 for axis in overlap):
        raise ValueError("tile overlap pixels must be non-negative")
    return overlap


def _overlap_label(overlap):
    return f"{overlap[0]}x{overlap[1]}"


def _overlaps(text):
    return (
        [None]
        if not text
        else [None, *(parse_overlap(value) for value in text.split(","))]
    )


def _arm(name):
    name = LEGACY_ARM_NAMES.get(name, name)
    if name not in PRESETS:
        choices = sorted({*PRESETS, *LEGACY_ARM_NAMES})
        raise ValueError(f"unknown arm {name!r}; choose from {choices}")
    return PRESETS[name]


def validate_cell(cell):
    """Validate one canonical ordinary benchmark cell."""
    if cell["sharding"] not in ("unsharded", "row"):
        raise ValueError(f"unknown sharding mode {cell['sharding']!r}")
    if cell["height"] <= 0 or cell["width"] <= 0 or cell["frames"] <= 0:
        raise ValueError("height, width, and frames must be positive")
    if cell["tile_distribution"] is not None and cell["tiling"] is None:
        raise ValueError("tile distribution requires a tile window")
    if cell["tile_distribution"] is not None and cell["sharding"] == "row":
        raise ValueError(
            "row sharding and whole-tile distribution are alternative execution modes"
        )
    if cell["overlap"] is not None and cell["tiling"] is None:
        raise ValueError("tile overlap requires a tile window")
    return cell


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
                validate_cell(
                    {
                        "name": (
                            name
                            if overlap is None
                            else f"{name}-ov{_overlap_label(overlap)}"
                        ),
                        **arm,
                        **shape,
                        "overlap": overlap,
                        "tile_distribution": arm.get("tile_distribution"),
                    }
                )
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
        ambiguous = [
            flag
            for flag, present in (
                ("--sharding", args.sharding is not None),
                ("--no-parallel-vae", args.no_parallel_vae),
                ("--enable-tiling", args.enable_tiling),
                ("--tile-window", args.tile_window is not None),
                ("--vae-tile-size", args.vae_tile_size is not None),
                ("--tile-distribution", args.tile_distribution is not None),
                ("--tile-split", args.tile_split is not None),
            )
            if present
        ]
        if ambiguous:
            raise ValueError(
                "--grid-arms cannot be combined with explicit composition axes: "
                + ", ".join(ambiguous)
            )
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
    overlap = None
    if args.tile_overlap:
        overlap_values = args.tile_overlap.split(",")
        if len(overlap_values) > 1:
            raise ValueError(
                "multiple tile overlap pairs require --grid-arms; "
                "non-grid runs accept exactly one HEIGHTxWIDTH pair"
            )
        overlap = parse_overlap(overlap_values[0])
    if overlap is not None and tiling is None:
        raise ValueError("tile overlap requires a tile window")
    return [
        validate_cell(
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
        )
    ]
