"""Command-line parsing and orchestration for the DistVAE benchmark."""

import argparse

import torch.distributed as dist

from . import arms, catalog, measure, report, shape_costs
from .distributed import (
    Runtime,
    aggregate_rank_errors,
    exception_record,
    gather_rank_errors,
)


def parser():
    """Build the compatibility command-line parser."""
    value = argparse.ArgumentParser(
        description=(
            "Measure native DistVAE VAE sharding, tiling, and tile distribution."
        )
    )
    value.add_argument("--family", default="flux2", choices=sorted(catalog.FAMILIES))
    value.add_argument("--half", default="decoder", choices=["decoder", "encoder"])
    value.add_argument("--height", type=int, default=2048)
    value.add_argument("--width", type=int, default=2048)
    value.add_argument("--frames", type=int, default=17)
    value.add_argument("--dtype", default="bfloat16", choices=sorted(measure.MAX_REL))
    value.add_argument("--warmup", type=int, default=2)
    value.add_argument("--iters", type=int, default=5)
    value.add_argument("--batch", type=int, default=1)
    value.add_argument(
        "--sharding",
        choices=["unsharded", "row"],
        help="decoder/encoder execution: intact or DistVAE row sharding",
    )
    value.add_argument(
        "--no-parallel-vae",
        "--no_parallel_vae",
        action="store_true",
        help="leave each VAE call unsharded",
    )
    value.add_argument(
        "--enable-tiling",
        "--enable_tiling",
        action="store_true",
        help="tile at the VAE's native window",
    )
    value.add_argument(
        "--tile-window",
        type=arms.parse_tile_window,
        help="native, half, quarter, or a positive pixel window; enables tiling",
    )
    value.add_argument(
        "--vae-tile-size",
        "--vae_tile_size",
        help="custom pixel window, or half/quarter; implies tiling",
    )
    value.add_argument(
        "--tile-overlap",
        help="absolute HEIGHTxWIDTH pixel overlap; comma-separated pairs for grids",
    )
    value.add_argument(
        "--tile-distribution",
        choices=["runs", "scattered"],
        help="distribute whole-tile runs or individual tile calls across ranks",
    )
    value.add_argument("--grid-arms", help="comma-separated compatibility arm names")
    value.add_argument(
        "--grid-shapes",
        help="comma-separated HxW or HxWxFRAMES measurement shapes",
    )
    value.add_argument(
        "--tile-split",
        choices=["tiles", "scattered", "rows"],
        help="compatibility spelling for tile distribution",
    )
    value.add_argument(
        "--phase-timing",
        action="store_true",
        help="measure decoder calls separately from tiled decode overhead",
    )
    value.add_argument(
        "--profile",
        action="store_true",
        help="profile one selected VAE-half call outside the timed iterations",
    )
    value.add_argument(
        "--profile-trace",
        action="store_true",
        help="export a harness-named Chrome trace; implies --profile",
    )
    value.add_argument(
        "--profile-memory",
        action="store_true",
        help="export a harness-named memory timeline; implies --profile",
    )
    value.add_argument(
        "--profile-dir",
        default="bench-profile",
        help="directory for requested profiler artifacts",
    )
    value.add_argument(
        "--tile-shape-costs",
        action="store_true",
        help="measure decoder latency and memory across tile shapes and batch sizes",
    )
    value.add_argument(
        "--tile-shape-batch",
        type=int,
        default=1,
        help="largest power-of-two tile batch to measure",
    )
    value.add_argument(
        "--tile-shape-sides",
        default="",
        help="comma-separated square latent tile sides to measure",
    )
    value.add_argument("--max-rel", type=float)
    value.add_argument("--skip-reference", action="store_true")
    value.add_argument("--reference-max-latent-elems", type=int, default=16384)
    value.add_argument(
        "--describe-only",
        action="store_true",
        help="describe native adapter selection on the meta device and stop",
    )
    value.add_argument("--timeout-min", type=int, default=30)
    value.add_argument("--out", help="write versioned JSON here")
    return value


def _shape(spec, cell):
    return {
        "height": cell["height"],
        "width": cell["width"],
        "frames": cell["frames"] if spec["temporal"] else None,
    }


def _describe(args, cells, provenance_data=None):
    spec = catalog.FAMILIES[args.family]
    records = []
    for cell in cells:
        vae = catalog.build_vae(args.family, args.dtype, "meta")
        description = catalog.describe_vae(vae, args.half)
        composition = {
            **cell,
            "execution": "describe-only",
            "adapter": description["adapter"],
        }
        records.append(
            report.make_record(
                args.family,
                args.half,
                _shape(spec, cell),
                composition,
                {"description": description},
                dtype=args.dtype,
                world_size=1,
                provenance_data=provenance_data,
            )
        )
    return records


def _measure(args, cells, runtime, provenance_data=None):
    spec = catalog.FAMILIES[args.family]
    if args.tile_shape_costs:
        error = None
        costs = {"frames": args.frames if spec["temporal"] else None}
        try:
            costs = shape_costs.tile_shape_costs(
                args,
                spec,
                runtime,
                lambda *parts: print(*parts, flush=True) if runtime.rank == 0 else None,
            )
            measurement = {"tile_shape_costs": costs}
        except (Exception, SystemExit) as caught:
            error = exception_record(caught, runtime.rank)
            measurement = {}
        aggregate_error = aggregate_rank_errors(gather_rank_errors(error, runtime))
        composition = {
            "name": "tile-shape-costs",
            "execution": "tile-shape-costs",
            "sharding": "unsharded",
            "tiling": None,
            "overlap": None,
            "tile_distribution": None,
        }
        record = report.make_record(
            args.family,
            "decoder",
            {"height": None, "width": None, "frames": costs.get("frames")},
            composition,
            measurement,
            aggregate_error,
            dtype=args.dtype,
            world_size=runtime.world_size,
            provenance_data=provenance_data,
        )
        if runtime.rank == 0:
            report.render(record, "decoder")
        return [record]

    references = {}
    records = []

    def say(*parts):
        if runtime.rank == 0:
            print(*parts, flush=True)

    for cell in cells:
        error = None
        try:
            composition, measurement = measure.measure_cell(
                args, spec, cell, runtime, references, say
            )
        except (Exception, SystemExit) as caught:
            error = exception_record(caught, runtime.rank)
            print(
                f"[rank {runtime.rank}] cell {cell['name']} failed: "
                f"{error['type']}: {error['message']}",
                flush=True,
            )
            composition, measurement = dict(cell), {}
        runtime.device_api.empty_cache()

        aggregate_error = aggregate_rank_errors(gather_rank_errors(error, runtime))
        record = report.make_record(
            args.family,
            args.half,
            _shape(spec, cell),
            composition,
            measurement,
            aggregate_error,
            dtype=args.dtype,
            world_size=runtime.world_size,
            provenance_data=provenance_data,
        )
        records.append(record)
        if runtime.rank == 0:
            report.render(record, args.half)
    return records


def main(argv=None):
    """Run describe-only or accelerator measurement mode and return an exit status."""
    command = parser()
    args = command.parse_args(argv)
    if args.tile_shape_costs and args.half != "decoder":
        command.error("--tile-shape-costs requires --half decoder")
    if args.tile_shape_costs and args.describe_only:
        command.error("--tile-shape-costs cannot be combined with --describe-only")
    if args.tile_shape_costs:
        cells = []
    else:
        try:
            arms.normalize_legacy_args(args)
            cells = arms.cells_from_args(args)
        except ValueError as error:
            command.error(str(error))
    provenance_data = report.provenance()

    if args.describe_only:
        records = _describe(args, cells, provenance_data)
        for record in records:
            report.render(record, args.half)
        if args.out:
            report.write_json(args.out, records)
        return report.report_status(records)

    runtime = Runtime.start(args.timeout_min)
    try:
        records = _measure(args, cells, runtime, provenance_data)
        if runtime.rank == 0 and args.out:
            report.write_json(args.out, records)
        status = report.report_status(records)
        statuses = [None] * runtime.world_size
        dist.all_gather_object(statuses, status, group=runtime.group)
        return max(statuses)
    finally:
        runtime.close()


if __name__ == "__main__":
    raise SystemExit(main())
