"""Optional profiler execution and artifact export."""

import importlib
from contextlib import ExitStack
from pathlib import Path

import torch

from .distributed import (
    RankError,
    aggregate_rank_errors,
    exception_record,
    gather_rank_errors,
)

PROFILE_SUMMARY_LIMIT = 16_000


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
    return activity, recorder, f"self_{device_type}_time_total"


def _artifact_stem(output_dir, stem, suffixes):
    candidate = stem
    occurrence = 1
    while any((output_dir / f"{candidate}.{suffix}").exists() for suffix in suffixes):
        occurrence += 1
        candidate = f"{stem}-{occurrence}"
    return candidate


def profile_once(run, args, cell=None, runtime=None):
    """Profile one VAE-half call and export only explicitly requested artifacts."""
    if not (args.profile or args.profile_trace or args.profile_memory):
        return None

    with ExitStack() as memory_setup:
        with ExitStack() as profiler_setup:
            local_error = None
            try:
                output_dir = Path(args.profile_dir)
                shape = f"{cell['height']}x{cell['width']}x{cell['frames']}"
                base = (
                    f"{args.family}-{args.half}-{cell['name']}-{shape}"
                    f"-rank{runtime.rank}"
                )
                suffixes = []
                if args.profile_trace:
                    suffixes.append("trace.json")
                if args.profile_memory:
                    suffixes.append("memory.html")
                stem = _artifact_stem(output_dir, base, suffixes)
                artifacts = {}
                if args.profile_trace:
                    artifacts["trace"] = str(output_dir / f"{stem}.trace.json")
                if args.profile_memory:
                    artifacts["memory"] = str(output_dir / f"{stem}.memory.html")
                if artifacts:
                    output_dir.mkdir(parents=True, exist_ok=True)

                accelerator, memory_recorder, sort_by = _profiler_backend(
                    runtime.device.type
                )
                activities = [torch.profiler.ProfilerActivity.CPU]
                if accelerator is not None:
                    activities.append(accelerator)

                if args.profile_memory and memory_recorder is not None:
                    memory_recorder(enabled="all")
                    memory_setup.callback(memory_recorder, enabled=None)
                profiler = profiler_setup.enter_context(
                    torch.profiler.profile(
                        activities=activities,
                        profile_memory=args.profile_memory,
                        record_shapes=args.profile_memory,
                        with_stack=args.profile_memory,
                    )
                )
            except (Exception, SystemExit) as error:
                local_error = exception_record(error, runtime.rank)
            failure = (
                aggregate_rank_errors(gather_rank_errors(local_error, runtime))
                if getattr(runtime, "world_size", 1) > 1
                else aggregate_rank_errors([local_error])
            )
            if failure is not None:
                raise RankError(failure, "profiler setup")

            run()

        summary = profiler.key_averages().table(sort_by=sort_by, row_limit=20)[
            :PROFILE_SUMMARY_LIMIT
        ]
        if args.profile_trace:
            profiler.export_chrome_trace(artifacts["trace"])
        if args.profile_memory:
            profiler.export_memory_timeline(artifacts["memory"])
        return {"summary": summary, "artifacts": artifacts}
