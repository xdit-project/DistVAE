"""Bench DistVAE's sharded VAE halves at real shapes, without a checkpoint.

What we tune in DistVAE is a property of the adapter stack, not of the weights: PatchGroupNorm
issues the same collectives whether its input came from Flux.2 or from torch.randn. What has to
be real is the shape of the work - channel widths, spatial sizes, layer counts, dtype, device,
rank count - and all of that lives in a VAE's config.json. So this builds the true architecture
with random weights and measures three things per decode:

  collectives  exact counts and bytes, by call site. The point of the harness. An optimisation
               that removes an all_reduce shows up as an integer, not as a timing delta the size
               of the noise on a consumer GPU.
  latency      wall time per decode, after warmup.
  agreement    the sharded output against a single-rank reference, which is the invariant every
               change here has to preserve.

Run under torchrun:
  torchrun --nproc_per_node=4 distvae_bench.py --family flux2 --height 2048 --width 2048

What this cannot tell you: anything about real activation distributions (random weights give
mean~0, variance~1, the easy case for any variance computation), anything about the pipeline
around the VAE, and anything about host RAM. Those need a real model.
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from datetime import timedelta

import torch
import torch.distributed as dist

# Captured before anything can swap it out, which importing xfuser does.
TORCH_GROUPNORM = torch.nn.GroupNorm


# --------------------------------------------------------------------------------------------
# Collective accounting
# --------------------------------------------------------------------------------------------


class CollectiveLog:
    """Counts and sizes every collective, attributed to the line that issued it

    Wraps the torch.distributed entry points DistVAE uses rather than sampling a profile, so the
    result is exact and cheap enough to leave on during a timed run. The caller is read with
    sys._getframe rather than traceback.extract_stack, which matters at a few thousand calls.
    """

    WRAPPED = (
        "all_reduce",
        "all_gather",
        "all_gather_into_tensor",
        "broadcast",
        "isend",
        "irecv",
        "recv",
        "send",
        "barrier",
        "batch_isend_irecv",
    )

    def __init__(self):
        self.enabled = False
        self.by_call = defaultdict(lambda: {"calls": 0, "bytes": 0})
        self.by_site = defaultdict(lambda: {"calls": 0, "bytes": 0})
        self._originals = {}

    @staticmethod
    def _nbytes(args):
        total = 0
        for arg in args:
            if isinstance(arg, torch.Tensor):
                total += arg.numel() * arg.element_size()
            elif isinstance(arg, (list, tuple)):
                for item in arg:
                    if isinstance(item, torch.Tensor):
                        total += item.numel() * item.element_size()
        return total

    def _wrap(self, name, original):
        def wrapper(*args, **kwargs):
            if self.enabled:
                # Frame 1 is the caller; DistVAE issues these directly, so one level is enough.
                frame = sys._getframe(1)
                site = f"{os.path.basename(frame.f_code.co_filename)}:{frame.f_lineno}"
                size = self._nbytes(args)
                # batch_isend_irecv runs its members through these same entry points, so counting
                # them in the total would charge a batched exchange for the round trips batching
                # is what avoids. They stay visible, under their own heading.
                nested = os.path.basename(frame.f_code.co_filename) == "distributed_c10d.py"
                entry = self.by_call[f"{name} (batched)" if nested else name]
                entry["calls"] += 1
                entry["bytes"] += size
                entry = self.by_site[f"{name} @ {site}"]
                entry["calls"] += 1
                entry["bytes"] += size
            return original(*args, **kwargs)

        return wrapper

    def install(self):
        # Both the package and the module it re-exports from. P2POp checks the op it is handed
        # against distributed_c10d's own isend and irecv, so wrapping only the re-export would
        # make dist.P2POp(dist.isend, ...) - which is how a batched halo exchange is written -
        # fail as an invalid op the moment counting was switched on.
        from torch.distributed import distributed_c10d

        for name in self.WRAPPED:
            original = getattr(dist, name, None)
            if original is None:
                continue
            self._originals[name] = original
            wrapper = self._wrap(name, original)
            setattr(dist, name, wrapper)
            if getattr(distributed_c10d, name, None) is original:
                setattr(distributed_c10d, name, wrapper)

    def reset(self):
        self.by_call.clear()
        self.by_site.clear()

    def report(self):
        return {
            "by_call": {k: dict(v) for k, v in sorted(self.by_call.items())},
            "by_site": {
                k: dict(v)
                for k, v in sorted(
                    self.by_site.items(), key=lambda kv: -kv[1]["calls"]
                )
            },
            "total_calls": sum(
                v["calls"] for k, v in self.by_call.items() if "(batched)" not in k
            ),
            # Bytes from every entry, though: batch_isend_irecv is handed P2POps rather than
            # tensors, so its members are the only place the halo volume can be read.
            "total_bytes": sum(v["bytes"] for v in self.by_call.values()),
        }


LOG = CollectiveLog()

# What sharding is allowed to move the output by, as a fraction of its largest value. Sharding
# changes the order operations happen in, and in bf16 that alone is worth a few percent: the
# measured 0.037 here is the same number whether or not the collectives have been optimised, so
# a tighter bound would only ever catch the dtype. Test the arithmetic in float32.
MAX_REL = {"float32": 1e-4, "float16": 2e-2, "bfloat16": 5e-2}


# --------------------------------------------------------------------------------------------
# VAE architectures, taken from the shipped checkpoints' vae/config.json - weights are random
# --------------------------------------------------------------------------------------------

# Only the fields that change the shape of the work. Anything a class defaults sensibly is left
# out so a diffusers upgrade does not have to be chased here.
FAMILIES = {
    "flux2": dict(
        cls="AutoencoderKLFlux2",
        config=dict(
            in_channels=3,
            out_channels=3,
            latent_channels=32,
            block_out_channels=[128, 256, 512, 512],
            layers_per_block=2,
            norm_num_groups=32,
            down_block_types=["DownEncoderBlock2D"] * 4,
            up_block_types=["UpDecoderBlock2D"] * 4,
            patch_size=[2, 2],
            mid_block_add_attention=True,
            use_quant_conv=True,
            use_post_quant_conv=True,
        ),
        note="black-forest-labs/FLUX.2-dev and FLUX.2-klein-*",
    ),
    "kl": dict(
        cls="AutoencoderKL",
        config=dict(
            in_channels=3,
            out_channels=3,
            latent_channels=16,
            block_out_channels=[128, 256, 512, 512],
            layers_per_block=2,
            norm_num_groups=32,
            down_block_types=["DownEncoderBlock2D"] * 4,
            up_block_types=["UpDecoderBlock2D"] * 4,
        ),
        note="the plain 2D VAE: SD3, Z-Image and friends",
    ),
}


def build_vae(family, dtype, device):
    import diffusers

    spec = FAMILIES[family]
    cls = getattr(diffusers, spec["cls"], None)
    if cls is None:
        raise SystemExit(
            f"the installed diffusers {diffusers.__version__} has no {spec['cls']}; "
            f"--family {family} needs a newer one"
        )
    torch.manual_seed(0)
    return cls(**spec["config"]).eval().to(device=device, dtype=dtype)


def latent_for(vae, height, width, dtype, device, batch=1):
    """A latent of the shape this VAE would decode into batch x height x width

    A batch stands in for xDiT's tile batching, where same-shaped tiles are stacked so that one
    decoder call covers many of them. What that is worth depends on the collective count staying
    flat as the batch grows, which is the thing to read off a run with --batch.
    """
    ratio = getattr(vae, "spatial_compression_ratio", None) or 8
    if height % ratio or width % ratio:
        raise SystemExit(
            f"{height}x{width} is not a whole number of latent rows at a compression "
            f"ratio of {ratio}"
        )
    channels = vae.config.latent_channels
    torch.manual_seed(1)
    return torch.randn(
        batch, channels, height // ratio, width // ratio, dtype=dtype, device=device
    )


# --------------------------------------------------------------------------------------------
# Sharding, via xDiT's own selection where it is installed
# --------------------------------------------------------------------------------------------


def _vae_parallel():
    """xDiT's adapter selection, which is the thing under test and not optional here"""
    # Choosing an adapter here instead would measure this file's opinion of which one fits, and
    # a run would keep going with the wrong one rather than say the installed xDiT is too old.
    try:
        from xfuser.core.utils import vae_parallel
    except ImportError as e:
        raise SystemExit(
            "xfuser.core.utils.vae_parallel is not importable, so there is no adapter selection "
            "to exercise. Point the runner at an xDiT that carries it (-XditBranch)."
        ) from e

    # xDiT does this while validating --use_parallel_vae, before it loads a pipeline: DistVAE's
    # GroupNormAdapter reads num_channels off the norm and AITER's GroupNorm does not carry it,
    # while still subclassing nn.GroupNorm well enough to be selected. A VAE built here rather
    # than by a runner model has to be brought to the same state by hand.
    if torch.nn.GroupNorm.__module__ == "aiter.ops.groupnorm":
        torch.nn.GroupNorm = TORCH_GROUPNORM

    return vae_parallel


def describe(vae, half):
    """What this half is assembled from, and which adapter xDiT picks for it

    Printed whether or not sharding then works, because a refusal or an assertion from inside a
    half-replaced decoder is only readable next to the blocks it was looking at.
    """
    vae_parallel = _vae_parallel()
    part = getattr(vae, half)
    blocks = tuple(getattr(part, "up_blocks" if half == "decoder" else "down_blocks", None) or ())
    chooser = (
        vae_parallel.decoder_adapter_name if half == "decoder" else vae_parallel.encoder_adapter_name
    )
    norm = getattr(part, "conv_norm_out", None)

    # Qualified, because selection is by isinstance and diffusers has more than one class per
    # name: a decoder can report the blocks an adapter wants and still not be the one it means.
    def named(obj):
        cls = type(obj)
        return f"{cls.__module__}.{cls.__name__}"

    from diffusers.models.unets.unet_2d_blocks import DownEncoderBlock2D, UpDecoderBlock2D

    wanted = UpDecoderBlock2D if half == "decoder" else DownEncoderBlock2D
    return {
        "class": named(part),
        "blocks": sorted({named(b) for b in blocks}),
        "blocks_are_2d": all(isinstance(b, wanted) for b in blocks) if blocks else False,
        "mid_block": named(getattr(part, "mid_block", None)),
        "conv_norm_out": named(norm),
        "norm_is_nn_groupnorm": isinstance(norm, torch.nn.GroupNorm),
        "adapter": chooser(vae),
    }


def parallelize(vae, group, half):
    """Shard one half of the VAE, returning the adapter's name"""
    vae_parallel = _vae_parallel()
    if half == "decoder":
        return vae_parallel.parallelize_decoder(vae, group)
    return vae_parallel.parallelize_encoder(vae, group)


# --------------------------------------------------------------------------------------------


def timed(run, iters, device):
    """Median and mean seconds over iters calls, synchronised and with the ranks lined up"""
    samples = []
    for _ in range(iters):
        dist.barrier()
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        run()
        torch.cuda.synchronize(device)
        samples.append(time.perf_counter() - start)
    samples.sort()
    return {
        "median_s": samples[len(samples) // 2],
        "mean_s": sum(samples) / len(samples),
        "min_s": samples[0],
        "max_s": samples[-1],
        "samples_s": samples,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", default="flux2", choices=sorted(FAMILIES))
    parser.add_argument("--half", default="decoder", choices=["decoder", "encoder"])
    parser.add_argument("--height", type=int, default=2048)
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--batch", type=int, default=1,
                        help="latents to decode in one call, standing in for batched tiles")
    parser.add_argument("--max-rel", type=float, default=None,
                        help="agreement tolerance, as a fraction of the reference's largest "
                             "value; defaults by dtype")
    parser.add_argument("--skip-reference", action="store_true",
                        help="skip the single-rank comparison, which needs the whole half to fit on one GPU")
    parser.add_argument("--reference-max-latent-elems", type=int, default=16384,
                        help="above this latent area the reference is skipped on its own: an unsharded "
                             "decode at that size is the thing sharding exists to avoid")
    parser.add_argument("--describe-only", action="store_true",
                        help="report the blocks and the adapter xDiT picks, then stop")
    parser.add_argument("--timeout-min", type=int, default=30,
                        help="process group timeout; the first decode on a new shape pays MIOpen autotune")
    parser.add_argument("--out", default=None, help="write the report here as JSON")
    args = parser.parse_args()

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dtype = getattr(torch, args.dtype)

    dist.init_process_group(
        backend="nccl", init_method="env://", timeout=timedelta(minutes=args.timeout_min)
    )
    group = dist.group.WORLD
    LOG.install()

    # Build the communicator here, while every rank is in the same place. The first collective is
    # what creates it, so if that turns out to be a barrier one rank reaches minutes after the
    # others, the others sit in init until the store times out rather than waiting on the barrier.
    dist.all_reduce(torch.zeros(1, device=device))

    def say(*parts):
        if rank == 0:
            print(*parts, flush=True)

    import diffusers
    import distvae

    say(f"world_size={world_size} device={torch.cuda.get_device_name(local_rank)}")
    say(f"torch={torch.__version__} diffusers={diffusers.__version__} "
        f"distvae={getattr(distvae, '__version__', 'unknown')}")
    say(f"family={args.family} half={args.half} {args.height}x{args.width} dtype={args.dtype}")

    # Before the VAE exists, because importing xfuser swaps torch.nn.GroupNorm for AITER's, and
    # both the adapters and xDiT's selection ask isinstance(norm, nn.GroupNorm). A VAE built
    # first holds the class from before the swap and matches nothing. Real runs import xfuser
    # long before they load a model, so this is the ordering being measured.
    _vae_parallel()

    vae = build_vae(args.family, dtype, device)
    sample = latent_for(vae, args.height, args.width, dtype, device, args.batch)
    say(f"latent {tuple(sample.shape)}")

    built = describe(vae, args.half)
    say(f"{args.half}: {json.dumps(built)}")
    if built["adapter"] is None:
        raise SystemExit(
            f"xDiT has no adapter for this {type(vae).__name__} {args.half}. Nothing to measure."
        )
    if args.describe_only:
        return

    # The reference has to be taken before sharding, which replaces the half in place. Every rank
    # computes it rather than rank 0 alone: the seeds match, so the weights match, and leaving it
    # to one rank would strand the others in the next collective for as long as it takes.
    latent_area = sample.shape[0] * sample.shape[-2] * sample.shape[-1]
    take_reference = not args.skip_reference and latent_area <= args.reference_max_latent_elems
    if not args.skip_reference and not take_reference:
        say(f"no single-rank reference: a {sample.shape[-2]}x{sample.shape[-1]} latent is over "
            f"--reference-max-latent-elems {args.reference_max_latent_elems}, and an unsharded "
            f"decode that size is what sharding exists to avoid. Check agreement at a smaller one.")
    reference = None
    if take_reference:
        with torch.no_grad():
            reference = vae.decode(sample).sample.float().cpu()

    adapter = parallelize(vae, group, args.half)
    say(f"adapter={adapter}")

    def decode():
        with torch.no_grad():
            return vae.decode(sample).sample

    for _ in range(args.warmup):
        decode()
    torch.cuda.synchronize(device)

    # Counted over one decode, so the numbers read per decode rather than per run.
    LOG.reset()
    LOG.enabled = True
    output = decode()
    LOG.enabled = False
    collectives = LOG.report()

    torch.cuda.reset_peak_memory_stats(device)
    timing = timed(decode, args.iters, device)
    peak_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

    agreement = None
    if reference is not None:
        actual = output.float().cpu()
        if actual.shape != reference.shape:
            agreement = {"ok": False, "why": f"shape {tuple(actual.shape)} != {tuple(reference.shape)}"}
        else:
            diff = (actual - reference).abs()
            # Against the reference's own scale, because an absolute tolerance means nothing on
            # random weights, and in bf16 a step at magnitude 1 is already about 0.008.
            scale = reference.abs().max().item()
            relative = diff.max().item() / scale if scale else 0.0
            tolerance = args.max_rel if args.max_rel is not None else MAX_REL[args.dtype]
            agreement = {
                "ok": bool(relative <= tolerance),
                "max_abs": diff.max().item(),
                "mean_abs": diff.mean().item(),
                "reference_max_abs": scale,
                "max_rel_to_scale": relative,
                "max_rel_allowed": tolerance,
            }

    report = {
        "family": args.family,
        "half": args.half,
        "height": args.height,
        "width": args.width,
        "dtype": args.dtype,
        "world_size": world_size,
        "adapter": adapter,
        "latent_shape": list(sample.shape),
        "collectives": collectives,
        "timing": timing,
        "peak_vram_mb": peak_mb,
        "agreement": agreement,
        "versions": {
            "torch": torch.__version__,
            "diffusers": diffusers.__version__,
            "distvae": getattr(distvae, "__version__", "unknown"),
        },
    }

    if rank == 0:
        print("\n--- collectives per decode ---", flush=True)
        for name, entry in collectives["by_call"].items():
            print(f"  {name:<24} {entry['calls']:>6} calls  {entry['bytes'] / 1e6:>10.2f} MB",
                  flush=True)
        print(f"  {'TOTAL':<24} {collectives['total_calls']:>6} calls  "
              f"{collectives['total_bytes'] / 1e6:>10.2f} MB", flush=True)
        print("\n--- top call sites ---", flush=True)
        for site, entry in list(collectives["by_site"].items())[:12]:
            print(f"  {entry['calls']:>6}  {site}", flush=True)
        print(f"\nmedian {timing['median_s'] * 1000:.1f} ms   peak {peak_mb:.0f} MB", flush=True)
        if agreement is not None:
            verdict = "matches" if agreement["ok"] else "DIFFERS FROM"
            print(f"output {verdict} the single-rank reference: {agreement}", flush=True)
        if args.out:
            with open(args.out, "w") as handle:
                json.dump(report, handle, indent=2)
            print(f"\nwrote {args.out}", flush=True)

    dist.barrier()
    dist.destroy_process_group()
    if agreement is not None and not agreement["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
