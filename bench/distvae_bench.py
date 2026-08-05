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

import torch
import torch.distributed as dist


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
                entry = self.by_call[name]
                entry["calls"] += 1
                entry["bytes"] += size
                entry = self.by_site[f"{name} @ {site}"]
                entry["calls"] += 1
                entry["bytes"] += size
            return original(*args, **kwargs)

        return wrapper

    def install(self):
        for name in self.WRAPPED:
            original = getattr(dist, name, None)
            if original is None:
                continue
            self._originals[name] = original
            setattr(dist, name, self._wrap(name, original))

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
            "total_calls": sum(v["calls"] for v in self.by_call.values()),
            "total_bytes": sum(v["bytes"] for v in self.by_call.values()),
        }


LOG = CollectiveLog()


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


def latent_for(vae, height, width, dtype, device):
    """A latent of the shape this VAE would decode into height x width"""
    ratio = getattr(vae, "spatial_compression_ratio", None) or 8
    if height % ratio or width % ratio:
        raise SystemExit(
            f"{height}x{width} is not a whole number of latent rows at a compression "
            f"ratio of {ratio}"
        )
    channels = vae.config.latent_channels
    torch.manual_seed(1)
    return torch.randn(
        1, channels, height // ratio, width // ratio, dtype=dtype, device=device
    )


# --------------------------------------------------------------------------------------------
# Sharding, via xDiT's own selection where it is installed
# --------------------------------------------------------------------------------------------


def parallelize(vae, group, half):
    """Shard one half of the VAE, returning the adapter's name

    Routed through xDiT's vae_parallel when it is available, so the harness exercises the same
    adapter-selection path a real run takes rather than a second copy of that judgement.
    """
    try:
        from xfuser.core.utils import vae_parallel
    except ImportError:
        vae_parallel = None

    if vae_parallel is not None:
        if half == "decoder":
            return vae_parallel.parallelize_decoder(vae, group)
        return vae_parallel.parallelize_encoder(vae, group)

    from distvae.modules.adapters.vae.decoder_adapters import DecoderAdapter
    from distvae.modules.adapters.vae.encoder_adapters import EncoderAdapter

    if half == "decoder":
        vae.decoder = DecoderAdapter(vae.decoder, vae_group=group).to(vae.device)
        return "DecoderAdapter"
    vae.encoder = EncoderAdapter(vae.encoder, vae_group=group).to(vae.device)
    return "EncoderAdapter"


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
    parser.add_argument("--atol", type=float, default=2e-2)
    parser.add_argument("--skip-reference", action="store_true",
                        help="skip the single-rank comparison, which needs the whole half to fit on one GPU")
    parser.add_argument("--out", default=None, help="write the report here as JSON")
    args = parser.parse_args()

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dtype = getattr(torch, args.dtype)

    dist.init_process_group(backend="nccl", init_method="env://")
    group = dist.group.WORLD
    LOG.install()

    def say(*parts):
        if rank == 0:
            print(*parts, flush=True)

    import diffusers
    import distvae

    say(f"world_size={world_size} device={torch.cuda.get_device_name(local_rank)}")
    say(f"torch={torch.__version__} diffusers={diffusers.__version__} "
        f"distvae={getattr(distvae, '__version__', 'unknown')}")
    say(f"family={args.family} half={args.half} {args.height}x{args.width} dtype={args.dtype}")

    vae = build_vae(args.family, dtype, device)
    sample = latent_for(vae, args.height, args.width, dtype, device)
    say(f"latent {tuple(sample.shape)}")

    # The reference before sharding, since sharding replaces the half in place.
    reference = None
    if not args.skip_reference and rank == 0:
        with torch.no_grad():
            reference = vae.decode(sample).sample.float().cpu()
    dist.barrier()

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
            agreement = {
                "ok": bool(diff.max().item() <= args.atol),
                "max_abs": diff.max().item(),
                "mean_abs": diff.mean().item(),
                "atol": args.atol,
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
