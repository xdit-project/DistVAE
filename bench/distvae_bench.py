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

The four arms a comparison usually wants, each differing from the one above it by one thing:

  --no-parallel-vae                              unsharded, untiled: the baseline
  (default)                                      sharded
  --enable-tiling                                sharded and tiled at the VAE's own window
  --vae-tile-size N                              the same, at a narrower window

Tiling is installed by xDiT's own calls in xDiT's own order, so an arm measures the policy that
ships rather than this file's reading of it - with one deliberate exception: --vae-tile-size is
not held at the useful floor the runner clamps to, since measuring below it is how that floor
gets checked. A run down there says so in its tiling facts.

What this cannot tell you: anything about real activation distributions (random weights give
mean~0, variance~1, the easy case for any variance computation), anything about the pipeline
around the VAE, and anything about host RAM. In particular the peak VRAM here is the VAE's own,
which is the whole point of measuring it apart - but it is NOT a run's peak, and a window that
halves the decode's memory moves a run's peak only while the VAE is what peaks. Those need a
real model.
"""

import argparse
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import torch
import torch.distributed as dist

# Captured before anything can swap it out, which importing xfuser does.
TORCH_GROUPNORM = torch.nn.GroupNorm


# --------------------------------------------------------------------------------------------
# What produced a measurement, so a result file can travel
# --------------------------------------------------------------------------------------------

# Bumped when the shape of a report changes in a way a reader has to know about.
SCHEMA = 1


def _git(cwd, *argv):
    try:
        done = subprocess.run(
            ["git", *argv], cwd=cwd, capture_output=True, text=True, timeout=30
        )
        return done.stdout.strip() if done.returncode == 0 else None
    except Exception:  # noqa: BLE001 - provenance never fails a run
        return None


def _checkout(module):
    """The branch and commit a package was installed from, where it came from a git checkout

    Branches are how this work moves between machines, so a version string cannot say what ran:
    two boxes can both hold "0.0.0b5" and disagree about everything that matters.
    """
    location = getattr(module, "__file__", None)
    if not location:
        return None
    start = Path(location).resolve().parent
    for parent in [start, *start.parents]:
        if not (parent / ".git").exists():
            continue
        return {
            "branch": _git(parent, "rev-parse", "--abbrev-ref", "HEAD"),
            "commit": _git(parent, "rev-parse", "--short", "HEAD"),
            "dirty": bool(_git(parent, "status", "--porcelain")),
        }
    return None


def _installed():
    """What is loaded, asked of sys.modules rather than by importing

    Importing xfuser to read its version would swap torch's GroupNorm for AITER's, and where in
    the run that swap happens is part of what this bench measures. Anything not already imported
    by now was not going to be used.
    """
    found = {}
    for name in ("torch", "diffusers", "distvae", "xfuser"):
        module = sys.modules.get(name)
        if module is None:
            continue
        found[name] = {
            "version": getattr(module, "__version__", None),
            "checkout": _checkout(module),
        }
    return found


def _this_script():
    """This file's own identity, which its installed library's commit does not give

    The bench script travels by other means than the package does - copied to a box, mounted into
    a container, delivered by ConfigMap - so the commit of the DistVAE beside it is no evidence at
    all of what was actually run. A digest is, and it costs one read of a file already on disk.
    """
    try:
        source = Path(__file__).resolve()
        return {
            "path": str(source),
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest()[:12],
            "checkout": _checkout(sys.modules[__name__]),
        }
    except Exception:  # noqa: BLE001 - provenance never fails a run
        return None


def _hardware(device_index):
    if not torch.cuda.is_available():
        return {"family": os.environ.get("HW_FAMILY", "cpu")}
    card = torch.cuda.get_device_properties(device_index)
    arch = getattr(card, "gcnArchName", "") or f"sm_{card.major}{card.minor}"
    return {
        # Deliberately not looked up in a table of known devices: whoever brings a machine up
        # sets HW_FAMILY, and until they do the architecture string stands in, which is wrong in
        # no way except being harder to read.
        "family": os.environ.get("HW_FAMILY") or arch,
        "product": card.name,
        "arch": arch,
        "vram_gib": round(card.total_memory / (1024 ** 3), 1),
        "visible": torch.cuda.device_count(),
        "runtime": (
            f"rocm {torch.version.hip}" if getattr(torch.version, "hip", None)
            else f"cuda {getattr(torch.version, 'cuda', None)}"
        ),
    }


REPORT_BEGIN = "===== BEGIN DISTVAE REPORT ====="
REPORT_END = "===== END DISTVAE REPORT ====="


def write_report(path, world_size, device_index, body):
    """Emit a result that answers for itself what produced it

    A report arriving from another machine cannot be read back out of its numbers: which card,
    which branch of DistVAE, which xDiT beside it, what was asked. Someone otherwise keeps that
    in a message alongside the file, and eventually keeps it wrong.

    Always to stdout as well as to the file, because the filesystem it was written to is often
    the thing that does not survive the run: a pod deleted the moment it finishes, a container
    someone brought up over ssh. The log is what comes back from those, so the report has to be
    in it.
    """
    report = {
        "schema": SCHEMA,
        "ran": {
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "host": socket.gethostname(),
            "python": platform.python_version(),
            "world_size": world_size,
            "hardware": _hardware(device_index),
            "installed": _installed(),
            "bench_script": _this_script(),
            "argv": list(sys.argv),
        },
        **body,
    }
    if path:
        with open(path, "w") as handle:
            json.dump(report, handle, indent=2)
        print(f"\nwrote {path}", flush=True)
    print(f"\n{REPORT_BEGIN}")
    print(json.dumps(report, indent=2))
    print(REPORT_END, flush=True)


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


def across_ranks(by_call, world_size):
    """The same counts as the busiest rank sees them, rather than as rank 0 does

    Rank 0 borders one neighbour where the ranks in the middle border two, so it sends and
    receives less of a halo than they do and its count understates the decode. What bounds the
    decode is the rank doing the most, since every collective is one they all wait on.
    """
    gathered = [None] * world_size
    dist.all_gather_object(gathered, {name: entry["calls"] for name, entry in by_call.items()})

    def total(counts):
        return sum(calls for name, calls in counts.items() if "(batched)" not in name)

    names = sorted({name for counts in gathered for name in counts})
    return {
        "by_call_max": {name: max(counts.get(name, 0) for counts in gathered) for name in names},
        "total_calls_max": max(total(counts) for counts in gathered),
        "total_calls_by_rank": [total(counts) for counts in gathered],
    }

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
#
# spatial and temporal are the compression ratios, and latent_channels the width of the latent.
# They are all readable off a built VAE on some classes and not on others, under a different name
# again on Wan, so they are stated here where the config they came from states them.
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
        latent_channels=32,
        spatial=8,
        temporal=None,
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
            # The tile window IS this number: AutoencoderKL assigns tile_sample_min_size from it
            # outright. The class defaults it to 32, which no shipped checkpoint carries, and a
            # 2048x2048 decode at a 32px window is four thousand tiles of nothing. SD3 and SDXL
            # both ship 1024.
            sample_size=1024,
        ),
        latent_channels=16,
        spatial=8,
        temporal=None,
        note="the plain 2D VAE: SD3, Z-Image and friends",
    ),
    "wan": dict(
        cls="AutoencoderKLWan",
        config=dict(
            base_dim=96,
            z_dim=16,
            dim_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            attn_scales=[],
            temperal_downsample=[False, True, True],
        ),
        latent_channels=16,
        spatial=8,
        temporal=4,
        note="Wan2.1 and Wan2.2 ship the same VAE config",
    ),
    "qwen_image": dict(
        cls="AutoencoderKLQwenImage",
        config=dict(
            base_dim=96,
            z_dim=16,
            dim_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            attn_scales=[],
            temperal_downsample=[False, True, True],
        ),
        latent_channels=16,
        spatial=8,
        # Qwen-Image's VAE is Wan's down to the numbers, frame axis included, and a still image
        # goes through it as a clip of one frame: run it with --frames 1.
        temporal=4,
        note="Qwen/Qwen-Image-2512 and Qwen-Image-Edit",
    ),
    "hunyuan_video": dict(
        cls="AutoencoderKLHunyuanVideo",
        config=dict(
            in_channels=3,
            out_channels=3,
            latent_channels=16,
            block_out_channels=[128, 256, 512, 512],
            layers_per_block=2,
            norm_num_groups=32,
            mid_block_add_attention=True,
            spatial_compression_ratio=8,
            temporal_compression_ratio=4,
        ),
        latent_channels=16,
        spatial=8,
        temporal=4,
        note="hunyuanvideo-community/HunyuanVideo",
    ),
    "hunyuan_video_15": dict(
        cls="AutoencoderKLHunyuanVideo15",
        config=dict(
            in_channels=3,
            out_channels=3,
            latent_channels=32,
            block_out_channels=[128, 256, 512, 1024, 1024],
            layers_per_block=2,
            downsample_match_channel=True,
            upsample_match_channel=True,
            spatial_compression_ratio=16,
            temporal_compression_ratio=4,
        ),
        latent_channels=32,
        spatial=16,
        temporal=4,
        note="hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-*",
    ),
    "ltx2": dict(
        cls="AutoencoderKLLTX2Video",
        config=dict(
            in_channels=3,
            out_channels=3,
            latent_channels=128,
            block_out_channels=[256, 512, 1024, 2048],
            decoder_block_out_channels=[256, 512, 1024],
            layers_per_block=[4, 6, 6, 2, 2],
            decoder_layers_per_block=[5, 5, 5, 5],
            spatio_temporal_scaling=[True, True, True, True],
            decoder_spatio_temporal_scaling=[True, True, True],
            decoder_inject_noise=[False, False, False, False],
            downsample_type=["spatial", "temporal", "spatiotemporal", "spatiotemporal"],
            upsample_factor=[2, 2, 2],
            upsample_residual=[True, True, True],
            encoder_causal=True,
            decoder_causal=False,
            encoder_spatial_padding_mode="zeros",
            decoder_spatial_padding_mode="reflect",
            patch_size=4,
            patch_size_t=1,
            resnet_norm_eps=1e-06,
            spatial_compression_ratio=32,
            temporal_compression_ratio=8,
        ),
        latent_channels=128,
        spatial=32,
        temporal=8,
        note="Lightricks/LTX-2; the 2.3 checkpoint differs in the decoder's shape",
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


def sample_for(spec, half, height, width, dtype, device, batch=1, frames=1):
    """What this half is handed: a latent for the decoder, an image or clip for the encoder

    A batch stands in for xDiT's tile batching, where same-shaped tiles are stacked so that one
    call covers many of them. What that is worth depends on the collective count staying flat as
    the batch grows, which is the thing to read off a run with --batch.
    """
    ratio = spec["spatial"]
    if height % ratio or width % ratio:
        raise SystemExit(
            f"{height}x{width} is not a whole number of latent rows at a compression "
            f"ratio of {ratio}"
        )
    temporal = spec["temporal"]
    if temporal and (frames - 1) % temporal:
        raise SystemExit(
            f"--frames {frames} does not land on a whole number of latent frames: these VAEs "
            f"keep the first frame and compress the rest by {temporal}, so ask for "
            f"1 + a multiple of {temporal}"
        )
    if half == "decoder":
        channels = spec["latent_channels"]
        rows, columns = height // ratio, width // ratio
        depth = 1 + (frames - 1) // temporal if temporal else None
    else:
        channels = spec["config"].get("in_channels", 3)
        rows, columns = height, width
        depth = frames if temporal else None
    shape = (batch, channels, rows, columns)
    if depth is not None:
        shape = (batch, channels, depth, rows, columns)
    torch.manual_seed(1)
    return torch.randn(*shape, dtype=dtype, device=device)


def run_half(vae, half, sample):
    """One call through the half under test, returning the tensor to compare"""
    if half == "decoder":
        return vae.decode(sample).sample
    encoded = vae.encode(sample)
    # Take the mean rather than a draw from it: two runs have to be comparable, and the sampling
    # is not what sharding changes. Newer classes hand back the latent directly.
    distribution = getattr(encoded, "latent_dist", None)
    return distribution.mean if distribution is not None else encoded.latent


# --------------------------------------------------------------------------------------------
# Sharding, via xDiT's own selection where it is installed
# --------------------------------------------------------------------------------------------


def _restore_torch_groupnorm():
    """Undo AITER's GroupNorm swap, which importing xfuser performs

    xDiT does this while validating --use_parallel_vae, before it loads a pipeline: DistVAE's
    GroupNormAdapter reads num_channels off the norm and AITER's GroupNorm does not carry it,
    while still subclassing nn.GroupNorm well enough to be selected. A VAE built here rather than
    by a runner model has to be brought to the same state by hand.
    """
    if torch.nn.GroupNorm.__module__ == "aiter.ops.groupnorm":
        torch.nn.GroupNorm = TORCH_GROUPNORM


def _vae_parallel():
    """xDiT's adapter selection, which is the thing under test and not optional here"""
    # Choosing an adapter here instead would measure this file's opinion of which one fits, and
    # a run would keep going with the wrong one rather than say the installed xDiT is too old.
    try:
        from xfuser.core.utils import vae_parallel
    except ImportError as e:
        raise SystemExit(
            "xfuser.core.utils.vae_parallel is not importable, so there is no adapter selection "
            "to exercise. Point the runner at an xDiT that carries it (-XditBranch), or ask for "
            "the `main` arm, which is what an xDiT without it can still do."
        ) from e

    _restore_torch_groupnorm()
    return vae_parallel


# --------------------------------------------------------------------------------------------
# The same two features as xDiT main composes them, which is the baseline everything else moves
# --------------------------------------------------------------------------------------------

# main has no adapter selection: each runner model names the DistVAE class it wants in its own
# _setup_parallel_vae, and DistVAE main carries only these two. The families missing here are not
# an omission - no runner model on main names an adapter for them, and DistVAE main has none to
# name, so there is no baseline to measure and the branch is the first thing that can do it.
MAIN_ADAPTERS = {
    "flux2": "DecoderAdapter",       # xFuserFlux2Model, via flux.py's _setup_parallel_vae
    "kl": "DecoderAdapter",          # the same call in every 2D runner model
    "wan": "WanDecoderAdapter",      # wan.py's own copy of it
}


def parallelize_as_main_does(vae, group, family, half):
    """Shard the decoder by naming a class, as main's runner models do"""
    if half != "decoder":
        raise ValueError(
            "main shards no encoder for these families, so there is no encoder baseline"
        )
    name = MAIN_ADAPTERS.get(family)
    if name is None:
        raise ValueError(
            f"nothing on xDiT main shards a {family} VAE: DistVAE main carries only "
            f"{sorted(set(MAIN_ADAPTERS.values()))} and no runner model names one for this "
            f"family, so this cell has no baseline rather than a slow one"
        )
    from distvae.modules.adapters.vae import decoder_adapters

    adapter = getattr(decoder_adapters, name, None)
    if adapter is None:
        raise ValueError(
            f"the installed DistVAE has no {name}; the `main` arm needs DistVAE main "
            f"(-DistVaeBranch main)"
        )
    vae.decoder = adapter(vae.decoder, vae_group=group).to(vae.device)
    return f"{name} (named, not selected)"


def _native_window(vae):
    """The VAE's own pixel tile window, read without xDiT, since main's arm has no xDiT to read
    it with"""
    windows = {
        value
        for attr in ("tile_sample_min_size", "tile_sample_min_height", "tile_sample_min_width")
        if isinstance(value := getattr(vae, attr, None), int) and value > 0
    }
    return windows.pop() if len(windows) == 1 else None


def tile_as_main_does(vae):
    """Turn tiling on the way main does, which is one call and no window to choose

    main's _enable_options is `self.pipe.vae.enable_tiling()` and nothing else: diffusers' own
    loop at the VAE's own window, decoding one tile at a time on every rank. There is no
    --vae_tile_size on main, so the window is not a lever this arm has.
    """
    vae.enable_tiling()
    return {
        "enabled": True,
        "requested_window": None,
        "window_px": _native_window(vae),
        "tile_latent_area": _latent_area(vae),
        "as_main_does": True,
    }


def describe(vae, half, family, select=True):
    """What this half is assembled from, and which adapter xDiT picks for it

    Printed whether or not sharding then works, because a refusal or an assertion from inside a
    half-replaced decoder is only readable next to the blocks it was looking at.

    `select` off is the main arm, which has no selection to report: main names an adapter per
    runner model, so the only answer available there is this file's table of what it names.
    """
    part = getattr(vae, half)
    blocks = tuple(getattr(part, "up_blocks" if half == "decoder" else "down_blocks", None) or ())
    chooser = None
    if select:
        vae_parallel = _vae_parallel()
        chooser = (
            vae_parallel.decoder_adapter_name if half == "decoder"
            else vae_parallel.encoder_adapter_name
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
        "adapter": chooser(vae) if chooser else MAIN_ADAPTERS.get(family),
    }


def parallelize(vae, group, half):
    """Shard one half of the VAE, returning the adapter's name"""
    vae_parallel = _vae_parallel()
    if half == "decoder":
        return vae_parallel.parallelize_decoder(vae, group)
    return vae_parallel.parallelize_encoder(vae, group)


# --------------------------------------------------------------------------------------------
# Tiling, in the order and by the calls the runner uses
# --------------------------------------------------------------------------------------------


def _vae_tiling():
    """xDiT's tiling knowledge, which is the other half of what these arms measure"""
    try:
        from xfuser.core.utils import vae_tiling
    except ImportError as e:
        raise SystemExit(
            "xfuser.core.utils.vae_tiling is not importable, so there is no tiling policy to "
            "exercise. Point the runner at an xDiT that carries it (-XditBranch)."
        ) from e
    return vae_tiling


def deals_tiles_out(vae) -> bool:
    """Whether this VAE's tiles can go out to a group whole, which is xDiT's answer and not this
    file's"""
    vae_tiling = _vae_tiling()
    if not hasattr(vae_tiling, "supports_tile_parallel"):
        raise SystemExit(
            "the installed xDiT does not deal tiles out across a group, so --tile-split tiles is "
            "not something it can be measured doing. Point the runner at an xDiT that carries it "
            "(-XditBranch), or ask for --tile-split rows."
        )
    return vae_tiling.supports_tile_parallel(vae)


# Seconds spent inside each phase of a tiled decode, summed over however many decodes ran since
# the last reset. Only filled when --phase-timing asks for it, since reading them means
# synchronising the device around each tile and that is not what the timed arms should measure.
PHASES = Counter()


def time_the_decoder(vae, device):
    """Record what every decoder call costs, wherever in the loop it was made from

    The question this answers is where a tiled decode's time actually goes: into the decoder, or
    into everything either side of it - slicing the latent, blending each tile into the canvas,
    and the exchanges. Wrapping the decoder rather than the dispatcher is what lets the three
    ways of splitting a decode be read against each other, since only one of them routes its
    calls through a dispatcher at all.
    """
    import torch.nn as nn

    class Timed(nn.Module):
        def __init__(self, decoder):
            super().__init__()
            self.decoder = decoder

        def forward(self, *args, **kwargs):
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            out = self.decoder(*args, **kwargs)
            torch.cuda.synchronize(device)
            PHASES["decode_s"] += time.perf_counter() - start
            PHASES["calls"] += 1
            return out

    vae.decoder = Timed(vae.decoder)


def timing_decode(tiled_decode, device):
    """The whole tiled decode, timed, so that the phases can be read against it"""
    def timed_decode(z, return_dict: bool = True):
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        out = tiled_decode(z, return_dict=return_dict)
        torch.cuda.synchronize(device)
        PHASES["decode_total_s"] += time.perf_counter() - start
        PHASES["decodes"] += 1
        return out

    return timed_decode


def phase_report(group=None, world_size=1):
    """What one tiled decode spent in each phase, in ms, empty unless --phase-timing asked

    `rest_ms` is the whole decode less the decoder itself: slicing the latent, blending each tile
    into the canvas, and the exchanges. That remainder is the part no scheme here divides by
    adding ranks unless it divides the blending, so it is what says whether one can.

    The spread of `decoder_ms` across the ranks is the other half of the story. Every scheme ends
    in a gather, so the slowest rank sets the pace, and a split that hands one rank more tiles
    than another pays that difference whatever it saved elsewhere.
    """
    decodes = PHASES.get("decodes", 0)
    if not decodes:
        return {}
    total = PHASES["decode_total_s"] / decodes
    decode = PHASES["decode_s"] / decodes
    report = {
        "total_ms": round(total * 1e3, 1),
        "decoder_ms": round(decode * 1e3, 1),
        "rest_ms": round((total - decode) * 1e3, 1),
        "calls_per_decode": round(PHASES["calls"] / decodes, 1),
    }
    if world_size > 1:
        share = [None] * world_size
        dist.all_gather_object(share, (decode, PHASES["calls"] / decodes), group=group)
        report["decoder_ms_by_rank"] = [round(one * 1e3, 1) for one, _ in share]
        report["calls_by_rank"] = [round(many, 1) for _, many in share]
        # One rank waiting on another is time no rank spends decoding. Stated as a share of the
        # slowest rank, so it reads the same whatever the shape costs.
        slowest = max(one for one, _ in share)
        idle = sum(slowest - one for one, _ in share) / (world_size * slowest or 1)
        report["idle_share"] = round(idle, 3)
    return report


def _latent_area(vae):
    """The latent area of the VAE's current tile, None where it has no square latent window"""
    size = getattr(vae, "tile_latent_min_size", None)
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        return None
    return size * size


def overlap_token(text):
    """A requested overlap: a fraction, or `half` for half of whatever this VAE's own is

    Named rather than numeric because the VAE's own overlap differs by family - a quarter on
    AutoencoderKL, elsewhere whatever its stride happens to work out to - so no single number
    means "half the default" across a sweep, and a table of per-family numbers is one that goes
    stale the first time a config changes upstream.
    """
    text = text.strip().lower()
    if text == "half":
        return "half"
    try:
        return float(text)
    except ValueError:
        raise SystemExit(f"--tile-overlap takes a fraction or `half`, not {text!r}") from None


def overlap_label(overlap):
    """What to call an arm measured at this overlap"""
    return overlap if isinstance(overlap, str) else f"{overlap:g}"


def _set_tile_overlap(vae, overlap, facts, say):
    """Widen the stride so tiles overlap by `overlap` of a tile rather than the VAE's own share

    A window is one lever on a tile grid and the overlap is the other, and only the first is
    exposed anywhere. They do different things: the window sets how big a tile is, which is what
    peak memory follows, while the overlap sets how much of the image is decoded twice, which is
    what the total work follows and what no window can change - scaling a window scales the stride
    with it and leaves the ratio where it was.

    Left in the harness rather than pushed into xDiT, because what it costs is seam fidelity and
    that is measured here, against the untiled reference, before anything is recommended.
    """
    facts["requested_overlap"] = overlap

    if overlap == "half":
        own = _vae_tiling().tile_overlap(vae)
        if own is None:
            raise SystemExit(
                f"--tile-overlap half has nothing to halve on this {type(vae).__name__}: it "
                f"reports no overlap of its own to read."
            )
        # The down and across shares are the same on every VAE here, and where they are not the
        # narrower one is the one that bounds the seam.
        overlap = min(own) / 2
        facts["own_overlap"] = min(own)
        say(f"tile overlap half of the VAE's own {min(own):.1%}, so {overlap:.1%}")

    # The overlap-factor family states it as a fraction already and there is nothing to round.
    if hasattr(vae, "tile_overlap_factor"):
        vae.tile_overlap_factor = overlap
        facts["overlap"] = overlap
        say(f"tile overlap factor set to {overlap:.1%}")
        return

    ratio = _vae_tiling().spatial_ratio(vae)
    if ratio is None or not hasattr(vae, "tile_sample_stride_height"):
        raise SystemExit(
            f"--tile-overlap has nothing to set on this {type(vae).__name__}: it reports neither "
            f"a tile_overlap_factor nor a pixel stride over a known compression ratio."
        )

    edges, strides = [], []
    for edge_attr, stride_attr in (
        ("tile_sample_min_height", "tile_sample_stride_height"),
        ("tile_sample_min_width", "tile_sample_stride_width"),
    ):
        edge = getattr(vae, edge_attr)
        # A stride walks the latent, so it has to land on a whole latent pixel; asking for one
        # that does not is rounded to the nearest that does and reported back as what it became.
        latent = min(edge // ratio, max(1, round(edge * (1.0 - overlap) / ratio)))
        setattr(vae, stride_attr, latent * ratio)
        edges.append(edge)
        strides.append(latent * ratio)

    facts.update(overlap=1.0 - strides[0] / edges[0], stride_px=strides[0])
    say(f"tile overlap set to {facts['overlap']:.1%}: a {edges[0]}px tile every {strides[0]}px")


def setup_tiling(
    vae, window, world_size, say, group=None, phase_timing=False, tile_split="tiles",
    overlap=None,
):
    """Turn tiling on the way the runner does, returning what it settled on

    Unlike the runner this does NOT hold the window at or above
    `vae_tiling.narrowest_useful_window`, because measuring below that floor is how the floor was
    found; `below_useful_floor` in the returned facts says when a run is down there. Nothing else
    here should differ from what the runner installs.

    A `group` is the tiles being dealt out across it, which is what the runner does instead of
    sharding when both flags are on. The decoder is then unsharded and the tile a rank is given
    is decoded whole.
    """
    vae_tiling = _vae_tiling()
    vae_tiling.require_vae_support(vae, "tiling", "--enable-tiling")
    vae.enable_tiling()

    native = vae_tiling.tile_window(vae)
    floor = vae_tiling.narrowest_useful_window(vae)
    facts = {
        "enabled": True,
        "requested_window": window,
        "window_px": native,
        "default_tile_latent_area": _latent_area(vae),
        "narrowest_useful_window_px": floor,
    }

    if window in ("half", "quarter"):
        if native is None:
            raise SystemExit(
                f"--vae-tile-size {window} needs a window to take a fraction of, and this "
                f"{type(vae).__name__} does not report one."
            )
        window = native // (2 if window == "half" else 4)
    elif window is not None:
        window = int(window)

    if window is not None:
        pixels, plan = vae_tiling.snap_tile_window(vae, window)
        if plan is None:
            raise SystemExit(
                f"no workable tile window at or below {window}px for this "
                f"{type(vae).__name__}: every candidate divides its tiling attributes into "
                f"something fractional."
            )
        # A tile is sharded over its rows, so a tile thinner than the group leaves some rank
        # holding nothing. The runner refuses rather than deadlocking inside the decoder. Dealing
        # whole tiles out divides nothing inside a tile, so the width of one stops mattering.
        rows = vae_tiling.latent_rows(vae, plan)
        if group is None and world_size > 1 and rows is not None and rows < world_size:
            raise SystemExit(
                f"a {pixels}px tile holds {rows} latent rows, fewer than the {world_size} ranks "
                f"sharding it. Ask for a wider window."
            )
        vae_tiling.apply_tile_plan(vae, plan)
        facts.update(snapped_window_px=pixels, tile_latent_rows=rows)
        if pixels != window:
            say(f"tile window snapped {window} -> {pixels}px, the widest that lands whole")

    if overlap is not None:
        _set_tile_overlap(vae, overlap, facts, say)

    facts["tile_latent_area"] = _latent_area(vae)
    facts["tile_parallel"] = group is not None
    snapped = facts.get("snapped_window_px", native)
    facts["below_useful_floor"] = bool(
        floor is not None and snapped is not None and snapped < floor
    )
    if facts["below_useful_floor"]:
        say(f"note: {snapped}px is below this VAE's {floor}px useful floor, which the runner "
            f"would have clamped; measuring it anyway.")

    dealing = group is not None
    dispatch = assemble = None
    if dealing:
        from xfuser.core.utils import vae_tile_parallel

        dispatch, assemble = vae_tile_parallel.sharing(group)
        if tile_split == "scattered":
            # The tiles still go out whole, but scattered through the grid rather than in a band,
            # which is what leaves the blending on every rank. Kept so the two can be measured
            # against each other rather than argued about.
            assemble = None
        facts["tile_split"] = tile_split
    if phase_timing:
        time_the_decoder(vae, vae.device)

    # A tile to a call either way, so without a group to deal to there is nothing to install and
    # the arm measures diffusers' own loop.
    if not dealing and dispatch is None:
        return facts

    if dealing:
        batched = vae_tiling.tiled_decode_for(vae, dispatch, assemble)
    else:
        batched = vae_tiling.overlap_tiled_decode(vae, dispatch)
    if batched is None:
        # A family whose loop xDiT does not reimplement keeps its own, so there is nothing to
        # install and the arm still measures upstream tiling rather than silently measuring
        # nothing.
        say(f"no reimplemented tiled_decode for {type(vae).__name__}. Measuring upstream tiling.")
        if phase_timing:
            vae.tiled_decode = timing_decode(vae.tiled_decode, vae.device)
        return facts
    vae.tiled_decode = timing_decode(batched, vae.device) if phase_timing else batched
    return facts


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


def tile_shape_costs(args, spec, device, dtype, say):
    """What each tile shape costs, against what its latent area says it should

    A grid is a few full-window tiles and a fringe of smaller ones, because the latent bounds
    clip the last row and the last column. The split weighs a tile by the area it covers, which
    is the right weight only if a tile of half the area costs half as much. It need not: an odd
    convolution shape can miss the kernels a square one is tuned for, and then the fringe is
    dearer than it reads and any split that gathers the fringe onto one rank is slower than the
    weighing promised.

    Timed apart from any grid so that nothing else is in the way: one decode, one tile shape.
    """
    vae = build_vae(args.family, dtype, device)
    window = _vae_tiling().tile_window(vae)
    if window is None:
        raise SystemExit(
            f"--family {args.family} sizes its tile height and width apart, so there is no one "
            f"window to clip against and no shape here that stands for a grid's fringe"
        )
    side = window // spec["spatial"]
    depth = 1 + (args.frames - 1) // spec["temporal"] if spec["temporal"] else None
    say(f"latent tile window {side}x{side}"
        + (f", {depth} latent frames of {args.frames}" if depth else ""))

    shapes = []
    if args.tile_shape_sides:
        # A narrowed window gives square tiles, and the small end of that is where batching is
        # supposed to pay for itself, so it is worth reaching below anything this window clips to.
        for text in args.tile_shape_sides.split(","):
            shapes.append((int(text), int(text)))
    else:
        for down in (1, 2, 4):
            for across in (1, 2, 4):
                shape = (side // down, side // across)
                if min(shape) >= 8 and shape not in shapes:
                    shapes.append(shape)

    # A rank does not decode its tiles one by one: same-shaped tiles are stacked and decoded in
    # one call under the batch budget. How many stack together depends on the shape, so two ranks
    # holding the same area can still be making very different calls, and a batch that does not
    # scale with its count would cost the rank holding the smaller shapes.
    counts, count = [], 1
    while count <= args.tile_shape_batch:
        counts.append(count)
        count *= 2
    measured, full, alone = [], None, {}
    for rows, columns in shapes:
        for count in counts:
            size = (count, spec["latent_channels"], rows, columns)
            if depth is not None:
                size = (count, spec["latent_channels"], depth, rows, columns)
            torch.manual_seed(1)
            latent = torch.randn(*size, dtype=dtype, device=device)
            torch.cuda.reset_peak_memory_stats(device)
            try:
                for _ in range(args.warmup):
                    run_half(vae, "decoder", latent)
                ms = timed(
                    lambda: run_half(vae, "decoder", latent), args.iters, device
                )["median_s"] * 1000
            except torch.OutOfMemoryError:
                # A batch that does not fit is a finding, not a failure: it is the budget asking
                # for a call the device cannot make. Bigger batches of this shape need not be
                # timed to know they will not fit either.
                say(f"  {rows:>4} x {columns:<4} x{count}  out of memory")
                del latent
                torch.cuda.empty_cache()
                measured.append({
                    "rows": rows, "columns": columns, "tiles_in_the_call": count,
                    "latent_area": rows * columns, "out_of_memory": True,
                })
                break
            peak = torch.cuda.max_memory_allocated(device) / 1024 ** 2
            each = ms / count
            area = rows * columns
            if count == 1:
                alone[(rows, columns)] = each
                if full is None:
                    full = (each, area)
            # What the split believes a tile of this shape costs, against the clock.
            predicted = full[0] * area / full[1]
            measured.append({
                "rows": rows,
                "columns": columns,
                "tiles_in_the_call": count,
                "latent_area": area,
                "ms": ms,
                "ms_per_tile": each,
                "peak_mb": peak,
                "ms_per_1k_latent_area": each / area * 1000,
                "against_what_area_predicts": each / predicted,
                "against_the_same_tile_alone": each / alone[(rows, columns)],
            })
            say(f"  {rows:>4} x {columns:<4} x{count}  area {area:>7}  {ms:8.1f} ms  "
                f"{each:8.1f} ms per tile  peak {peak:7.0f} MB  "
                f"{each / predicted:5.2f}x what area predicts  "
                f"{each / alone[(rows, columns)]:5.2f}x the same tile alone")
            del latent
            torch.cuda.empty_cache()

    fitted = [r for r in measured if not r.get("out_of_memory")]
    batched = [r for r in fitted if r["tiles_in_the_call"] > 1]
    if batched:
        worst = max(batched, key=lambda r: r["against_the_same_tile_alone"])
        say(f"\nstacking tiles into one call is worst at {worst['rows']}x{worst['columns']} "
            f"{worst['tiles_in_the_call']} to a call, where each tile costs "
            f"{worst['against_the_same_tile_alone']:.2f}x what it costs decoded alone")
    dearest = max(fitted, key=lambda r: r["against_what_area_predicts"])
    say(f"\nthe dearest tile against its area is {dearest['rows']}x{dearest['columns']} "
        f"{dearest['tiles_in_the_call']} to a call, at "
        f"{dearest['against_what_area_predicts']:.2f}x, so a split that weighs by area alone "
        f"under-charges it by {(dearest['against_what_area_predicts'] - 1) * 100:.0f}%")
    return {"family": args.family, "latent_window": side, "frames": args.frames, "shapes": measured}


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
                        help="latents to decode in one call. Not tile batching, which happens "
                             "inside tiled_decode: this stacks whole independent latents")
    parser.add_argument("--no-parallel-vae", action="store_true",
                        help="leave the VAE unsharded, for the baseline arm every other arm is "
                             "measured against. Every rank then decodes the whole thing")
    parser.add_argument("--enable-tiling", action="store_true",
                        help="tile the decode at the VAE's own window, as --enable_tiling does")
    parser.add_argument("--vae-tile-size", default=None,
                        help="narrow the tile window to this many pixels, as --vae_tile_size does; "
                             "implies --enable-tiling. Also takes 'half' or 'quarter', which is what "
                             "one matrix across families needs: each VAE has its own native window, "
                             "so a fixed number is a different fraction of it for every one of them")
    parser.add_argument("--tile-overlap", default=None,
                        help="overlap the tiles by this fraction of a tile instead of by the "
                             "VAE's own share, comma separated for several, e.g. '0.25,0.125,0'. "
                             "`half` means half of whatever this VAE's own overlap is, which is "
                             "the only way to say that once across families that do not share a "
                             "default. "
                             "Crossed with the tiled arms, so each one is measured at each "
                             "overlap. This is the lever the window is not: a window sets how big "
                             "a tile is and so what memory peaks at, while the overlap sets how "
                             "much of the image is decoded twice and so what the work totals - "
                             "and scaling a window scales the stride with it, leaving that ratio "
                             "exactly where it was")
    parser.add_argument("--grid-arms", default=None,
                        help="measure several arms in ONE process, comma separated, e.g. "
                             "'none,pvae,tile,tile-half'. Most of a pod's wall clock is startup, "
                             "install, imports and building the VAE, none of which a second arm "
                             "needs to pay again, so a grid of 16 costs far less than 16 runs")
    parser.add_argument("--grid-shapes", default=None,
                        help="shapes to cross the arms with, comma separated, HxW or HxWxFRAMES, "
                             "e.g. '1024x1024,2048x2048,4096x4096'. Defaults to --height/--width")
    parser.add_argument("--tile-split", choices=["tiles", "scattered", "rows"], default="tiles",
                        help="what the group divides when both tiling and parallel VAE are on. "
                             "tiles gives each rank a band of tile rows, which divides the "
                             "blending too; scattered deals whole tiles round-robin, which "
                             "leaves the blending on every rank; rows shards inside every tile, "
                             "which is what composing the two flags did before either was an "
                             "option. All three are kept so they can be measured against "
                             "each other")
    parser.add_argument("--tile-shape-costs", action="store_true",
                        help="time a decode at each tile shape a grid contains, full window and "
                             "clipped, and report what each costs against what its area predicts. "
                             "Answers whether weighing a split by latent area is weighing the "
                             "right thing. Runs on its own, ignoring the arms and shapes")
    parser.add_argument("--tile-shape-batch", type=int, default=1,
                        help="how many tiles to stack into one call in --tile-shape-costs, which "
                             "is what the batch budget does on a rank holding several tiles of "
                             "one shape. 1 times each shape alone, and anything above doubles up "
                             "to it")
    parser.add_argument("--tile-shape-sides", default="",
                        help="latent tile edges to time in --tile-shape-costs, comma separated, "
                             "in place of the ones this VAE's window clips to. Square, since that "
                             "is what a narrowed window gives")
    parser.add_argument("--phase-timing", action="store_true",
                        help="split a tiled decode into the decoder calls and everything else, "
                             "which is where the blending lives. Diagnostic only: it synchronises "
                             "the device around every tile, so the latency it reports is not the "
                             "latency the arm has without it")
    parser.add_argument("--frames", type=int, default=17,
                        help="frames, for the VAEs that have a frame axis; ignored by the rest")
    parser.add_argument("--max-rel", type=float, default=None,
                        help="agreement tolerance, as a fraction of the reference's largest "
                             "value; defaults by dtype")
    parser.add_argument("--skip-reference", action="store_true",
                        help="skip the single-rank comparison, which needs the whole half to fit on one GPU")
    parser.add_argument("--reference-max-latent-elems", type=int, default=16384,
                        help="above this latent area the reference is skipped on its own: an unsharded "
                             "decode at that size is the thing sharding exists to avoid. Raise it "
                             "deliberately when the error against an untiled, unsharded decode is the "
                             "measurement you came for, and the unsharded decode still fits on one GPU")
    parser.add_argument("--describe-only", action="store_true",
                        help="report the blocks and the adapter xDiT picks, then stop")
    parser.add_argument("--timeout-min", type=int, default=30,
                        help="process group timeout; the first decode on a new shape pays MIOpen autotune")
    parser.add_argument("--out", default=None, help="write the report here as JSON")
    args = parser.parse_args()
    if args.grid_arms and not args.grid_shapes:
        args.grid_shapes = f"{args.height}x{args.width}x{args.frames}"

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
    say(f"family={args.family} half={args.half} dtype={args.dtype} "
        f"shapes={args.grid_shapes or f'{args.height}x{args.width}'} "
        f"arms={args.grid_arms or 'single'}")

    cells = grid_cells(args)

    # Before the VAE exists, because importing xfuser swaps torch.nn.GroupNorm for AITER's, and
    # both the adapters and xDiT's selection ask isinstance(norm, nn.GroupNorm). A VAE built
    # first holds the class from before the swap and matches nothing. Real runs import xfuser
    # long before they load a model, so this is the ordering being measured.
    if all(cell.get("as_main_does") for cell in cells):
        # A grid of nothing but main's arms has to run on an xDiT with no selection to ask, which
        # is the point of it. The environment is still the one being measured, so xfuser is
        # imported as a run imports it and the swap is then undone exactly where main undoes it,
        # in _validate_config, whenever parallel VAE is on.
        try:
            import xfuser  # noqa: F401
        except ImportError:
            pass
        _restore_torch_groupnorm()
    else:
        _vae_parallel()

    spec = FAMILIES[args.family]

    if args.tile_shape_costs:
        costs = tile_shape_costs(args, spec, device, dtype, say)
        if rank == 0:
            write_report(args.out, world_size, local_rank, {"tile_shape_costs": costs})
        dist.barrier()
        dist.destroy_process_group()
        return

    references = {}
    reports = []

    for index, cell in enumerate(cells):
        if len(cells) > 1:
            say(f"\n===== cell {index + 1}/{len(cells)}: {cell['name']} "
                f"{cell['height']}x{cell['width']}"
                f"{'x' + str(cell['frames']) + 'f' if spec['temporal'] else ''} =====")
        # One failed cell costs that cell. Ranks agree on the verdict before anyone moves on,
        # because a rank that carried on into the next cell's collectives while the others were
        # unwinding an exception would hang the pod rather than lose a row.
        try:
            report = measure_cell(
                args, spec, cell, device, dtype, group, world_size, rank, say, references
            )
            failed = None
        # SystemExit alongside Exception, because a refusal deep in the harness is raised that
        # way and it does not derive from Exception: unhandled, one rank would unwind out of the
        # loop while the others waited in the next cell's collectives, and the pod would hang
        # until its timeout rather than lose the one row.
        except (Exception, SystemExit) as error:  # noqa: BLE001 - keeping the grid going is the point
            report, failed = None, f"{type(error).__name__}: {error}"
            # From whichever rank raised, not only from rank 0. A cell that fails on some ranks
            # and not others is the case most worth seeing and the one `say` hides, and it is
            # also the case the vote below cannot rescue: a rank still inside the decode is in
            # that decode's collectives, not in this all_reduce, so the two sit until the
            # watchdog fires and the only evidence left is a timeout naming two different
            # collectives. Printed before the vote so it survives the deadlock.
            print(f"[rank {rank}] cell failed: {failed}", flush=True)
        torch.cuda.empty_cache()
        votes = torch.tensor([0.0 if failed else 1.0], device=device)
        dist.all_reduce(votes)
        if votes.item() < world_size:
            reports.append({**cell, "error": failed or "another rank failed this cell"})
            continue
        reports.append(report)
        if rank == 0:
            print_report(report, args.half)

    if rank == 0:
        # Always a list, even for one cell. A reader that has to find out whether it is holding a
        # cell or a grid before it can start is a reader everyone writes slightly differently.
        write_report(args.out, world_size, local_rank, {"cells": reports})

    dist.barrier()
    dist.destroy_process_group()
    # A grid is a measurement, not a gate: it is expected to contain arms that disagree with the
    # reference, so only a single run answers with its exit code.
    if len(reports) == 1:
        agreement = (reports[0] or {}).get("agreement")
        if reports[0] is None or (agreement is not None and not agreement["ok"]):
            raise SystemExit(1)


def grid_cells(args) -> list:
    """The arms and shapes to measure, one dict each; a plain run is a grid of one"""
    tiling = "native" if args.enable_tiling else None
    if args.vae_tile_size is not None:
        tiling = args.vae_tile_size
    single = {
        "name": "single",
        "parallel_vae": not args.no_parallel_vae,
        "tiling": tiling,
        "height": args.height,
        "width": args.width,
        "frames": args.frames,
        # A single run has one cell to put an overlap in, so it takes the first of a list.
        "overlap": overlap_token(args.tile_overlap.split(",")[0]) if args.tile_overlap else None,
    }
    if not args.grid_arms:
        return [single]

    arms = {
        # The four arms in the order they are read: each is the one above it plus one thing.
        "none": {"parallel_vae": False, "tiling": None},
        "pvae": {"parallel_vae": True, "tiling": None},
        "tile": {"parallel_vae": True, "tiling": "native"},
        "tile-half": {"parallel_vae": True, "tiling": "half"},
        "tile-quarter": {"parallel_vae": True, "tiling": "quarter"},
        # Tiling with nothing to amortise, which separates the collective saving from the plain
        # effect of handing the GPU smaller convolutions.
        "tile-nopvae": {"parallel_vae": False, "tiling": "native"},
        # What xDiT main and DistVAE main already do with these two flags on, which is the number
        # every arm above has to beat to be worth shipping. Needs -DistVaeBranch main to mean it:
        # run against the branch's library it measures main's WIRING over new adapters, which is
        # a different claim.
        "main": {"parallel_vae": True, "tiling": "native", "as_main_does": True},
        "main-notile": {"parallel_vae": True, "tiling": None, "as_main_does": True},
    }
    shapes = []
    for text in args.grid_shapes.split(","):
        parts = text.strip().lower().split("x")
        if len(parts) not in (2, 3):
            raise SystemExit(f"--grid-shapes takes HxW or HxWxFRAMES, not {text!r}")
        shapes.append(
            {
                "height": int(parts[0]),
                "width": int(parts[1]),
                "frames": int(parts[2]) if len(parts) == 3 else args.frames,
            }
        )

    # None is the VAE's own overlap, which is the arm as it was before this was a knob, so it
    # stays first and every other overlap is read against it.
    overlaps = [None]
    if args.tile_overlap:
        overlaps += [overlap_token(text) for text in args.tile_overlap.split(",")]

    cells = []
    for shape in shapes:
        for name in args.grid_arms.split(","):
            name = name.strip()
            if name not in arms:
                raise SystemExit(f"unknown arm {name!r}; pick from {sorted(arms)}")
            for overlap in overlaps:
                # Only a tiled arm has tiles to overlap, and main's arms are what main does with
                # no window and no overlap to choose, so both are measured once and left alone.
                if overlap is not None and (
                    not arms[name].get("tiling") or arms[name].get("as_main_does")
                ):
                    continue
                cells.append(
                    {
                        "name": name if overlap is None else f"{name}-ov{overlap_label(overlap)}",
                        **arms[name],
                        **shape,
                        "overlap": overlap,
                    }
                )
    return cells


def measure_cell(args, spec, cell, device, dtype, group, world_size, rank, say, references):
    """Build, optionally shard, optionally tile, and measure one arm at one shape

    The VAE is rebuilt per cell rather than reused: sharding and the batched decode both replace
    parts of it in place, and unpicking that reliably is harder than paying for a fresh one from
    a fixed seed. What is reused is the reference, which depends only on the shape - and which is
    the expensive part, being an unsharded decode of the whole thing.
    """
    vae = build_vae(args.family, dtype, device)
    sample = sample_for(
        spec, args.half, cell["height"], cell["width"], dtype, device, args.batch, cell["frames"]
    )
    say(f"{'latent' if args.half == 'decoder' else 'input'} {tuple(sample.shape)}")

    as_main_does = bool(cell.get("as_main_does"))
    built = describe(vae, args.half, args.family, select=not as_main_does)
    say(f"{args.half}: {json.dumps(built)}")
    if built["adapter"] is None and cell["parallel_vae"]:
        raise ValueError(
            f"{'xDiT main names' if as_main_does else 'xDiT has'} no adapter for this "
            f"{type(vae).__name__} {args.half}. Nothing to measure."
        )
    if args.describe_only:
        return None

    # The reference has to be taken before sharding, which replaces the half in place. Every rank
    # computes it rather than rank 0 alone: the seeds match, so the weights match, and leaving it
    # to one rank would strand the others in the next collective for as long as it takes.
    # In latent space for both halves, so the one threshold means the same thing either way.
    latent_area = sample.shape[0] * sample.shape[-2] * sample.shape[-1]
    if sample.ndim == 5:
        latent_area *= sample.shape[2]
    if args.half == "encoder":
        latent_area //= spec["spatial"] ** 2
    key = (cell["height"], cell["width"], cell["frames"])
    take_reference = not args.skip_reference and latent_area <= args.reference_max_latent_elems
    if not args.skip_reference and not take_reference and key not in references:
        say(f"no single-rank reference: a {sample.shape[-2]}x{sample.shape[-1]} latent is over "
            f"--reference-max-latent-elems {args.reference_max_latent_elems}, and an unsharded "
            f"decode that size is what sharding exists to avoid. Check agreement at a smaller one.")
    if take_reference and key not in references:
        with torch.no_grad():
            references[key] = run_half(vae, args.half, sample).float().cpu()
    reference = references.get(key)

    # Three ways for a group to divide a tiled decode, and the runner picks the first one wherever
    # the tiling loop is one xDiT owns: a band of tile rows to a rank, leaving the decoder
    # unsharded. --tile-split scattered deals whole tiles without the bands, and rows shards
    # inside every tile.
    tile_parallel = bool(
        cell["parallel_vae"]
        and cell["tiling"]
        and args.half == "decoder"
        and args.tile_split in ("tiles", "scattered")
        and not as_main_does
        and deals_tiles_out(vae)
    )

    adapter = None
    if not cell["parallel_vae"]:
        say("parallel VAE off: every rank decodes the whole half, as an unsharded run does")
    elif as_main_does:
        adapter = parallelize_as_main_does(vae, group, args.family, args.half)
        say(f"adapter={adapter}")
    elif tile_parallel:
        say(f"parallel VAE by whole tiles ({args.tile_split}): the decoder is left unsharded and "
            f"each rank decodes the tiles it is given")
    else:
        adapter = parallelize(vae, group, args.half)
        say(f"adapter={adapter}")

    # After sharding, which is the order the runner uses: _setup_parallel_vae runs during load and
    # _enable_options after it, so the batched decode is installed over an already-sharded decoder.
    tiling = {"enabled": False}
    if cell["tiling"] and as_main_does:
        tiling = tile_as_main_does(vae)
        say(f"tiling: {json.dumps(tiling)}")
    elif cell["tiling"]:
        if args.half != "decoder":
            raise ValueError("tiling is a decode-side feature; --enable-tiling needs --half decoder")
        window = None if cell["tiling"] == "native" else cell["tiling"]
        tiling = setup_tiling(
            vae, window, world_size, say,
            group=group if tile_parallel else None,
            phase_timing=args.phase_timing,
            tile_split=args.tile_split,
            overlap=cell.get("overlap"),
        )
        say(f"tiling: {json.dumps(tiling)}")

    def once():
        with torch.no_grad():
            return run_half(vae, args.half, sample)

    for _ in range(args.warmup):
        once()
    torch.cuda.synchronize(device)

    # Counted over one call, so the numbers read per decode rather than per run.
    LOG.reset()
    LOG.enabled = True
    output = once()
    LOG.enabled = False
    collectives = LOG.report()
    collectives.update(across_ranks(LOG.by_call, world_size))

    PHASES.clear()
    torch.cuda.reset_peak_memory_stats(device)
    timing = timed(once, args.iters, device)
    peak_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    phases = phase_report(group, world_size)
    if phases:
        say(f"phases: {json.dumps(phases)}")

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
            # A max is one element and says nothing about how much of the output moved, which is
            # the question an arm that tiles raises: tiling is not a rounding difference, it
            # normalises each tile over less context, so it shifts whole regions a little rather
            # than one element a lot. The share off by more than a hundredth of scale is the
            # harness's read of the same thing the sweeps measure as "pixels more than 10% off".
            off = (diff > 0.01 * scale).float().mean().item() if scale else 0.0
            agreement = {
                "ok": bool(relative <= tolerance),
                "max_abs": diff.max().item(),
                "mean_abs": diff.mean().item(),
                "reference_max_abs": scale,
                "max_rel_to_scale": relative,
                "mean_rel_to_scale": diff.mean().item() / scale if scale else 0.0,
                "share_off_by_1pc": off,
                "max_rel_allowed": tolerance,
            }
            # Sharding has to be numerically invisible and the tolerance is how we hold it to
            # that. Tiling does not: it is a different computation, normalising each tile over
            # less context, and the whole reason to measure it here is to put a number on how
            # different. Failing the run for that would be failing it for working as designed.
            if tiling.get("enabled"):
                agreement["ok"] = True
                agreement["measured_not_enforced"] = (
                    "tiling changes the arithmetic; this is the size of that change, not a gate"
                )

    import diffusers
    import distvae

    return {
        "arm": cell["name"],
        "family": args.family,
        "half": args.half,
        "height": cell["height"],
        "width": cell["width"],
        "frames": cell["frames"] if spec["temporal"] else None,
        "dtype": args.dtype,
        "world_size": world_size,
        "parallel_vae": cell["parallel_vae"],
        "adapter": adapter,
        "tiling": tiling,
        "latent_shape": list(sample.shape),
        "collectives": collectives,
        "timing": timing,
        "phases": phases or None,
        "peak_vram_mb": peak_mb,
        "agreement": agreement,
        "versions": {
            "torch": torch.__version__,
            "diffusers": diffusers.__version__,
            "distvae": getattr(distvae, "__version__", "unknown"),
        },
    }


def print_report(report: dict, half: str) -> None:
    """One cell's numbers, in the shape the collector reads them back out of"""
    collectives, timing = report["collectives"], report["timing"]
    print(f"\n--- collectives per {half} call (rank 0, and the most any rank made) ---", flush=True)
    for name, entry in collectives["by_call"].items():
        print(f"  {name:<24} {entry['calls']:>6} calls  "
              f"{collectives['by_call_max'][name]:>6} max  "
              f"{entry['bytes'] / 1e6:>10.2f} MB", flush=True)
    print(f"  {'TOTAL':<24} {collectives['total_calls']:>6} calls  "
          f"{collectives['total_calls_max']:>6} max  "
          f"{collectives['total_bytes'] / 1e6:>10.2f} MB", flush=True)
    print(f"  by rank: {collectives['total_calls_by_rank']}", flush=True)
    print("\n--- top call sites ---", flush=True)
    for site, entry in list(collectives["by_site"].items())[:12]:
        print(f"  {entry['calls']:>6}  {site}", flush=True)
    print(f"\nmedian {timing['median_s'] * 1000:.1f} ms   peak {report['peak_vram_mb']:.0f} MB",
          flush=True)
    phases = report.get("phases")
    if phases:
        print(f"phases: decoder {phases['decoder_ms']:.1f} ms   rest {phases['rest_ms']:.1f} ms"
              f"   of {phases['total_ms']:.1f} ms"
              f"   over {phases['calls_per_decode']:.0f} calls", flush=True)
        if "decoder_ms_by_rank" in phases:
            print(f"  decoder by rank {phases['decoder_ms_by_rank']}   "
                  f"calls by rank {phases['calls_by_rank']}   "
                  f"idle {phases['idle_share'] * 100:.1f}%", flush=True)
    agreement = report.get("agreement")
    if agreement is not None:
        verdict = "matches" if agreement["ok"] else "DIFFERS FROM"
        print(f"output {verdict} the single-rank reference: {agreement}", flush=True)
        print(f"error vs untiled unsharded: "
              f"max {agreement.get('max_rel_to_scale', 0) * 100:.2f}%  "
              f"mean {agreement.get('mean_rel_to_scale', 0) * 100:.3f}%  "
              f"share off by >1% {agreement.get('share_off_by_1pc', 0) * 100:.2f}%", flush=True)


if __name__ == "__main__":
    main()
