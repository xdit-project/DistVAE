# Benchmarking DistVAE

`distvae_bench.py` measures a real diffusers VAE architecture without downloading a
checkpoint. Every cell rebuilds the architecture with seed 0 and creates its input with seed 1.
The weights are synthetic; layer shapes, memory use, collectives, and scheduling are real.

Copy `bench/` to the target machine, install the DistVAE revision under test, and run the launcher
with `torchrun`.

## Requirements

- PyTorch with a working `torch.distributed` CUDA or ROCm build
- `diffusers`
- DistVAE installed from the revision being measured

The report records package versions, the DistVAE checkout revision when available, and a digest
of the benchmark sources. Set `HW_FAMILY` to add your own hardware label:

```bash
HW_FAMILY=mi355 torchrun --nproc_per_node=8 bench/distvae_bench.py ...
```

No hardware label is inferred when the variable is absent.

## The bounded suite

This command runs the default decoder suite for one 2048×2048 input:

```bash
torchrun --nproc_per_node=4 bench/distvae_bench.py \
  --family flux2 --half decoder --height 2048 --width 2048 \
  --out flux2-decoder-2048.json
```

The suite carries only the compositions a caller can select:

1. unsharded, untiled
2. row sharded, untiled
3. whole-tile distribution at each selected plan

Five cases where the sample supports three plans, four where it supports two.

The plans are named `coarse`, `balanced` and `fine`, for fewest tiles through most. They name
geometry rather than an outcome, because an outcome is a claim about a device: the profiles were
once called throughput and memory, and throughput scored plans by least total work, which always
chose the widest window since a wide tile overlaps its neighbours fewer times. On gfx1201 those
arms measured both the slowest and heavier than plain row sharding - 5034 MB against row's 3526
at 2048x2048 on four ranks - so the label asserted the reverse of what the hardware did.

The planner therefore brackets the axis instead of predicting a winner on it. Both ends are
pinned by bounds that hold anywhere: the banding floor at the fine end, and nothing left to
divide at the coarse end. Which end wins in between is what the bench is for, and it is allowed
to differ per device. Since fewest tiles also means fewest seams, prefer `coarse` where the
memory allows it and reach for `fine` when it does not - `beats_row_sharding` in the report says
whether a plan is a memory win at all. The fine end has to be strictly finer than the coarse one
to earn its cases; on a square sample the runner-up is otherwise a transpose, scoring identically
on every objective and measuring materially heavier.

`--diagnostics` adds local tiling at each plan and row sharding beneath the lightest plan. An
orchestrator reaches neither - xFuser branches straight between marking a VAE for tile
parallelism and parallelizing its decoder, with nothing in between - and together they are about
60% of the suite's compute. They are worth their cost when characterising a new geometry rather
than comparing plans: `local` is the only case with no collectives at all, so it separates what
tiling does to the decode from what the collectives cost, and its peak is the true floor for a
window.

Tiling is decode-only. `--half encoder` runs the two untiled baselines.

The planner enumerates tile grids up to four tiles per rank and, at each grid, a ladder of
overlaps down to a quarter of the window. It validates every rectangular window and absolute
overlap through DistVAE and removes candidates dominated on window area, decoded area, rank
imbalance, and tile columns. It then chooses the least-work plan, a frontier knee, and the
smallest-window plan, each with a distinct window. An inactive strip axis receives zero overlap.
The JSON records every objective, the frontier size, and the candidate limit.

Three bounds shape which plans are reachable, and each is a measurement rather than a margin:

- **Overlap is searched, not pinned.** A tile is a memory win over row sharding only when its
  window area is under the `(height / ranks) * width` a rank already holds. Since window is
  pitch plus overlap, pinning overlap at the VAE native value floors every window at that value
  and, on a 1024x1024 sample at four ranks, made the whole suite memory-neutral by construction.
- **A blend is at least a quarter of its window.** Overlap decides whether a tile's tone drift
  from its neighbours reads as a gradient or a band. At 128x1024 on FLUX.2, a 32px blend is
  clean and a 16px blend bands.
- **A tile is at least sixteen latent on its narrow axis.** Below that a tile normalizes over
  content too unrepresentative of the image, and no blend repairs it.

Selection uses topology only. Hardware timings never feed back into the plans, so machines run
the same suite when family, shape, and world size match.

Use `--shape` to request more input shapes explicitly:

```bash
torchrun --nproc_per_node=4 bench/distvae_bench.py \
  --family wan --half decoder \
  --shape 720x1280x81 --shape 1080x1920x81 \
  --out wan-decoder.json
```

Each requested shape gets its own bounded suite. Avoid adding shapes without a comparison
question; VAE runs are expensive.

## Exact cases

Repeat `--case` to bypass automatic selection. Baselines are `unsharded` and `row`. A tiled case
uses `MODE:WINDOW_HxW@OVERLAP_HxW`, where `MODE` is `local`, `tile-runs`, or `row-tiled`.
Window and overlap values are output pixels.

```bash
torchrun --nproc_per_node=4 bench/distvae_bench.py \
  --family kl --half decoder --height 1024 --width 1536 \
  --case unsharded \
  --case row \
  --case 'local:480x736@64x32' \
  --case 'tile-runs:480x736@64x32' \
  --out kl-exact.json
```

Exact cases and `--shape` cannot be combined. Run a second invocation when both the input and
the composition must change.

## Shape-cost mode and profiling

`--tile-shape-costs` is decoder-only and separate from the ordinary suite. By default it measures
the three selected rectangular plans. Override them with latent-space windows:

```bash
torchrun --nproc_per_node=4 bench/distvae_bench.py \
  --family kl --half decoder --height 2048 --width 2048 \
  --tile-shape-costs --tile-shape-windows 88x144,88x88,56x88 \
  --tile-shape-batch 4 --out kl-shape-costs.json
```

`--profile`, `--profile-trace`, and `--profile-memory` add one profiler call after timed
measurement. Artifacts go under `--profile-dir`; repeated names receive numeric suffixes.

## Output and exit status

`--out` writes schema 6 JSON. One exact case is an object; a suite is an array. Stdout contains
progress and compact human-readable summaries, not a recoverable copy of the JSON. Always supply
`--out` when collecting results from another machine.

Every record is self-contained. It includes versions, provenance, runtime world size and dtype,
the requested composition, effective tile facts, latency, peak accelerator memory, collective
counts, and agreement with an unsharded reference when the reference-size limit permits one.
Windows and overlaps are `[height, width]`.

The process exits nonzero for setup or execution errors and for enforced agreement failures.
Row-sharded numerical agreement is enforced. Numerical differences caused by tiling are measured
and reported but do not control the exit status. Structural failures still fail every mode.

Run identical family, shape, world-size, dtype, and benchmark digests before comparing machines.

## Limits

Synthetic weights do not model activation distributions from a trained checkpoint. The harness
does not measure the diffusion pipeline, host memory, image quality, or visual seam quality.
Peak memory covers the selected VAE half. Use a real model run for end-to-end peak memory and
quality decisions.

## Glossary

- **adapter:** DistVAE wrapper that gives a diffusers encoder or decoder distributed behavior
- **case:** one input shape and execution composition measured as a record
- **coverage:** decoded tile area divided by image area; overlap raises it above one
- **halo:** neighboring rows exchanged so a sharded convolution has its required context
- **overlap:** output pixels shared and blended between adjacent tiles
- **patchify:** split an activation into rank-local row bands
- **window:** output-pixel height and width decoded by one spatial tile
