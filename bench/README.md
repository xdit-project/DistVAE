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

The report records package versions, the DistVAE revision, a source digest, and accelerator
details under `provenance.device`. Compare results only when this context is available.

`HW_FAMILY` adds your own label alongside it, for naming a fleet or a node type:

```bash
HW_FAMILY=mi355 torchrun --nproc_per_node=8 bench/distvae_bench.py ...
```

The label is null when the variable is absent; the measured device is recorded either way.

## Reproducing a run elsewhere

`--matrix` runs the family's canonical shapes rather than one:

```bash
torchrun --nproc_per_node=4 bench/distvae_bench.py \
  --family wan --half decoder --matrix --out wan-decoder.json
```

| family | shapes (height x width x frames) |
| --- | --- |
| `flux2` | 1024x1024, 2048x2048 |
| `kl` | 1024x1024, 2048x2048 |
| `qwen_image` | 1024x1024x1, 2048x2048x1 |
| `wan` | 832x480x81, 1280x720x81 |
| `hunyuan_video` | 832x480x129, 1280x720x129 |
| `hunyuan_video_15` | 832x480x129, 1280x720x129 |
| `ltx2` | 1536x1024x121, 1920x1280x121 |

Canonical shapes are versioned with their architectures in `harness/catalog.py`. `--shape`
overrides the matrix for a one-off run. Video families use their normal frame counts because their
temporal compression ratios differ. Qwen-Image uses one frame.

LTX-2 uses larger spatial shapes because its 32× compression must still leave at least sixteen
latent units on a tile's narrow axis. Its 1920x1280 case also provides enough tiles for eight
ranks. Use `--shape` to test a different resolution.

Large unsharded video cases may exceed device memory. The failure is recorded for that case and
the remaining cases continue.

## Default suite

This command runs the default decoder suite for one 2048×2048 input:

```bash
torchrun --nproc_per_node=4 bench/distvae_bench.py \
  --family flux2 --half decoder --height 2048 --width 2048 \
  --out flux2-decoder-2048.json
```

The suite runs the modes available to an application:

1. unsharded, untiled
2. row sharded, untiled
3. whole-tile distribution at each selected plan

Five cases where the sample supports three plans, four where it supports two.

The plans are named `coarse`, `balanced`, and `fine`, from fewest tiles to most. The names describe
geometry; benchmark results determine which is fastest on a device. `coarse` usually has fewer
seams and less repeated work. `fine` uses less memory per tile. The report's
`beats_row_sharding` field shows whether a plan's window area is smaller than one row-sharded
rank's activation area.

`--diagnostics` adds local tiling for each plan and row sharding inside the finest tile plan.
Applications do not normally use these combinations, and they add substantial runtime. Use them
to separate tile overhead from communication overhead. The local case has no collectives and
shows the minimum measured memory for that window.

Tiling is decode-only. `--half encoder` runs the two untiled baselines.

The planner considers grids with up to four tiles per rank and overlaps down to one quarter of the
window. DistVAE validates each rectangular window and absolute overlap. The planner removes
candidates that are worse in window area, decoded area, rank imbalance, and tile columns. It then
selects the coarsest plan, the finest plan, and a balanced plan between them. An untiled axis uses
zero overlap. The JSON records the objectives, candidate limit, and Pareto frontier size.

Three constraints limit the search:

- **Overlap is searched, not pinned.** A tile is a memory win over row sharding only when its
  window area is under the `(height / ranks) * width` a rank already holds. Since window is
  pitch plus overlap, pinning overlap at the VAE native value floors every window at that value
  and, on a 1024x1024 sample at four ranks, made the whole suite memory-neutral by construction.
- **A blend is at least a quarter of its window.** Overlap decides whether a tile's tone drift
  from its neighbours reads as a gradient or a band. At 128x1024 on FLUX.2, a 32px blend is
  clean and a 16px blend bands.
- **A tile is at least sixteen latent on its narrow axis.** Below that a tile normalizes over
  content too unrepresentative of the image, and no blend repairs it.

Plan selection uses geometry only. Matching family, shape, and world size therefore produce the
same plans on different machines.

Use `--shape` to request more input shapes explicitly:

```bash
torchrun --nproc_per_node=4 bench/distvae_bench.py \
  --family wan --half decoder \
  --shape 720x1280x81 --shape 1080x1920x81 \
  --out wan-decoder.json
```

Each requested shape gets its own suite. Add only shapes needed for a specific comparison because
VAE runs are expensive.

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

Exact cases and `--shape` cannot be combined. Run a second command to change both the input and
execution mode.

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

`--out` writes schema 7 JSON. One exact case is an object; a suite is an array. Stdout contains
progress and compact human-readable summaries, not a recoverable copy of the JSON. Always supply
`--out` when collecting results from another machine.

Every record includes versions, provenance, world size, dtype, execution mode, effective tile
settings, latency, peak accelerator memory, collective counts, and agreement with an unsharded
reference when the reference-size limit permits one.
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
- **case:** one input shape and execution mode measured as a record
- **coverage:** decoded tile area divided by image area; overlap raises it above one
- **halo:** neighboring rows exchanged so a sharded convolution has its required context
- **overlap:** output pixels shared and blended between adjacent tiles
- **patchify:** split an activation into rank-local row bands
- **window:** output-pixel height and width decoded by one spatial tile
