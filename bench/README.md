# Benching DistVAE on a machine you have not benched before

`distvae_bench.py` measures the sharded VAE halves at real shapes without downloading a
checkpoint. It builds the true architecture from a config with random weights, because what we
tune here is a property of the adapter stack rather than of the weights: `PatchGroupNorm` issues
the same collectives whether its input came from Flux.2 or from `torch.randn`.

It is one file, it takes no cluster, and it writes one JSON that says what produced it. That is
the whole portability story — copy it to the box, run it, send back the JSON.

## What the machine needs

| | Why |
|---|---|
| PyTorch with a working `torch.distributed` | ROCm and CUDA builds both work unchanged: torch presents HIP under `torch.cuda` and RCCL under the `nccl` backend, so nothing here branches on vendor |
| `diffusers` | the VAE architectures are read from its classes |
| DistVAE, installed | the thing under test |
| xDiT (`xfuser`), installed | see below — needed for every arm except `main` |

**On xDiT.** The DistVAE library imports nothing from xDiT and never will. The *bench* does, on
purpose: xDiT is what chooses which adapter fits a VAE and what order the tiling calls happen in,
and those choices are part of what is being measured. Letting this file pick an adapter instead
would measure this file's opinion, and a run would sail on with the wrong one rather than tell you
the installed xDiT is too old. The one exception is the `main` arm, which names its adapter
directly and so runs with no xDiT present at all — at the cost of covering only the decoder of
`flux2`, `kl` and `wan`.

Install DistVAE and xDiT from the branches you mean to compare, not from a release. Two machines
can both hold `distvae 0.0.0b5` and disagree about everything that matters; the report records the
branch and commit of each so this is at least visible afterwards.

## Running it

```bash
torchrun --nproc_per_node=4 bench/distvae_bench.py \
  --family flux2 --half decoder --height 2048 --width 2048 \
  --out flux2-decoder-2048.json
```

`--out` is optional. The report is printed to stdout regardless, between
`===== BEGIN DISTVAE REPORT =====` and `===== END DISTVAE REPORT =====`, because the filesystem
it was written to is often the thing that does not survive — a container that is discarded when it
exits, a box you only have a terminal on. **Capturing the log is enough**; the report can be cut
out of it afterwards, and nothing else needs to come back.

Set `HW_FAMILY` to whatever you want this machine called in the results. It is not looked up in a
table of known devices — nobody should have to edit a list to add hardware — and if you leave it
unset the architecture string (`gfx1201`, `sm_90`) stands in, which is correct but harder to read.

```bash
HW_FAMILY=mi355 torchrun --nproc_per_node=8 bench/distvae_bench.py ...
```

### Families

`flux2`, `kl`, `wan`, `qwen_image`, `hunyuan_video`, `hunyuan_video_15`, `ltx2`. The video
families take `--frames`. Either half runs: `--half decoder` or `--half encoder`.

### Arms

A single run is one arm, chosen by flags, each differing from the one above by one thing:

```
--no-parallel-vae      unsharded, untiled: the baseline
(default)              sharded
--enable-tiling        sharded and tiled at the VAE's own window
--vae-tile-size N      the same, at a narrower window
--tile-overlap F       the same, at a wider stride between tiles
```

`--grid-arms` runs several in one job against one reference, which is both faster and more
comparable than several jobs. Named arms are `none`, `pvae`, `tile`, `tile-half`, `tile-quarter`,
`tile-nopvae`, `main`, `main-notile`. `--grid-shapes` takes `HxW` or `HxWxFRAMES`, comma
separated.

```bash
torchrun --nproc_per_node=4 bench/distvae_bench.py \
  --family wan --half decoder \
  --grid-arms none,pvae,tile,tile-half \
  --grid-shapes 720x1280x81,1080x1920x81 \
  --out wan-decoder-grid.json
```

**Run the same arms and shapes on every machine.** Nothing enforces it, and a table assembled
from runs that each picked their own shapes compares nothing.

## What comes back

Three things per cell, and the first is the point of the harness:

- **collectives** — exact counts and bytes, by call site. An optimisation that removes an
  `all_reduce` shows up as an integer, not as a timing delta the size of the noise on a consumer
  GPU. This is the number that is worth carrying between machines, because it is the one that does
  not depend on the machine.
- **latency** — wall time per decode after warmup.
- **agreement** — the sharded output against a single-rank reference, which is the invariant every
  change has to preserve. A run of a single cell exits non-zero if it disagrees; a grid does not,
  because a grid is expected to contain arms that disagree and is a measurement rather than a gate.

The JSON is `{"schema": 1, "ran": {...}, "cells": [...]}`. The `ran` block carries the hardware,
the world size, the branch and commit of everything installed, and the exact argv, so a file that
arrives by scp needs no accompanying message to be read. Reports from before this envelope existed
are a bare cell or a bare list, with no `schema` key.

It also carries a digest of this script itself, which is not the same claim as the commit of the
installed DistVAE. The bench file travels by other means than the package does — copied to a box,
mounted into a container, delivered by ConfigMap — so the commit beside it is no evidence of what
actually ran. When two machines disagree, check the digests match before reading anything into the
numbers.

## What it cannot tell you

Nothing about real activation distributions — random weights give mean about 0 and variance about
1, the easy case for any variance computation. Nothing about the pipeline around the VAE, and
nothing about host RAM. The peak VRAM here is the VAE's own, which is the point of measuring it
apart, but it is **not** a run's peak: a window that halves the decode's memory moves a run's peak
only while the VAE is the thing that peaks. Those questions need a real model.
