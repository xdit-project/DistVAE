# Benching DistVAE on a machine you have not benched before

`distvae_bench.py` measures the sharded VAE halves at real shapes without downloading a
checkpoint. It builds the true architecture from a config with random weights, because what we
tune here is a property of the adapter stack rather than of the weights: `PatchGroupNorm` issues
the same collectives whether its input came from Flux.2 or from `torch.randn`.

The launcher and `harness/` package take no cluster, and write JSON that says what produced it.
Copy the `bench` package to the box, run it, and send back the JSON.

## What the machine needs

| | Why |
|---|---|
| PyTorch with a working `torch.distributed` | ROCm and CUDA builds both work unchanged: torch presents HIP under `torch.cuda` and RCCL under the `nccl` backend, so nothing here branches on vendor |
| `diffusers` | the VAE architectures are read from its classes |
| DistVAE, installed | the thing under test |
Install DistVAE from the branch you mean to compare, not from a release. Two machines
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
--tile-overlap HxW     exact output-pixel overlap between tiles (for example 64x32)
```

`--grid-arms` runs several in one job against one reference, which is both faster and more
comparable than several jobs. Canonical presets are `unsharded`, `row`, `row-tiled`,
`row-tiled-half`, `row-tiled-quarter`, `tiled`, `tile-runs`, `tile-runs-half`, and
`tile-runs-quarter`. Existing names such as `none`, `pvae`, `tile`, and `tile-dist` remain
accepted as compatibility aliases. `--grid-shapes` takes comma-separated `HxW` or
`HxWxFRAMES` values. Explicit composition flags cannot be mixed with `--grid-arms`;
`--tile-overlap` remains an orthogonal grid axis. A grid takes comma-separated pixel pairs,
for example `--tile-overlap 64x32,32x16,0x0`.

```bash
torchrun --nproc_per_node=4 bench/distvae_bench.py \
  --family wan --half decoder \
  --grid-arms none,pvae,tile,tile-half \
  --grid-shapes 720x1280x81,1080x1920x81 \
  --out wan-decoder-grid.json
```

`--tile-shape-costs` is a separate decoder-only mode. It ignores ordinary composition axes and
measures the decoder across tile shapes selected by `--tile-shape-sides` and batch sizes up to
`--tile-shape-batch`.

`--profile`, `--profile-trace`, and `--profile-memory` run one additional call after timed
measurement. Requested artifacts are written under `--profile-dir`; repeated cells receive a
numeric suffix rather than replacing an existing artifact.

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

The JSON is one schema-versioned record for a single cell and a list of records for a grid. Each
record retains its own versions and provenance so it remains self-contained when separated from
the grid. Provenance is collected once per invocation and reused across those records. Schema 5
records tile windows and overlaps as two-axis values: `native_window_px`, `window_px`,
`native_overlap_px`, `overlap`, and the shape-cost `latent_window` are all `[height, width]`
in JSON.

It also carries one digest over the launcher and harness implementation, which is not the same
claim as the commit of the installed DistVAE. The bench package can travel by other means than the
library, so the commit beside it is no evidence of what actually ran. When two machines disagree,
check the digests match before reading anything into the numbers.

## What it cannot tell you

Nothing about real activation distributions — random weights give mean about 0 and variance about
1, the easy case for any variance computation. Nothing about the pipeline around the VAE, and
nothing about host RAM. The peak VRAM here is the VAE's own, which is the point of measuring it
apart, but it is **not** a run's peak: a window that halves the decode's memory moves a run's peak
only while the VAE is the thing that peaks. Those questions need a real model.
