# DistVAE

DistVAE replaces supported diffusers VAE encoders and decoders with distributed adapters. The rest
of the diffusion pipeline stays unchanged.

## Installation

```bash
pip install distvae
```

Python 3.10 or newer, with `torch>=2.2` and `diffusers>=0.30.3`. Individual VAE families may require
a newer Diffusers release.

The pipeline quickstart also needs Transformers:

```bash
pip install "distvae[pipeline]"
```

## Quickstart

Every rank builds the same pipeline, and DistVAE shards the VAE inside it. Save this as `decode.py`:

```python
import os

import torch
import torch.distributed as dist
from diffusers import DiffusionPipeline
from distvae import vae as vae_api

dist.init_process_group(backend="nccl")
device = torch.device(f"cuda:{os.environ['LOCAL_RANK']}")
torch.cuda.set_device(device)

# The group the VAE is split over. Every rank that enters the VAE call must be a
# member. If you create a subgroup, gate the pipeline call to those ranks too.
vae_group = dist.group.WORLD

pipe = DiffusionPipeline.from_pretrained(
    os.environ["MODEL_ID"], torch_dtype=torch.bfloat16
).to(device)

vae_api.parallelize_decoder(pipe.vae, vae_group)
vae_api.parallelize_encoder(pipe.vae, vae_group)

image = pipe("A cat holding a sign that says hello world", height=1024, width=1024).images[0]
if dist.get_rank() == 0:
    image.save("out.png")
```

Then launch it across your GPUs with any pipeline whose VAE DistVAE supports. For example, with a
recent Diffusers release:

```bash
MODEL_ID=black-forest-labs/FLUX.2-dev torchrun --nproc_per_node=4 decode.py
```

Both calls raise if there is no adapter for the VAE, so an unsupported model fails at setup rather
than part way through a decode.

## Supported VAEs

Every family below supports both row sharding and tiling. Qwen-Image is listed with the video VAEs
because its Wan-derived autoencoder has a frame axis.

| VAE              | Frame axis | Tiles by                | A tile is                                          |
| ---------------- | ---------- | ----------------------- | -------------------------------------------------- |
| `AutoencoderKL`  | no         | overlap-derived strides | one decoder call                                   |
| Flux.2           | no         | overlap-derived strides | one decoder call                                   |
| HunyuanVideo 1.5 | yes        | overlap-derived strides | one decoder call                                   |
| HunyuanVideo     | yes        | a stored stride         | one decoder call per temporal chunk                |
| LTX-2            | yes        | a stored stride         | one decoder call unless temporal tiling is enabled |
| Wan              | yes        | a stored stride         | a call per frame, threading a causal cache         |
| Qwen-Image       | yes        | a stored stride         | a call per frame, threading a causal cache         |

Tile size affects the families differently. A smaller tile reduces tile-local activation memory when
one tile is one decoder call, but allocations outside the spatial tile can determine the measured
peak. Wan and Qwen-Image decode one frame at a time, so their peak memory is often set by temporal
state.

`tile_overlap_plan` accepts exact output-pixel `(height, width)` values and maps them to each VAE's
stride settings. DistVAE owns the tiling loop for every family in the table. CogVideoX is excluded
because it tiles frames inside the spatial loop, so its spatial tiles are not independent.

## Distributed decode strategies

DistVAE provides two distributed decode strategies:

- **Row sharding** gives each rank a band in every adapted layer. It exchanges convolution halos and
  normalization statistics, preserves the unsharded result within numerical tolerance, and usually
  reduces activation memory as ranks are added.
- **Whole-tile distribution** gives each rank complete windows. Ranks exchange tile-edge data and
  gather decoded pieces for assembly. Peak activation memory usually follows the tile window,
  including on one GPU, while overlap repeats work and tile-local normalization can change the
  output.

The figure compares the two distributed paths at two tile sizes. Each row reports peak activations,
decoded work, seams, load imbalance, and synchronization:

![Row sharding and two whole-tile distributions for a 1024 by 1024 image on four GPUs, compared by peak activations, work, seams, load imbalance, and synchronization](docs/figure.png)

[Choosing a decode path](docs/strategies.md) explains how VAE family, input shape, rank count, and
interconnect affect the choice. The [benchmark guide](bench/README.md) shows how to measure both
strategies against a vanilla unsharded Diffusers decode.

## Usage

The quickstart uses `distvae.vae`, which picks the adapter for a whole VAE. To shard a single
diffusers module instead, wrap it in its adapter:

```python
import os

import torch
import torch.distributed as dist
from diffusers.models.autoencoders.vae import Decoder
from distvae.modules.adapters.vae.decoder_adapters import DecoderAdapter

dist.init_process_group(backend="nccl")
local_rank = int(os.environ["LOCAL_RANK"])
device = torch.device(f"cuda:{local_rank}")
torch.cuda.set_device(device)
torch.manual_seed(42)  # every rank must build the same weights and the same input

decoder = Decoder(
    in_channels=4, out_channels=3,
    up_block_types=["UpDecoderBlock2D"] * 4,
    block_out_channels=(128, 256, 512, 512), layers_per_block=2,
    norm_num_groups=32, act_fn="silu",
).to(device)

hidden_state = torch.randn(1, 4, 128, 128, device=device)
with torch.no_grad():
    expected = decoder(hidden_state)

# The adapter takes ownership of decoder and replaces its distributed layers in
# place. Do not use decoder as an unmodified reference after this call.
patch_decoder = DecoderAdapter(decoder, dist.group.WORLD).to(device)
with torch.no_grad():
    assert torch.allclose(expected, patch_decoder(hidden_state), atol=1e-2)
```

There are more runnable examples in `test/`.

### Tiling

Diffusers decides whether to tile. DistVAE resizes the window and distributes the tiles across the
group:

```python
from distvae import vae as vae_api

vae_api.require_vae_support(pipe.vae, "tiling", "enable_tiling()")
pipe.vae.enable_tiling()

# Optional: ask for an exact 192x192px window. Invalid shapes are refused rather
# than silently changed.
plan = vae_api.tile_shape_plan(pipe.vae, 192, 192)
if plan is None:
    raise ValueError("this VAE cannot use a 192x192px tile shape")
vae_api.apply_tile_plan(pipe.vae, plan)

# Optional: overlap neighbouring tiles by 32 output pixels vertically and 64
# horizontally. This reads the window now set on the VAE, so apply it second.
step = vae_api.tile_overlap_plan(
    pipe.vae, 32, 64, sample_shape=(1024, 1024)
)
if step is None:
    raise ValueError("this VAE cannot use a 32x64px tile overlap")
vae_api.apply_tile_plan(pipe.vae, step)
replacement = vae_api.tiled_decode_for(pipe.vae)
if replacement is not None:
    pipe.vae.tiled_decode = replacement

# Decode the tiles across the group instead of one after another.
if not vae_api.supports_tile_parallel(pipe.vae):
    raise ValueError("this VAE does not support distributed tiled decode")
dispatch, assemble = vae_api.sharing(vae_group)
tiled_decode = vae_api.tiled_decode_for(pipe.vae, dispatch, assemble)
if tiled_decode is None:
    raise ValueError("no distributed tiled decode is available for this VAE")
pipe.vae.tiled_decode = tiled_decode
```

Window and overlap are separate controls in output pixels. The window sets the memory required for
one tile. The overlap reduces the stride and increases repeated work.

Both planners return `None` when a request cannot be represented exactly. Apply `tile_shape_plan`
first because `tile_overlap_plan` reads the current tile shape. Requested overlap values are never
rounded.

[Choosing a tile window](docs/tiling.md) explains rectangular windows, clipped edge tiles, and
overlap.

### xDiT integration

xDiT chooses the tile settings and calls the DistVAE planners. Supply `vae_tile_overlap_height` and
`vae_tile_overlap_width` together in output pixels. Use zero on an axis that is not tiled.
Installing new shape or overlap settings replaces the previous tiled decode callable.

## Performance

Latency and memory depend on the VAE family, input shape, rank count, device, and interconnect. The
benchmark chooses up to three rectangular plans and records their work, memory estimate, and load
imbalance before running them. See `bench/README.md` for the suite and its limits.

## Development

```bash
git clone https://github.com/xdit-project/DistVAE
cd DistVAE
pip install -e ".[dev]"
mdformat --extensions gfm --wrap 100 README.md bench/README.md docs/*.md
pytest
```

Tests marked `gloo` spawn several ranks over gloo and need no accelerator, so `pytest -m gloo`
exercises the distributed paths on a CPU-only machine.

`docs/make_figure.py` regenerates `docs/figure.svg` and, when `cairosvg` is installed,
`docs/figure.png`.

## License

MIT. See `LICENSE.txt`.
