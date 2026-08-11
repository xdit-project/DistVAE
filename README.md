# DistVAE

Split a diffusers VAE across GPUs. DistVAE swaps the encoder and decoder for sharded versions through a set of adapters and leaves the rest of the model untouched, so the VAE stops being the memory spike in high-resolution generation.

## Installation

``` bash
pip install distvae
```

Python 3.10 or newer, with `torch>=2.2` and `diffusers>=0.35`.

## Quickstart

Every rank builds the same pipeline, and DistVAE shards the VAE inside it. Save this as `decode.py`:

``` python
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
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
).to(device)

vae_api.parallelize_decoder(pipe.vae, vae_group)
vae_api.parallelize_encoder(pipe.vae, vae_group)

image = pipe("an astronaut riding a horse", height=1024, width=1024).images[0]
if dist.get_rank() == 0:
    image.save("out.png")
```

Then launch it across your GPUs:

``` bash
torchrun --nproc_per_node=4 decode.py
```

Both calls raise if there is no adapter for the VAE, so an unsupported model fails at setup rather than part way through a decode.

## Supported VAEs

Every family below has both adapters and can be row sharded or tiled. Qwen-Image is grouped with the video VAEs because its autoencoder is Wan-derived and takes a frame axis, not because it makes video.

| VAE | Frame axis | Tiles by | A tile is |
| --- | --- | --- | --- |
| `AutoencoderKL` | no | overlap-derived strides | one decoder call |
| Flux.2 | no | overlap-derived strides | one decoder call |
| HunyuanVideo 1.5 | yes | overlap-derived strides | one decoder call |
| HunyuanVideo | yes | a stored stride | one decoder call |
| LTX-2 | yes | a stored stride | one decoder call |
| Wan | yes | a stored stride | a call per frame, threading a causal cache |
| Qwen-Image | yes | a stored stride | a call per frame, threading a causal cache |

Read the last column before narrowing a window. Where a tile is one call, the window sets how much memory a rank needs. Where it is a call per frame, that memory is already spent elsewhere and narrowing the window does nothing. `tile_overlap_plan` takes an exact output-pixel `(height, width)` overlap for every family and maps that request to the attributes its loop stores. `supports_tile_parallel` is true for every row, because DistVAE owns the tiling loop. CogVideoX is the notable absence, since it tiles frames inside the spatial loop rather than above it and its tiles are therefore not independent.

## Row sharding or tiling

Two ways to cut a decode down to size, and they cost different things. The figure prices both, and tiling at two windows, in the same five columns:

![Generating a 1024 by 1024 image from a 128 by 128 latent on four GPUs: row sharding, then tile distribution at two windows, each priced in the same five columns](docs/figure.png)

**Row sharding** gives every rank a band of rows and syncs inside every layer, so the image matches an unsharded decode. Communication scales with the depth of the decoder. Every rank still runs that whole decoder, so per-rank memory falls with the GPU count only down to the weights.

**Tiling** gives each rank whole windows and exchanges twice for the entire decode. Peak memory tracks the tile rather than the image or the GPU count, which is why it is the only one that helps on a single GPU. The cost is redundant work at the overlaps, and some fidelity: a group norm inside a tile sees only that tile.

[Row sharding or tiling](docs/strategies.md) covers the rest: why DistVAE deals whole tiles out rather than sharding inside the loop, what that costs in granularity, why the best window on a square latent is rectangular, and where video fits.

## Usage

The quickstart uses `distvae.vae`, which picks the adapter for a whole VAE. To shard a single diffusers module instead, wrap it in its adapter:

``` python
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

Diffusers decides whether to tile. DistVAE resizes the window and deals the tiles across the group:

``` python
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

The window and the overlap are separate controls, and both are set in absolute output pixels rather than as a fraction of anything. The window sets what one tile costs in memory. The overlap sets how much of the decode is redundant: it narrows the stride the loop walks, and tiling both axes covers `(height_window / height_stride) × (width_window / width_stride)` times the latent.

Three things to know about the planners. Both return `None` when they cannot meet a request exactly, so check before applying. Apply `tile_shape_plan` before `tile_overlap_plan`, which reads the shape currently set on the VAE. Overlap is never rounded or widened: each requested pixel count must map exactly to the loop's stride arithmetic.

[Choosing a tile window](docs/tiling.md) covers what to ask them for: how the two axes differ, why clipping rather than tile count is what unbalances a grid, and where widening the overlap is free.

### xDiT integration

xDiT owns tile-policy choices and calls the DistVAE planners. Its
`vae_tile_overlap_height` and `vae_tile_overlap_width` settings are exact output pixels and must
be supplied together. Use zero for an inactive strip axis. Custom shape or overlap settings
install a fresh tiled-decode replacement; a later installation replaces the earlier callable
rather than wrapping it.

## Performance

Latency and memory depend on the VAE family, input shape, rank count, device, and interconnect.
The benchmark chooses three bounded rectangular plans and records their work, memory proxy, and
load imbalance before measuring them. See `bench/README.md` for the suite and its limits.

## Development

``` bash
git clone https://github.com/xdit-project/DistVAE
cd DistVAE
pip install -e ".[dev]"
pytest
```

Tests marked `gloo` spawn several ranks over gloo and need no accelerator, so `pytest -m gloo` exercises the distributed paths on a CPU-only machine.

`docs/make_figure.py` redraws the figure above. It writes the SVG with the standard library alone, and the PNG too if `cairosvg` is installed.

## License

MIT. See `LICENSE.txt`.
