# Choosing a decode path

DistVAE provides row sharding and whole-tile distribution. The benchmark compares both strategies
with a vanilla unsharded Diffusers decode:

![Row sharding and two whole-tile distributions for a 1024 by 1024 image on four GPUs, compared by peak activations, work, seams, load imbalance, and synchronization](figure.png)

[`make_figure.py`](make_figure.py) generates the diagram from the same scheduler used at runtime.
Tile sizes and rank assignments are exact for the example.

## Comparison

|                                   | Vanilla unsharded         | Row sharding                                     | Whole-tile distribution                        |
| --------------------------------- | ------------------------- | ------------------------------------------------ | ---------------------------------------------- |
| Work assigned to a rank           | Complete decode           | A band in every adapted layer                    | One or more complete tiles                     |
| Communication during the VAE call | None                      | Halos, metadata, and normalization statistics    | Tile-edge exchange and output assembly         |
| Peak activation memory            | Full decode on every rank | Usually falls as ranks are added                 | Usually follows the largest tile               |
| Repeated work                     | None                      | None                                             | Overlap between tiles                          |
| Output                            | Reference                 | Matches the reference within numerical tolerance | Can differ because normalization sees one tile |

The benchmark uses vanilla Diffusers decoding as its numerical reference. When that decode fits and
no VAE distribution is needed, DistVAE does not need to replace it. Of DistVAE's two strategies, row
sharding is the default when numerical agreement matters. Whole-tile distribution is useful when the
activation memory of a row-sharded band is still too large.

Latency and memory depend on the VAE family, input shape, rank count, tile geometry, and
interconnect. Run all available paths on the target system rather than selecting one from rank count
alone.

## Family guidance

| Family                     | Starting point                             | What to check                                                                                                             |
| -------------------------- | ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------- |
| `AutoencoderKL` and Flux.2 | Whole-tile strips under memory pressure    | A full-width or full-height strip can be faster and lighter than row sharding. Compare every planned window.              |
| Qwen-Image                 | Whole-tile strips under memory pressure    | Its frame-at-a-time decoder can retain memory outside the spatial tile, so smaller windows do not guarantee a lower peak. |
| Wan                        | Row sharding for latency                   | Tiling can reduce peak memory, but row sharding may remain faster.                                                        |
| LTX-2                      | Row sharding                               | Tiled plans can use more memory than row sharding. Measure peak memory before enabling them.                              |
| HunyuanVideo               | Tiling when row-sharded memory is too high | Tiling can reduce memory while remaining slower than row sharding.                                                        |
| HunyuanVideo 1.5           | Benchmark both distributed paths           | A selected tiled plan may reduce both latency and memory.                                                                 |

The table lists plausible tradeoffs to measure; none is guaranteed on a given system. The benchmark
uses synthetic weights, so it cannot measure trained-model quality or end-to-end pipeline memory.

## Communication

Row sharding communicates inside adapted layers. Convolutions exchange neighboring rows, distributed
group normalization reduces statistics, and uneven outputs require metadata and output gathers. The
number of distributed API calls therefore follows the decoder architecture.

Whole-tile distribution communicates around independent decoder calls. Ranks exchange tile-edge data
for blending and gather completed pieces for assembly. Most families do this once for the complete
decode. HunyuanVideo repeats the spatial tiling operation for each temporal chunk.

The unsharded path performs no distributed operation inside the measured VAE call. Benchmark
barriers around timed iterations are excluded from that statement.

## Whole tiles rather than rows inside them

DistVAE assigns complete tiles to ranks. Sharding every tile by rows would add patching, halo
exchange, and gathering to each tile. Complete tiles can be decoded independently.

The scheduler balances tile area, not tile count, because tiles on the last row or column may be
clipped. A tile cannot be divided between ranks, so balance improves when each rank receives several
tiles. If the grid has fewer tiles than ranks, distributed tiling is disabled and every rank decodes
the full grid.

Latency depends on the VAE, shape, interconnect, and tile geometry. The
[benchmark guide](../bench/README.md) explains how to compare them.

## Why the window is rectangular

Height and width affect the grid independently. A rectangular window can reduce clipping or avoid
cutting one axis. The figure's 432 × 296 window produces a 3 × 5 grid with better load balance than
a 384 × 384 window at the same overlap.

[Choosing a tile window](tiling.md) covers strips, clipping, overlap, and rank count.

## Video

Both distributed paths split only spatial axes. Every band or tile keeps its current temporal
extent. [Temporal decoding](tiling.md#temporal-decoding) describes how each VAE iterates over that
extent.
