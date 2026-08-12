# Row sharding or tiling

DistVAE can split one decoder call across ranks or distribute complete tiles:

![Row sharding and two whole-tile distributions for a 1024 by 1024 image on four GPUs, compared by peak activations, work, seams, load imbalance, and synchronization](figure.png)

[`make_figure.py`](make_figure.py) generates the diagram from the same scheduler used at runtime. Tile sizes and rank assignments are exact for the example.

## Comparison

| | Row sharding | Whole-tile distribution |
| --- | --- | --- |
| Work assigned to a rank | A band of every layer | One or more complete tiles |
| Communication | Inside adapted layers | Tile distribution and output assembly |
| Peak activation memory | Falls as ranks are added | Follows the tile size |
| Repeated work | None | Overlap between tiles |
| Output | Matches the unsharded decode within numerical tolerance | Can differ because normalization sees one tile |

Row sharding is the better fit when exact agreement matters or when a large tile already fits. Tiling is useful when peak activation memory is the limit.

For 2D VAEs and HunyuanVideo, one tile is one decoder call, so a smaller window usually lowers peak memory. Wan and Qwen-Image decode a tile one frame at a time; reducing the spatial window may not lower their peak.

## Whole tiles rather than rows inside them

DistVAE assigns complete tiles to ranks. Sharding every tile by rows would add patching, halo exchange, and gathering to each tile. Complete tiles can be decoded independently.

The scheduler balances tile area, not tile count, because tiles on the last row or column may be clipped. A tile cannot be divided between ranks, so balance improves when each rank receives several tiles. If the grid has fewer tiles than ranks, distributed tiling is disabled and every rank decodes the full grid.

Latency depends on the VAE, shape, interconnect, and tile geometry. The [benchmark guide](../bench/README.md) explains how to compare them.

## Why the window is rectangular

Height and width affect the grid independently. A rectangular window can reduce clipping or avoid cutting one axis. The figure's 432 × 296 window produces a 3 × 5 grid with better load balance than a 384 × 384 window at the same overlap.

[Choosing a tile window](tiling.md) covers strips, clipping, overlap, and rank count.

## Video

Both modes split only spatial axes. Every band or tile keeps all of its frames. Wan and Qwen-Image decode those frames one at a time to maintain a causal cache; the other supported video VAEs decode each spatial tile in one call.
