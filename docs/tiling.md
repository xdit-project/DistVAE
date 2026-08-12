# Choosing a tile window

The [`Tiling` section of the README](../README.md#tiling) shows the API. This page explains how window shape, overlap, and rank count affect a tiled decode. Examples use the 128 × 128 latent from [Row sharding or tiling](strategies.md).

## The two axes cost differently

`tile_shape_plan` sets height and width separately. Tile area determines memory, while each tiled axis adds overlap. In the figure, full-width strips decode 1.26 times the latent area; the two-axis grid decodes 1.46 times.

`tile_overlap_plan` also sets each axis separately. A rectangular window does not require different overlaps. Use different values only when the two axes have different seam or stride requirements.

A window at least as wide as the image leaves the width axis untiled and produces full-width strips. Pass `sample_shape` to `tile_overlap_plan`; an untiled axis must request zero overlap.

``` python
from distvae import vae as vae_api

height, width = 1024, 1024

# 224 rows deep, and wider than the image across, so the across stride clears it in one
# step and the grid comes out as one column of full-width strips.
shape = vae_api.tile_shape_plan(pipe.vae, 224, 1408)
if shape is None:
    raise ValueError("this VAE cannot use a 224x1408px tile shape")
vae_api.apply_tile_plan(pipe.vae, shape)
step = vae_api.tile_overlap_plan(
    pipe.vae, 56, 0, sample_shape=(height, width)
)
if step is None:
    raise ValueError("this VAE cannot use a 56x0px tile overlap")
vae_api.apply_tile_plan(pipe.vae, step)
replacement = vae_api.tiled_decode_for(pipe.vae)
if replacement is not None:
    pipe.vae.tiled_decode = replacement
```

Strips repeat less work and create fewer seams than a two-axis grid, but retain the full size of the untiled axis. In the figure, four strips hold 34% of the activations and create three seams. The 3 × 5 grid holds 12% and creates twenty-two seams.

For a wide image, columns can keep the tiled dimension larger; for a tall image, rows can do the same.

## Clipping unbalances a grid, not the tile count

The last row and column may contain smaller, clipped tiles. Assigning the same number of tiles to each rank can therefore assign different amounts of work.

The figure's four strips cover 43, 43, 43, and 32 latent rows. This leaves the heaviest rank 6.8% above an even split. The final strip is shorter by exactly the overlap, so the imbalance comes from blending at the image boundary.

The 3 × 5 grid is only 0.5% above an even split because each rank receives several tiles. More tiles give the scheduler more ways to balance clipped edges.

When each rank receives one tile, full tiles determine peak memory and wall time. Increasing overlap can sometimes use otherwise idle time without changing either value. This happens only when the wider overlap does not increase the largest tile or the number of tiles; verify it with the benchmark.

## The tile count caps the GPU count

The window and image shape determine the tile count. Distributed tiling cannot use more ranks than tiles. Load balance also depends on how full and clipped tiles divide among the ranks. A smaller window creates more tiles and can improve balance, at the cost of more overlap and seams.

## Benchmark plan selection

The DistVAE planners validate an exact request; they do not choose policy for an application.
The benchmark searches grids containing between `max(2, ranks)` and `4 × ranks` tiles. It rejects
unsupported windows and removes candidates that are worse in window area, decoded area, rank
imbalance, and tile columns. It selects up to three distinct windows from the remaining frontier:
`coarse` has the largest window, `fine` has the smallest, and `balanced` minimizes the worst
normalized window-area, decoded-area, and imbalance score among the other candidates.

The narrow axis must contain at least sixteen latent units and at least one unit per rank. Overlap
on each tiled axis must be at least one quarter of the window. These are conservative limits, not
image-quality measurements. Synthetic weights cannot measure normalization drift or visible
seams. Test the selected window on a trained model before using it in production.
