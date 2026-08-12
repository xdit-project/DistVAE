# Choosing a tile window

The [`Tiling` section of the README](../README.md#tiling) covers the calls. This page is about what to ask them for. Throughout, "the figure's latent" is the 128 × 128 one from [Row sharding or tiling](strategies.md), cut into three rows of five by a 432 × 296 window overlapping 72 pixels on each axis.

## The two axes cost differently

The window has a shape as well as a size, and `tile_shape_plan` sets its height and width separately. That matters because what a rank holds is the product of the two, but the overlap is paid once per axis that is actually cut. On the figure's latent, at much the same overlap, cutting only the rows covers 1.26× the latent, where cutting both covers 1.46×.

The overlap has two axes as well, and `tile_overlap_plan` takes them separately, but a rectangular window is not on its own a reason to make them differ. Asking for the diffusers quarter of one hands a deeper blend to whichever axis is longer, which is a per-axis decision nobody made: a seam is a seam, and what a viewer notices is the thinnest blend on the page. The figure's grid takes one 72-pixel overlap on both axes for that reason. Set the two apart when the axes want different things, not because the window is not square.

A window wider than the image is how to say that an axis should not be cut at all: its stride then clears the image in one step, and the grid comes out as one column of full-width strips. `tile_overlap_plan` takes the output shape for exactly this case. Given `sample_shape`, an axis whose sample fits its window is inactive and must request zero overlap. Active axes still take exact output-pixel counts.

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

Strips are therefore the cheapest tiling in both work and seams, and the most expensive in memory, because the axis left alone still costs its full extent. They also suit the memory layout best: a full-width strip is one unbroken span of a row-major tensor, where a grid's tile is a stride through every row it touches. Four full-width strips over the figure's latent hold 34% of the activations where the three-by-five grid holds 12%, and leave three seams where the grid leaves twenty-two. The figure's lower two rows are that pair.

Which way the strips run barely changes that: a given number of them holds about the same share whichever axis they lie along, since the latent is as long as it is wide. What changes is how thin each one gets. Cutting the long axis leaves each strip more depth in the direction it was cut, so a wide image wants columns and a tall one wants rows.

## Clipping unbalances a grid, not the tile count

The bounds cut the last row and the last column short, so the cheap tiles are gathered at one end of the grid rather than spread through it, and a rank holding a single tile may be holding the cheapest one.

Four strips over the figure's latent cannot come out even at all, whatever you ask for. All four have to start inside the 128 rows, so the stride is at least 32, and the fourth still has to stop at the bottom. The best available is three strips at 32 rows plus the overlap and a last one at 32, which is what the figure's 344-pixel window overlapping 88 gives: 43, 43, 43 and 32, leaving the heaviest rank 6.8% past an even split. At that floor the short strip is short by exactly the overlap, so the idle time is not a bad split point but the blend, priced in time.

The same latent cut into the figure's fifteen tiles is half a percent past, because the short tiles are a smaller part of what each rank carries. Giving each rank several tiles is what averages the clipping out, and it is the reliable way to get a balanced grid.

Where a rank does hold one tile, the stride stops being a cost and becomes spare capacity. The decode waits for a full window however the strips are spaced, so the stride cannot make it quicker; all it decides is how much of the idle rank's time goes on overlap. Round the window up to 352 pixels at that same 88 and it steps by 33 instead. The last strip drops to 29 rows, its rank now sits out a third of the decode rather than a quarter, and the blend is no deeper for it. Widening the request to 96 pixels steps it back to 32, adding another row of blend across the whole image at the same peak memory and the same wall clock, because the three full strips set both and they have not changed. That extra 0.02× of coverage comes entirely out of time that was being wasted. This is the one case where widening the overlap is free, and it is worth checking for whenever the tiles divide evenly among the ranks.

## The tile count caps the GPU count

The window and the requested image shape fix the tile count, and that count is the ceiling on how many GPUs the image can use. The figure's fifteen tiles come within 14% of an even split at eight ranks. The square sixteen they beat come within 53%, because nine of those are full size and eight ranks cannot avoid giving one of them two full tiles. Narrowing to a 288 × 256 window gives thirty tiles and comes within 5%. That is arithmetic rather than a scheduling failure, and it is the one place where the GPU count does bear on the grid.

## The benchmark search is bounded

The DistVAE planners validate an exact request; they do not choose policy for an application.
The benchmark adds a small topology search for measurement. It considers grids with at most four
tiles per rank, rejects windows that the VAE cannot represent, and removes candidates dominated
on window area, decoded area, and rank imbalance. Three plans remain in the timed suite:
throughput, a frontier knee, and memory.

That frontier does not include visual quality. DistVAE enforces no minimum window beyond what the
VAE can represent, and synthetic benchmark weights cannot price group-normalization drift or
seams. Use the shortlist to measure latency and memory, then check the chosen window on a trained
model. The window controls memory and how much of the image changes. Overlap controls redundant
work and the blend at each seam.
