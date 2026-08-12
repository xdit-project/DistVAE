"""Distribute complete VAE tiles among the ranks of a process group.

Tiling and sharding both split a VAE decode, and composing them splits it twice. DistVAE shards
the rows of each tile independently, so every tile requires Patchify, a halo exchange per
convolution, a reduction per norm, and a gather. This communication cost is per tile rather than
per pixel, so it increases as the window narrows and the tile count grows.

Tiles are independent, although rows within a tile are not. Distributing complete tiles requires
two exchanges for the full decode regardless of tile count, and each rank decodes its assigned
tiles without row sharding.

The caller supplies one callable per decoder invocation and receives every result on every rank
in call order.
"""

import functools
import math
import warnings
from typing import Callable, Dict, List, NamedTuple, Optional, Sequence, Set, Tuple

import torch
import torch.distributed as dist

from distvae.utils import ParallelContext

# Recorded on the VAE itself, because the decision is made when the decoder would otherwise be
# sharded and acted on later, when the tile window is settled and the decode is installed.
GROUP_ATTR = "_distvae_tile_parallel_context"

Call = Callable[[], torch.Tensor]
Dispatch = Callable[[Sequence[Call]], List[torch.Tensor]]

Where = Tuple[int, int]
Decode = Callable[[Sequence[Where]], Dict[Where, torch.Tensor]]

# Both diffusers tiling loops blend down the second-from-last axis and across the last one, on a
# 4D sample and a 5D one alike, so the assembly below needs no axis of its own to be told.
DOWN, ACROSS = -2, -1


class Blend(NamedTuple):
    """Functions and dimensions used by Diffusers tiling loops to combine adjacent tiles."""

    down: Callable  # blend_v: mixes a tile's first `deep_down` rows with the tile above's last
    across: Callable  # blend_h: mixes its first `deep_across` columns with the left tile's last
    deep_down: int
    deep_across: int
    crop: Callable[
        [torch.Tensor], torch.Tensor
    ]  # the corner of a blended tile that is kept
    # The configured window gives every rank the same whole-tile dimensions. A rank-local decoded
    # tile may be clipped; using it could make one rank fall back while the others enter a gather.
    tile_down: int
    tile_across: int


def mark(vae, context: ParallelContext) -> None:
    """Record this VAE's immutable tile-distribution context."""
    if not isinstance(context, ParallelContext):
        raise TypeError("tile-parallel metadata requires a ParallelContext")
    setattr(vae, GROUP_ATTR, context)


def context_of(vae) -> Optional[ParallelContext]:
    """Return this VAE's tile-parallel context, if one was recorded."""
    return getattr(vae, GROUP_ATTR, None)


def group_of(vae):
    """Return the process group recorded for this VAE's tile distribution."""
    context = context_of(vae)
    return context.group if context is not None else None


def _distributed(context_or_group):
    """Return group, rank, and size from a context or a direct group argument."""
    if isinstance(context_or_group, ParallelContext):
        return (
            context_or_group.group,
            context_or_group.rank,
            context_or_group.world_size,
        )
    group = context_or_group
    return group, dist.get_rank(group), dist.get_world_size(group)


def in_order(calls: Sequence[Call]) -> List[torch.Tensor]:
    """Execute all calls sequentially in input order."""
    return [call() for call in calls]


def dispatch_over(group) -> Dispatch:
    """Distribute calls across `group` and return all results on every rank."""
    group, rank, world_size = _distributed(group)
    if world_size < 2:
        return in_order

    def dispatch(calls: Sequence[Call]) -> List[torch.Tensor]:
        # Fewer calls than ranks and some rank contributes nothing to the exchange, with no
        # tensor of its own to take a dtype and a device from. A decode that small is a tile or
        # two, so every rank simply making every call costs less than arranging not to.
        if len(calls) < world_size:
            return in_order(calls)
        made = [
            call() if n % world_size == rank else None for n, call in enumerate(calls)
        ]
        return _share(made, group, world_size)

    return dispatch


def sharing(group) -> Tuple[Dispatch, Callable]:
    """Return contiguous-run assembly and per-call dispatch for a process group.

    Use contiguous-run assembly when every rank can receive a tile and each tile is large enough
    to blend locally. Otherwise, distribute individual decoder calls and assemble the results on
    every rank.
    """
    return dispatch_over(group), functools.partial(assemble_in_runs, group)


def runs(weights: Sequence[int], world_size: int) -> List[Tuple[int, int]]:
    """Split tiles into one contiguous, weight-balanced run per rank.

    Contiguous runs preserve tiling-loop order and keep most adjacent tiles on the same rank.
    Tiles provide finer load balancing than complete grid rows.

    Balance by tile area rather than tile count. Boundary clipping makes tiles in the last grid
    row and column smaller, so equal tile counts can assign less work to the last rank.

    Minimize the maximum run weight using binary search over feasible weight limits and a greedy
    feasibility check.
    """
    if world_size < 2:
        return [(0, len(weights))]
    low, high = max(weights), sum(weights)
    while low < high:
        middle = (low + high) // 2
        if len(_greedy(weights, middle)) <= world_size:
            high = middle
        else:
            low = middle + 1
    return _widen(_greedy(weights, low), world_size)


def shares(weights: Sequence[int], world_size: int) -> List[int]:
    """Which rank decodes each tile: contiguous runs, levelled by moving or swapping a few tiles

    A run is the cheap shape to blend, since its tiles' neighbours are mostly its own, but it is
    a coarse shape to balance. Nine tiles over four ranks split by weight as evenly as contiguity
    allows still leaves the heaviest rank a quarter above the average, because the tiles are large
    against the share and a run cannot skip one. No weighing fixes that; only a finer assignment.

    So the runs are a starting point rather than the answer. Moves and pairwise swaps are searched
    together across every rank pair. Each accepted change strictly lowers the descending load
    vector, or keeps that vector while restoring a tile to its original run. Among equally balanced
    choices, fewer tiles displaced from those runs win, followed by tiles already beside their new
    owner. The total tie-break is deterministic because every rank computes this independently.

    A move never takes a rank's last tile. Swaps preserve every rank's tile count.
    """
    owner: List[int] = []
    for rank, (start, stop) in enumerate(runs(weights, world_size)):
        owner.extend([rank] * (stop - start))
    if world_size < 2:
        return owner

    load = [0] * world_size
    for n, weight in enumerate(weights):
        load[owner[n]] += weight

    original = owner.copy()
    count = [owner.count(rank) for rank in range(world_size)]
    displaced = 0

    def objective(loads, moved):
        return tuple(sorted(loads, reverse=True)), moved

    # Every accepted operation strictly lowers `objective`, so no ownership state can recur.
    # There are world_size ** tile_count states, which is a conservative finite round bound; the
    # search normally reaches its fixed point after only a handful.
    for _ in range(world_size ** len(weights)):
        current = objective(load, displaced)
        best = None

        for moved, weight in enumerate(weights):
            donor = owner[moved]
            if count[donor] == 1:
                continue
            for receiver in range(world_size):
                if receiver == donor:
                    continue
                loads = load.copy()
                loads[donor] -= weight
                loads[receiver] += weight
                next_displaced = displaced
                next_displaced -= int(owner[moved] != original[moved])
                next_displaced += int(receiver != original[moved])
                candidate = objective(loads, next_displaced)
                if candidate >= current:
                    continue
                beside = any(
                    0 <= neighbour < len(weights) and owner[neighbour] == receiver
                    for neighbour in (moved - 1, moved + 1)
                )
                key = (candidate, 0 if beside else 1, 0, donor, receiver, moved)
                if best is None or key < best[0]:
                    best = (key, "move", moved, receiver, loads, next_displaced)

        for first in range(len(weights)):
            first_rank = owner[first]
            for second in range(first + 1, len(weights)):
                second_rank = owner[second]
                if first_rank == second_rank:
                    continue
                loads = load.copy()
                loads[first_rank] += weights[second] - weights[first]
                loads[second_rank] += weights[first] - weights[second]
                next_displaced = displaced
                next_displaced -= int(first_rank != original[first])
                next_displaced -= int(second_rank != original[second])
                next_displaced += int(second_rank != original[first])
                next_displaced += int(first_rank != original[second])
                candidate = objective(loads, next_displaced)
                if candidate >= current:
                    continue

                def rank_after(tile):
                    if tile == first:
                        return second_rank
                    if tile == second:
                        return first_rank
                    return owner[tile]

                beside = 0
                for tile, receiver in (
                    (first, second_rank),
                    (second, first_rank),
                ):
                    beside += not any(
                        0 <= neighbour < len(weights)
                        and rank_after(neighbour) == receiver
                        for neighbour in (tile - 1, tile + 1)
                    )
                key = (
                    candidate,
                    beside,
                    1,
                    first_rank,
                    second_rank,
                    first,
                    second,
                )
                if best is None or key < best[0]:
                    best = (
                        key,
                        "swap",
                        first,
                        second,
                        loads,
                        next_displaced,
                    )

        if best is None:
            break
        _, operation, first, second, load, displaced = best
        if operation == "move":
            donor = owner[first]
            owner[first] = second
            count[donor] -= 1
            count[second] += 1
        else:
            owner[first], owner[second] = owner[second], owner[first]
    return owner


def _greedy(weights: Sequence[int], ceiling: int) -> List[Tuple[int, int]]:
    """The fewest contiguous runs none of which weighs more than `ceiling`"""
    out, start, carried = [], 0, 0
    for at, weight in enumerate(weights):
        if carried and carried + weight > ceiling:
            out.append((start, at))
            start, carried = at, 0
        carried += weight
    out.append((start, len(weights)))
    return out


def _widen(split: List[Tuple[int, int]], world_size: int) -> List[Tuple[int, int]]:
    """Enough runs for every rank, by halving the ones holding most tiles

    A ceiling that a few ranks can meet leaves the rest with nothing, and a rank holding no tile
    has no tensor of its own to take a dtype and a device from. Halving cannot raise the heaviest
    run, so nothing found above is given up here.
    """
    while len(split) < world_size:
        widest = max(range(len(split)), key=lambda n: split[n][1] - split[n][0])
        start, stop = split[widest]
        if stop - start < 2:
            break  # fewer tiles than ranks, which the caller declines before asking
        middle = (start + stop) // 2
        split[widest : widest + 1] = [(start, middle), (middle, stop)]
    return split


def assemble_in_runs(
    group,
    rows: int,
    columns: int,
    decode: Decode,
    blend: Blend,
    weights: Sequence[int],
) -> Optional[torch.Tensor]:
    """Assemble a tile grid with each rank decoding and blending its own run, None if it can't

    Dealing tiles out divides the decoding and leaves the blending on every rank, a cost that
    does not shrink however many ranks join the group. Giving a rank a share of neighbouring
    tiles lets it blend its own and send only the finished pieces, so the blending divides too.

    The reason a share can be blended alone is a property of the two blends. `blend_v` writes a
    tile's *first* rows and `blend_h` its *first* columns, so neither ever writes the last rows or
    the last columns - and those are the only parts of a tile that the tiles after it read. A
    tile's edges are therefore final while it is still raw, and one exchange of raw edges lets
    every rank blend its run exactly as a single rank walking the whole grid would, waiting on
    nobody else's blending.

    What comes back is each rank's cropped tiles, which are disjoint and tile the image exactly,
    so the gather carries the image once rather than every overlapping tile.

    Where a tile is smaller than twice the blend the argument fails, because the rows and columns
    the blends write would reach into the ones their neighbours read. Every reason to decline is
    one every rank reaches the same way, from the grid and the window rather than from the tiles
    a rank happens to hold: a rank that fell back alone would leave the others waiting in a
    gather it never joins, which hangs a decode rather than failing it.
    """
    group, rank, world_size = _distributed(group)
    if world_size < 2:
        return None
    order = [(i, j) for i in range(rows) for j in range(columns)]
    # Fewer tiles than ranks and some rank would hold nothing, with no tensor of its own to take a
    # dtype and a device from. A decode that small has nothing worth dividing anyway.
    if len(order) < world_size:
        if rank == 0:
            warnings.warn(
                f"VAE tile grid has {len(order)} tiles for {world_size} ranks; "
                f"whole-tile distribution is disabled and every rank will decode all "
                f"{len(order)} tiles locally. Use fewer VAE ranks or a smaller tile window.",
                RuntimeWarning,
                stacklevel=2,
            )
        return None
    if (
        blend.tile_down < 2 * blend.deep_down
        or blend.tile_across < 2 * blend.deep_across
    ):
        return None

    owner = shares(weights, world_size)
    mine = decode([at for n, at in enumerate(order) if owner[n] == rank])

    # Exchanged before anything is blended, both because the edges are raw at that point and
    # because a rank waiting on a neighbour's blending would serialise what this is dividing.
    edges = _share_edges(
        order,
        mine,
        owner,
        rank,
        _wanted(owner, columns, blend),
        group,
        world_size,
        blend,
    )

    blended: Dict[Where, torch.Tensor] = {}
    kept: List[Optional[torch.Tensor]] = [None] * len(order)
    for n, (i, j) in enumerate(order):
        if owner[n] != rank:
            continue
        tile = mine[(i, j)]
        # A blend no rows deep is one the tiles do not overlap enough to need, which a wide
        # enough stride leaves. Skipped rather than called with a zero depth, because a depth of
        # zero reads as "the whole tile" everywhere an edge is sliced off the end of one.
        if i > 0 and blend.deep_down:
            # The neighbour itself where this rank blended it, and its edge rebuilt from the raw
            # ones otherwise. Both carry the same values; only the cost differs.
            above = blended.get((i - 1, j))
            tile = blend.down(
                above if above is not None else _edge_above(edges, i, j, blend),
                tile,
                blend.deep_down,
            )
        if j > 0 and blend.deep_across:
            left = blended.get((i, j - 1))
            tile = blend.across(
                left if left is not None else _edge_left(edges, i, j, blend),
                tile,
                blend.deep_across,
            )
        blended[(i, j)] = tile
        kept[n] = blend.crop(tile)

    shared = _share(kept, group, world_size)
    return torch.cat(
        [
            torch.cat(shared[i * columns : (i + 1) * columns], dim=ACROSS)
            for i in range(rows)
        ],
        dim=DOWN,
    )


def assemble_here(
    rows: int, columns: int, decode: Decode, blend: Blend
) -> torch.Tensor:
    """Assemble the whole grid on this rank, which is what diffusers' own loop does"""
    mine = decode([(i, j) for i in range(rows) for j in range(columns)])
    # Both blends write into the tile they are handed, so each tile is blended against neighbours
    # that were themselves already blended, and the scan order that makes is part of the result.
    made = []
    above: Optional[List[torch.Tensor]] = None
    for i in range(rows):
        row = [mine[(i, j)] for j in range(columns)]
        kept = []
        for j, tile in enumerate(row):
            if above is not None:
                tile = blend.down(above[j], tile, blend.deep_down)
            if j > 0:
                tile = blend.across(row[j - 1], tile, blend.deep_across)
            row[j] = tile
            kept.append(blend.crop(tile))
        made.append(torch.cat(kept, dim=ACROSS))
        above = row
    return torch.cat(made, dim=DOWN)


def _wanted(owner: Sequence[int], columns: int, blend: Blend) -> Set[int]:
    """Return tiles whose unblended edges are needed by another rank.

    A tile blends with its upper and left neighbors. An edge must be transferred only when that
    neighbor belongs to another rank. Contiguous assignments usually require one boundary row per
    rank; load-balancing moves may add boundaries around the moved tile.

    An axis with zero overlap requires no edge transfer.
    """
    wanted: Set[int] = set()
    for n, rank in enumerate(owner):
        row, column = divmod(n, columns)
        if blend.deep_down and row > 0 and owner[n - columns] != rank:
            wanted.add(n - columns)
            if blend.deep_across and column > 0:
                wanted.add(n - columns - 1)
        if blend.deep_across and column > 0 and owner[n - 1] != rank:
            wanted.add(n - 1)
            if blend.deep_down and row > 0:
                wanted.add(n - columns - 1)
    return wanted


def _share_edges(
    order: Sequence[Where],
    mine: Dict[Where, torch.Tensor],
    owner: Sequence[int],
    rank: int,
    wanted: Set[int],
    group,
    world_size: int,
    blend: Blend,
) -> Dict[Where, Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]]:
    """The last rows and last columns of the tiles another rank will read, raw, on every rank

    The edges rather than the tiles: what a neighbour reads is one blend deep, so this carries a
    fraction of what dealing the tiles themselves round would have to.
    """
    sending: List[Optional[torch.Tensor]] = [None] * (2 * len(order))
    for n in wanted:
        if owner[n] != rank:
            continue
        tile = mine[order[n]]
        # Cloned because the blending below writes into the tiles these came from, and an edge is
        # only the edge a neighbour needs while it is still raw. Guarded on the depth because a
        # slice from -0 is the whole tile rather than none of it, which would send the grid
        # itself round in place of its seams.
        if blend.deep_down:
            sending[2 * n] = tile[..., -blend.deep_down :, :].clone()
        if blend.deep_across:
            sending[2 * n + 1] = tile[..., -blend.deep_across :].clone()
    shared = _share(sending, group, world_size, next(iter(mine.values())))
    return {at: (shared[2 * n], shared[2 * n + 1]) for n, at in enumerate(order)}


def _edge_above(edges, i: int, j: int, blend: Blend) -> torch.Tensor:
    """The last rows of the tile above, as its own rank would have blended them

    Only its blending across the columns reaches its last rows, and that reads its left
    neighbour's last columns, which nothing writes. One blend of two raw edges rebuilds it.
    """
    below, _ = edges[(i - 1, j)]
    if j == 0 or not blend.deep_across:
        return below
    left, _ = edges[(i - 1, j - 1)]
    return blend.across(left, below.clone(), blend.deep_across)


def _edge_left(edges, i: int, j: int, blend: Blend) -> torch.Tensor:
    """The last columns of the tile to the left, as its own rank would have blended them

    Only its blending down the rows reaches its last columns, and that reads the corner where the
    tile above it meets the tile above and to its left - raw on both counts.
    """
    _, beside = edges[(i, j - 1)]
    if i == 0 or not blend.deep_down:
        return beside
    above, _ = edges[(i - 1, j - 1)]
    return blend.down(above[..., -blend.deep_across :], beside.clone(), blend.deep_down)


def _share(
    made: List[Optional[torch.Tensor]],
    group,
    world_size: int,
    like: Optional[torch.Tensor] = None,
) -> List[torch.Tensor]:
    """Fill in the calls this rank did not make from the ranks that did.

    `like` supplies edge metadata when this rank has no local tensor of the required shape. The
    last run needs it because no following run provides an edge shape.
    """
    mine = [(n, tensor) for n, tensor in enumerate(made) if tensor is not None]

    # A rank cannot work out the shape of a call it did not make: tiles at the right and bottom
    # edges are clipped by the latent bounds, and a rank can hold none of them. One object
    # exchange settles that for the whole decode.
    manifest: List = [None] * world_size
    dist.all_gather_object(
        manifest, [(n, tuple(t.shape)) for n, t in mine], group=group
    )

    # Nothing to send is a real answer here, not an empty group: the last run has no run after it
    # to read its edges. Its rank still joins the exchange above, so nobody is left waiting.
    width = max(sum(math.prod(shape) for _, shape in entries) for entries in manifest)
    if width == 0:
        return list(made)

    # Then one tensor exchange for the results themselves, flattened together and padded to the
    # largest share, since all_gather wants every rank sending the same count. Ranks differ by at
    # most one call, so the padding is at most one call's worth of the traffic.
    #
    # Filled a call at a time rather than concatenated into the buffer, which would hold a second
    # copy of everything this rank decoded while the first was still alive. Left uninitialised
    # past what this rank sends: the manifest bounds what each share is read back out of, so the
    # padding is never looked at.
    sample = mine[0][1] if mine else like
    sending = torch.empty(width, dtype=sample.dtype, device=sample.device)
    at = 0
    for _, tensor in mine:
        sending[at : at + tensor.numel()] = tensor.reshape(-1)
        at += tensor.numel()
    received = [torch.empty_like(sending) for _ in range(world_size)]
    dist.all_gather(received, sending, group=group)

    shared: List[torch.Tensor] = list(made)
    for entries, buffer in zip(manifest, received):
        at = 0
        for n, shape in entries:
            size = math.prod(shape)
            # What this rank decoded itself is kept as it decoded it, rather than read back out
            # of its own copy in the buffer.
            if shared[n] is None:
                shared[n] = buffer[at : at + size].view(shape)
            at += size
    return shared
