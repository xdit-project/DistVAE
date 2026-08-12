"""Draw the README's figure: row sharding, then tile distribution at two windows.

Three rows, one comparison. The first is row sharding. The second is tiling cut on the row
axis alone, which lands on four full-width strips, one per rank: the same shape as the bands
above it, so the only thing that changes between the two is the mechanism. The third cuts
both axes. Reading down, one variable moves at a time.

Every row ends in the same five columns: what a rank holds, what the overlap costs in
redundant work, how many joins a blend has to cover, how far past an even split the heaviest
rank lands, and how often the ranks sync. The first four are all worse for tiling, so
without the fifth the readout says only that tiling is a mistake. Row sharding goes through
the same formulas as the other two, which is what makes it a baseline rather than a special
case.

The right-hand panels are all the same axis, time, with one lane per rank. Each is scaled to
its own heaviest rank, so all three rows end at the same x and the lengths mean nothing
across rows; the work column is what to read for that. Within a row the blocks stay
proportional to the work in their tile, which is what makes the clipped tiles at a grid's
edges visibly cheap, and cheap is why dealing tiles out by area beats dealing them by count.

Both windows are written as the pair of planner calls that would set them, a window and an
overlap in output pixels, so the figure cannot show a configuration the API could not be
asked for. The grids are drawn at the extents that pair leaves, so tiles overlap on the page
as they do in the loop. Neither window is a recommendation, and the heading says so: they
are the two ends of the trade, and the strips end has advantages no column here can show,
being one contiguous span of a row-major tensor where a grid's tile is a stride through
every row it touches.

No row draws its output, because all three produce the same picture. What tiling changes is
a seam a good window renders invisible, so a panel of the result would be either blank or an
exaggeration of what a blend leaves behind. Each heading says what its mode costs the image
in words instead, beside what it costs memory and the interconnect.

Written as plain SVG, so the figure rebuilds with no toolchain and stays legible in a diff.
The tile-to-rank assignment is asked of the scheduler rather than drawn by hand, so the
picture cannot drift from what a decode actually does. Nothing is positioned at an absolute
y: every block reports where it ended and the next starts from there, so a caption can be
added without re-tuning the page.

    python docs/make_figure.py
"""

import importlib.util
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "figure.svg")
# The README points at the PNG, because a local SVG does not preview in every editor a README
# is read in. It is written only where cairosvg is installed, so rebuilding the SVG itself
# still needs nothing but a Python interpreter.
PNG = os.path.join(HERE, "figure.png")
RASTER = 2

RANKS = 4
# A 1024 by 1024 image, and so 128 latent rows on an eight-fold VAE.
BOUND = 128
SCALE_VAE = 8

NUMBERS = ("no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
           "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen")


def word(n):
    """A small count spelled out, since the captions are prose and the columns are not"""
    return NUMBERS[n] if n < len(NUMBERS) else str(n)


def scheduler():
    """The tile-to-rank assignment the library ships, so the figure cannot invent one"""
    try:
        from distvae.vae.tile_parallel import shares
        return shares
    except ImportError:
        pass
    # shares() is pure integer arithmetic, so rebuilding a docs figure should not need a
    # torch install. Load that one module, with the imports it never reaches stubbed out.
    for name, attrs in (("torch", {"Tensor": object}), ("torch.distributed", {}),
                        ("distvae", {}),
                        ("distvae.utils", {"ParallelContext": type("Ctx", (), {})})):
        sys.modules[name] = types.ModuleType(name)
        sys.modules[name].__dict__.update(attrs)
    spec = importlib.util.spec_from_file_location(
        "_tile_parallel", os.path.join(ROOT, "distvae", "vae", "tile_parallel.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.shares


SHARES = scheduler()


class Axis:
    """One axis of the grid, in the two numbers the planners are actually given

    Both are absolute output pixels, because that is the interface: `tile_shape_plan` takes
    a window and `tile_overlap_plan` takes an overlap, never a fraction of one. An axis the
    sample already fits is inactive and has to be asked for zero, which is how a full-width
    strip is spelled. The stride is not a control; it is what the pair leaves.

    Everything below is in latent units, since that is what the grid is drawn in. `at` is
    where each tile starts and `extent` how far it reaches once the bound has clipped it.
    `deep` is the overlap, and so both the band a second tile also covers and how far a
    blend reaches into the tile on the far side of a join.
    """

    def __init__(self, window_px, overlap_px):
        self.window_px, self.overlap_px = window_px, overlap_px
        self.window, self.deep = window_px // SCALE_VAE, overlap_px // SCALE_VAE
        stride = self.window - self.deep
        self.at = list(range(0, BOUND, stride))
        self.extent = [min(o + self.window, BOUND) - o for o in self.at]
        self.count = len(self.at)


class Split:
    """A way of dividing the latent, priced in the four terms every row is closed with

    `load` is what each rank ends up decoding, in latent units, and everything else follows
    from it and from the tiles behind it. Row sharding and tiling are both measured through
    here, by the same arithmetic, which is what lets the three rows be compared at all.
    """

    def __init__(self, weight, owner, seams):
        self.weight, self.owner, self.seams = weight, owner, seams
        self.run = [[n for n, who in enumerate(owner) if who == r] for r in range(RANKS)]
        self.load = [sum(weight[n] for n in run) for run in self.run]
        # Activations follow area, so the largest single call is what sets the memory a
        # rank needs, whatever else it goes on to decode afterwards.
        self.held = max(weight) / BOUND ** 2
        # The tiles together cover more latent than there is, and every unit over is a
        # patch of image decoded twice.
        self.work = sum(weight) / BOUND ** 2
        # How far past an even split the heaviest rank lands, which is what the others
        # spend waiting for it.
        self.imbalance = max(self.load) / (sum(self.load) / RANKS) - 1


class Grid(Split):
    """The tiles a window and an overlap leave, and who decodes each of them"""

    # The one column tiling wins; on the other four it loses to the bands it is shaped like.
    # Reads under the header as "syncs: twice", against sharding's "syncs: every layer".
    syncs = "twice"

    def __init__(self, down, across):
        self.down, self.across = down, across
        self.tiles = down.count * across.count
        weight = [d * a for d in down.extent for a in across.extent]
        # A seam is a join between two tiles, which is a pair of neighbours rather than a
        # band of overlap: the grid has one per adjacency on each axis. Corners, where four
        # tiles meet, are left out of the count for the same reason the legend leaves them
        # out, so this is the number of places a blend has to work rather than of blends.
        seams = down.count * (across.count - 1) + across.count * (down.count - 1)
        super().__init__(weight, SHARES(weight, RANKS), seams)
        self.biggest = max(range(self.tiles), key=lambda n: weight[n])


class Bands(Split):
    """Row sharding, put through the same arithmetic so it can be the baseline

    There is no window and no overlap, so the weights are the bands themselves, one to a
    rank. The work comes out at exactly the latent and the seams at none, which is the
    contrast the two rows below are read against.
    """

    syncs = "every layer"

    def __init__(self):
        rows = [BOUND // RANKS + (r < BOUND % RANKS) for r in range(RANKS)]
        super().__init__([r * BOUND for r in rows], list(range(RANKS)), 0)


# Both windows are written as the pair of planner calls that would set them, so the figure
# cannot describe a configuration the API could not be asked for:
#
#     tile_shape_plan(vae, 352, 1408)
#     tile_overlap_plan(vae, 88, 0, sample_shape=(1024, 1024))
#
# Cutting the rows alone. 344 pixels is 43 latent rows and 88 pixels of overlap is 11, which
# steps by 32 and leaves four strips for four ranks with the last clipped to 32. Across, the
# window is asked for at 1408 pixels: past 1366 a window clears the latent in one step
# whatever it overlaps, so the axis is inactive, has to be given zero, and runs full width.
#
# The window is 344 and not a rounder 352 because the last strip is what a reader will
# object to, and it should be the best one available rather than the first one tried. Four
# overlapping strips need all four starts inside 128 rows, so the stride is at least 32 and
# the fourth still has to stop at the bottom: no such split is ever even, and the most the
# short strip can be is 32 against the others' 32 plus the overlap. This window is at that
# floor, which makes the shortfall exactly the overlap and nothing else. A 352 window at the
# same 88 steps by 33 instead, drops the short strip to 29, and idles a rank a third of the
# decode for no more blend than this one gets.
STRIPS = Grid(Axis(344, 88), Axis(1408, 0))

# Cutting both, at 432 by 296 pixels overlapping 72 on each axis. The window is rectangular,
# and deliberately so. A square latent does not imply a square tile: what a rank holds is the
# window's area, but the overlap is paid once per axis and the bounds clip whichever axis
# does not divide. Sweeping every window the planners will land on this latent, this one
# beats the squarest grid on every count at once: 12.2% held against 14.1%, 1.46x the work
# against 1.47x, 0.4% past an even split against 15.1%, and 22 seams against 24. Finer grids
# hold less; what this one does is dominate the square a reader would guess at.
#
# One overlap serves both axes because a seam is a seam: what a reader sees is the thinnest
# blend on the page, so spending more on one axis improves joins that were already the
# better ones. Asking for the diffusers quarter instead would give 120 by 72 here, which
# looks like a per-axis decision and is only a fraction wearing pixels. The window has to be
# re-picked to go with it, though, since the overlap sets the stride and the stride decides
# where the last tile lands: hold 480 by 288 and drop to 72 on both and the last row clips
# to 26 rows instead of 38, taking the imbalance from 0.4% to 8.3%.
TILED = Grid(Axis(432, 72), Axis(296, 72))

SHARDED = Bands()

FONT = "system-ui, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
MONO = "'SF Mono', Menlo, Consolas, monospace"

INK = "#1b1f24"
MUTED = "#6a737d"
# The rule that opens a mode. Darker and longer than the hairline over a costs row, because
# it has to read as the top of a section rather than as one more underline inside one.
EDGE = "#c2c8ce"
# Black is spoken for: it marks what a rank holds. Everything a rank says is this instead,
# light enough to sit under the blocks it crosses rather than on top of them.
SYNC = "#a7b0b8"
RULE = "#e3e6e8"

RANK = [
    ("#dce9f7", "#3d6d99"),
    ("#dff0da", "#4f8b3f"),
    ("#fce8d0", "#bf7a2e"),
    ("#f9d9dc", "#b04a52"),
]
# How far a rank's fill is carried towards white, so that the tiles panel can lay them on
# top of one another and have two deep still read as a fill rather than as ink.
TINT = 0.55

# Two columns: the latent, and the timeline it is decoded on. There is no third for the
# result, because the result is the same picture every way and a panel of it would either
# be blank or, drawn with anything visible on it, overstate what a blend leaves behind.
# What each mode does to the image is said in its heading instead, beside what it does to
# memory and to the interconnect, so all three read side by side.
COL1, PANEL = 40, 120
TRACK = 216
LABEL = 42
# Where a heading's muted tail starts, the same for every row and every window under them,
# so one column of bold runs down the page and one column of grey runs beside it.
TAIL = COL1 + 150
# Set to the longest line of writing, which is what bounds the page now that no panel
# reaches further than the headings do. Measured against a wide fallback rather than the
# font the README will pick, so a substitution loses margin instead of clipping a word.
# The height is not a constant: draw() adds it up from what the rows come to.
W = 680

LANE = 30
# One layer of the sharded decode, and the room after it for whatever that layer syncs. The
# two are sized together so the timelines reach the width the headings above them set,
# rather than stopping short and leaving the right of the page empty.
LAYER = 48
GAP = 18
# The lanes stack to the same height the latent panels are drawn at, which is what lets a
# row read straight across.
TALL = RANKS * LANE
UNIT = PANEL / BOUND
# How far the busiest lane in a row runs, in pixels. Each row is scaled to its own heaviest
# rank rather than to the heaviest in the figure, so all three end their collectives at the
# same x. One clock across the rows would be the more informative drawing, but the rows are
# far apart and the difference between them is under a tenth: at that size a short row reads
# as a rendering fault rather than as a shorter decode, and the work column says it better.
# Within a row the blocks stay proportional, which is what makes the clipped tiles cheap.
SPAN = 5 * (LAYER + GAP)
# The gap the elided layers leave in the row-sharding timeline. Wide enough for a run of
# dots in every lane with the memory bracket's open edge clear of them, since that edge
# lands three short of the collective that closes the row.
ELIDED = 24
# One monospace digit at size 9, the size the blocks are numbered, which is what decides
# whether a block is wide enough to hold its own number.
DIGIT = 5.4

# The columns every row is closed with. Four of them are what a way of splitting the latent
# costs; the fifth is what it buys, and it is here because without it the readout says only
# that tiling is worse, which is true of every column and beside the point.
COSTS = (
    ("peak activations", lambda s: f"{s.held:.0%}"),
    ("work", lambda s: f"{s.work:.2f}×"),
    # A count, except at zero, where the difference is not a small number of seams but a
    # mode that never blends anything and so has none to hide.
    ("seams", lambda s: str(s.seams) if s.seams else "none"),
    ("imbalance", lambda s: f"{s.imbalance:.1%}"),
    ("syncs", lambda s: s.syncs),
)
# Label over value rather than beside it, so a column is as wide as its widest single word
# and five of them fit where four sat before. The stack also stops the readout reading as
# another line of caption, which is what it looked like set on one line.
PITCH = 112
STACK = 15

HATCH = (
    '<pattern id="seam" width="5" height="5" patternUnits="userSpaceOnUse" '
    'patternTransform="rotate(45)">'
    f'<line x1="0" y1="0" x2="0" y2="5" stroke="{INK}" stroke-width="1.5" '
    'opacity="0.3"/></pattern>'
)


out = []


def add(s):
    out.append(s)


# --------------------------------------------------------------------------- primitives


def rect(x, y, w, h, fill, stroke, rx=3, sw=1.2, opacity=None, fill_opacity=None):
    o = f' opacity="{opacity}"' if opacity is not None else ""
    f = f' fill-opacity="{fill_opacity}"' if fill_opacity is not None else ""
    add(
        f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" rx="{rx}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{o}{f}/>'
    )


def text(x, y, s, size=11, fill=INK, anchor="start", weight="400", font=FONT):
    add(
        f'<text x="{x:.2f}" y="{y:.2f}" font-family="{font}" font-size="{size}" '
        f'fill="{fill}" text-anchor="{anchor}" font-weight="{weight}">{s}</text>'
    )


def note(x, y, s):
    """A muted caption line, which is most of the writing on the page"""
    text(x, y, s, size=10, fill=MUTED)


def tag(x, y, s):
    """The small monospace label that names a mark rather than describes it"""
    text(x, y, s, size=8.5, fill=MUTED, anchor="middle", font=MONO)


def line(x1, y1, x2, y2, colour, width=1.2, extra=""):
    add(
        f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
        f'stroke="{colour}" stroke-width="{width}"{extra}/>'
    )


def flow(x1, y, x2):
    line(x1, y, x2, y, MUTED, 1.4, ' marker-end="url(#fwd)"')


def seam(x, y, w, h):
    """Where tiles overlap, and so where the blend writes: the same band, twice over"""
    rect(x, y, w, h, "url(#seam)", "none", rx=0, sw=0)


def collective(x, top, bottom):
    """A bar across every lane: a call no rank leaves before the others arrive"""
    rect(x, top, 5, bottom - top, SYNC, "none", rx=2, sw=0)


def halo(x, boundaries):
    """Short arrows across each internal lane boundary: neighbours only, not the group"""
    for yb in boundaries:
        line(x, yb - 9, x, yb + 9, SYNC, 1.3,
             ' marker-end="url(#down)" marker-start="url(#up)"')


def peak(x, y, w, h, open_right=False):
    """What one rank holds at once, which is what sets the memory it needs

    Left open where the layers it spans are themselves elided, so the edge is dotted for
    the same reason the dots beside it are: the drawing stops there, the decode does not.
    """
    if not open_right:
        rect(x, y, w, h, "none", INK, rx=1, sw=2.6)
        return
    add(
        f'<path d="M{x + w:.2f},{y:.2f} H{x:.2f} V{y + h:.2f} H{x + w:.2f}" fill="none" '
        f'stroke="{INK}" stroke-width="2.6"/>'
    )
    line(x + w, y, x + w, y + h, INK, 2.6,
         ' stroke-linecap="round" stroke-dasharray="0.1 5"')


def head(mid, colour, width=5, back=False):
    d = "M10,0 L0,5 L10,10 z" if back else "M0,0 L10,5 L0,10 z"
    ref = 1 if back else 9
    return (
        f'<marker id="{mid}" viewBox="0 0 10 10" refX="{ref}" refY="5" '
        f'markerWidth="{width}" markerHeight="{width}" orient="auto">'
        f'<path d="{d}" fill="{colour}"/></marker>'
    )


def lanes(y):
    return [y + r * LANE for r in range(RANKS)]


def carries_on(x, y):
    """The layers the drawing stops short of, marked in every lane rather than between them

    What row sharding costs is that the block-sync-block pattern to the left repeats for the
    whole depth of the decoder, so the elision has to read as every rank going on doing that;
    a single glyph between the lanes reads as one gap in the middle instead.
    """
    for ly, (_, stroke) in zip(lanes(y), RANK):
        for step in range(3):
            add(
                f'<circle cx="{x + step * 6:.2f}" cy="{ly + LANE / 2:.2f}" r="2.1" '
                f'fill="{stroke}"/>'
            )


# ------------------------------------------------------------------------------- blocks


def heading(y, n, title, tail, headline, *notes):
    """A mode's header: what it is, what it costs, and the detail under that

    Numbered because the two modes are alternatives and a reader arriving at the top of a
    long page can otherwise take them for the two halves of one pipeline. The number counts
    the modes and nothing else, which is why the stages below are no longer numbered too.

    Returns the y the panels below it start at, so a row that grows a line pushes the page
    down instead of needing every coordinate under it re-tuned.
    """
    text(COL1, y, f"{n}. {title}", size=13, weight="700")
    text(TAIL, y, tail, size=11, fill=MUTED)
    text(COL1, y + 18, headline, size=11, weight="600")
    for k, line_ in enumerate(notes):
        note(COL1, y + 34 + k * 16, line_)
    return y + 40 + len(notes) * 16


def divider(y):
    """The rule that opens a mode, and the only thing on the page drawn edge to edge

    The two modes get one and the windows inside tile distribution do not, which is what
    keeps a section from reading as a row: "Row sharding" and "Cut the rows only" are set a
    point and a half apart, and on their own that is not enough to rank them.
    """
    line(COL1, y, W - COL1, y, EDGE, 1.4)
    return y


def caption(x, y, title, *lines):
    """A stage and what it says, under whichever of the two panels it belongs to

    Set at the panel's own left edge, and read in the order the arrow between the panels
    already points, so neither stage needs a number to say where it comes.
    """
    text(x, y, title, size=10.5, weight="600")
    for k, line_ in enumerate(lines):
        note(x, y + 16 + k * 14, line_)
    return y + 16 + len(lines) * 14


def costs(y, split):
    """What a row costs and what it buys, in the columns every other row uses

    Drawn from the same attributes whichever way the latent was divided, so a reader
    comparing the three rows is comparing arithmetic rather than prose. Returns the baseline
    of the values, since that is what the captions below have to clear.
    """
    line(COL1, y - 12, COL1 + (len(COSTS) - 1) * PITCH + 66, y - 12, RULE)
    for k, (name, show) in enumerate(COSTS):
        text(COL1 + k * PITCH, y, name, size=9.5, fill=MUTED)
        text(COL1 + k * PITCH, y + STACK, show(split), size=11.5, weight="600")
    return y + STACK


# --------------------------------------------------------------------------- the latent


def box(x0, y0, grid, n):
    """Tile n's origin on the page and the window the bounds leave it"""
    i, j = divmod(n, grid.across.count)
    return (x0 + grid.across.at[j] * UNIT, y0 + grid.down.at[i] * UNIT,
            grid.across.extent[j] * UNIT, grid.down.extent[i] * UNIT)


def tiles(x0, y0, grid):
    """Every tile at its true extent, so the overlaps are the drawing's own

    Fills are transparent and lie on top of one another, so a band two tiles cover comes out
    twice as deep, and the four-way corners deeper still. That is the redundant work.
    """
    for n, who in enumerate(grid.owner):
        rect(*box(x0, y0, grid, n), RANK[who][0], "none", rx=0, sw=0, fill_opacity=TINT)
    for n, who in enumerate(grid.owner):
        rect(*box(x0, y0, grid, n), "none", RANK[who][1], rx=1, sw=1)
    for n in range(grid.tiles):
        bx, by, bw, bh = box(x0, y0, grid, n)
        text(bx + bw / 2, by + bh / 2 + 3, str(n), anchor="middle", size=8.5, font=MONO)


def overlaps(x0, y0, grid):
    """The bands a second tile also covers, and so where the blend writes

    Every band is the overlap deep, because only the last tile on an axis is ever clipped
    and nothing starts after it. An axis the window already spans is not cut at all, and so
    contributes nothing here, which is what the strip row's three clean joins come from.
    """
    for k in range(1, grid.down.count):
        seam(x0, y0 + grid.down.at[k] * UNIT, PANEL, grid.down.deep * UNIT)
    for k in range(1, grid.across.count):
        seam(x0 + grid.across.at[k] * UNIT, y0, grid.across.deep * UNIT, PANEL)


# --------------------------------------------------------------------------------- rows


def sharding(y):
    """Row sharding: one decoder call, split into bands, syncing all the way down"""
    y = heading(
        y, 1, "Row sharding",
        f"one decoder call, split into {word(RANKS)} bands of {BOUND // RANKS} latent rows",
        # A sync rather than a collective, because the convolutions swap halos with their
        # neighbours and only the norms reduce across the group, and the legend draws those
        # as two different things.
        "Peak memory is rank-bound and every layer syncs, so the interconnect can be the "
        "bottleneck.",
        # Nothing to pick here, which is the contrast the tiling header is written against.
        "There is nothing to choose here, since the band is the latent divided by the GPU "
        "count.",
        # The third thing a reader is choosing between, said where the other two are said.
        # The caveat is only that a reduction adds its terms in a different order, so the
        # last bits move. Naming that mechanism costs a clause and buys nothing: what the
        # reader is weighing is this line against the tiling row's, and the contrast is
        # between rounding they will never see and blocky colour they might.
        "The image is what a single GPU would produce, apart from floating-point rounding.",
    )
    bottom = y + TALL

    for r, ly in enumerate(lanes(y)):
        rect(COL1, ly, PANEL, LANE, *RANK[r], rx=0)
        text(COL1 + PANEL / 2, ly + LANE / 2 + 4, f"rank {r}", anchor="middle", size=10.5)
    peak(COL1, y, PANEL, LANE)

    flow(COL1 + PANEL + 12, y + TALL / 2, TRACK - 10)

    for r, ly in enumerate(lanes(y)):
        note(TRACK, ly + LANE / 2 + 4, f"rank {r}")
    first = x = TRACK + LABEL
    # Every rank runs the same layer at the same moment, so one header names them all. What
    # follows a layer is the layer's own business: a convolution wants rows from its
    # neighbours, a norm wants a statistic from everybody.
    for layer in ("conv", "norm", "conv", "norm", "conv"):
        for r, ly in enumerate(lanes(y)):
            rect(x, ly + 4, LAYER, LANE - 8, *RANK[r], rx=2)
        tag(x + LAYER / 2, y - 6, layer)
        x += LAYER
        if layer == "conv":
            halo(x + GAP / 2, lanes(y)[1:])
        else:
            collective(x + GAP / 2 - 2.5, y, bottom)
        x += GAP
    carries_on(x + 3, y)
    x += ELIDED
    collective(x, y, bottom)
    # The band is one allocation and it is live the whole way down: the axis here is time,
    # so what the mark spans is how long a rank holds it, not how much it is holding.
    peak(first - 3, y + 1.5, x - first, LANE - 3, open_right=True)
    # Named for what it carries, as the tile rows' are: all the bars are the same gather,
    # and the count is the difference worth reading.
    tag(x + 2.5, bottom + 12, "image")

    cap = costs(bottom + 32, SHARDED) + 26
    return max(
        # The one fact the strip row below is written against: bands meet, tiles overlap.
        # Short because the left caption column ends where the right one starts, at TRACK.
        caption(COL1, cap, "Split the rows", "The bands never overlap."),
        # One line, not two: how often it syncs is a column now, so this is left to say
        # only what a sync is, which the legend then splits into its two marks.
        caption(TRACK, cap, "Decode in lockstep",
                "Convolutions swap edge rows, and norms reduce across all four ranks."),
    )


def window(y, grid, name, tail, aside, first, *second):
    """One tiling window: the grid it leaves, the lanes it runs, and what the pair cost

    Both windows come through here, so the only thing separating the two rows below the
    tiling heading is the two numbers each was built from.
    """
    text(COL1, y, name, size=11.5, weight="700")
    text(TAIL, y, tail, size=10.5, fill=MUTED)
    note(COL1, y + 17, aside)
    y += 32
    bottom = y + TALL

    tiles(COL1, y, grid)
    overlaps(COL1, y, grid)
    peak(*box(COL1, y, grid, grid.biggest))

    flow(COL1 + PANEL + 12, y + TALL / 2, TRACK - 10)

    scale = SPAN / max(grid.load)
    start = TRACK + LABEL
    last = start + max(grid.load) * scale

    # Idle belongs to a lane and not to the row: three of the four strips run the whole
    # length, so one band across every lane would say they were waiting too. Each lane gets
    # its own tail instead, from where its work runs out to where the last rank lands.
    for r, ly in enumerate(lanes(y)):
        note(TRACK, ly + LANE / 2 + 4, f"rank {r}")
        at = start
        for n in grid.run[r]:
            width = grid.weight[n] * scale
            rect(at + 1, ly + 4, width - 2, LANE - 8, *RANK[r], rx=2)
            # Checked against the width of this number rather than assumed from the full
            # blocks, so a clipped corner tile is either named like the rest or left blank
            # instead of overrunning its block.
            if width - 2 >= DIGIT * len(str(n)) + 3:
                text(at + width / 2, ly + LANE / 2 + 3.5, str(n), anchor="middle",
                     size=9, font=MONO)
            at += width
        if last - at >= 1:
            rect(at, ly + 4, last - at, LANE - 8, MUTED, "none", rx=1, sw=0, opacity=0.16)

    # Named inside the tail rather than above the row, so the word sits in the lane it is
    # true of, and only where the tail is wide enough to hold it. The grid leaves nothing
    # to label, which is the comparison: its imbalance is a column, not a picture.
    waiting = min(range(RANKS), key=lambda r: grid.load[r])
    soonest = start + grid.load[waiting] * scale
    if last - soonest >= 24:
        tag((soonest + last) / 2, lanes(y)[waiting] + LANE / 2 + 3, "idle")

    # One tile at a time, and the memory follows the largest of them rather than the first.
    heaviest = max(grid.run[0], key=lambda n: grid.weight[n])
    before = sum(grid.weight[n] for n in grid.run[0][:grid.run[0].index(heaviest)])
    peak(start + before * scale - 1.5, lanes(y)[0] + 1.5,
         grid.weight[heaviest] * scale + 3, LANE - 3)

    # Nothing crosses between the ranks until every tile is decoded.
    collective(last + 10, y, bottom)
    collective(last + 24, y, bottom)
    tag(last + 26, bottom + 12, "edges, image")

    cap = costs(bottom + 32, grid) + 26
    return max(caption(COL1, cap, first), caption(TRACK, cap, *second))


def tiling(y):
    """Tile distribution: a window's worth per call, dealt out, gathered twice"""
    y = heading(
        y, 2, "Tile distribution", f"the same {word(RANKS)} GPUs, at two windows",
        "Peak memory is tile-bound and the decode needs only two collectives, but it "
        "repeats more work.",
        # Said plainly, because the row otherwise reads as a default. Named in the terms the
        # planners take, too: a window and an absolute overlap, not a fraction of a window.
        "Window and overlap are yours to set in output pixels, so tune them to your VAE "
        "and your GPUs.",
        # The figure shows two windows and a reader will take the better-looking one for
        # advice, so the disclaimer has to be here rather than left to the docs.
        "The two windows below are worked examples, chosen to show the trade rather than "
        "to be copied.",
        # The honest summary of that trade, and the one thing the columns cannot show: a
        # full-width strip is one contiguous span of a row-major tensor, where a grid's
        # tile is a stride through every row it touches.
        "Full-width strips overlap less and stay contiguous in row-major memory, while a "
        "grid holds less at once.",
        # Against the sharding row's line in the same place: what the choice costs the
        # image. The seam a blend can hide; the norms it cannot, since a tile's are its own
        # contents and nothing else, which is why a window can be too small rather than
        # merely slow.
        "The image is close but not exact: a blend hides seams, but norms over too small a "
        "tile can leave the colour blocky.",
    )

    y = window(
        y + 4, STRIPS,
        "Cut the rows only",
        f"{STRIPS.down.window_px} px tall overlapping {STRIPS.down.overlap_px} px, "
        "full width",
        f"{word(STRIPS.tiles).capitalize()} strips, one per rank, have the same shape as "
        "the bands above, but they overlap and nothing syncs until the end.",
        "Cut and overlap the rows",
        "One call each, and one rank waits",
        # Was "nothing to deal out", which only meant anything to a reader who had already
        # read the grid row below and knew there was a scheduler to have nothing to do.
        "With one strip per rank, there is nothing for the scheduler to decide.",
        f"The last strip is {STRIPS.down.extent[-1]} latent rows where the others are "
        f"{STRIPS.down.window}.",
        # The line that answers the reader who suspects a window was picked to flatter the
        # grid below. Not that no split does better, since a thinner blend plainly does:
        # that at this depth of blend none does, because the gap is the blend.
        "That shortfall is exactly the overlap, so closing it would thin the blend.",
    )

    heavy = max(range(RANKS), key=lambda r: len(TILED.run[r]))
    light = min(range(RANKS), key=lambda r: len(TILED.run[r]))
    return window(
        y + 26, TILED,
        "Cut both axes",
        f"{TILED.down.window_px} × {TILED.across.window_px} px overlapping "
        # One number when one number was asked for, so the row does not imply a per-axis
        # decision that was not made.
        + (f"{TILED.down.overlap_px} px on both axes"
           if TILED.down.overlap_px == TILED.across.overlap_px
           else f"{TILED.down.overlap_px} × {TILED.across.overlap_px} px"),
        f"{word(TILED.tiles).capitalize()} tiles across {word(RANKS)} ranks let the load "
        "be levelled, and a rank now holds a window rather than a strip.",
        "Deal the tiles out",
        "Each rank decodes its own, in turn",
        # A run is the cheap shape to blend but a coarse one to balance, so the scheduler
        # moves single tiles off it, which is why two lanes hold tiles from either end.
        "Each rank starts with a contiguous run, then single tiles move to level it.",
        f"Rank {heavy} decodes {word(len(TILED.run[heavy]))} tiles to rank {light}'s "
        f"{word(len(TILED.run[light]))}, and they still finish together.",
    )


def legend(y):
    """What the marks mean, in one row, read before the rows that use them

    Four marks and no swatch for the blend, because the blend is not drawn anywhere. What a
    good one leaves is too slight to put on a page at this size without overstating it, so
    the tiling heading says it in words instead.

    The four are spaced off the width of the longest label in a fallback font, which is the
    widest the row can come out, so the line holds together whichever font renders it.
    """
    collective(COL1, y - 9, y + 5)
    note(COL1 + 14, y + 2, "collective: every rank waits")
    halo(COL1 + 191, [y - 2])
    note(COL1 + 203, y + 2, "halo swap: neighbours only")
    peak(COL1 + 380, y - 8, 14, 13)
    note(COL1 + 400, y + 2, "held at once")
    seam(COL1 + 502, y - 8, 10, 13)
    # Two tiles at most joins, four where the corners meet, so the count is left out.
    note(COL1 + 518, y + 2, "tile overlap")
    return y + 5


def draw():
    text(COL1, 30, "DistVAE parallelism", size=17, weight="700")
    # The example every row runs on, said once so no header has to carry it. On its own
    # line rather than trailing the title, since a fallback font only ever sets the bold
    # wider and there is nothing to the right of it to absorb that.
    text(COL1, 49, f"The two modes below each decode a {BOUND * SCALE_VAE} × "
                   f"{BOUND * SCALE_VAE} image from a {BOUND} × {BOUND} latent on "
                   f"{word(RANKS)} GPUs.", size=11, fill=MUTED)
    # What the two numbers below are counting. A figure this tall is met one screen at a
    # time, so the word alternatives has to appear at the top: numbered headings alone
    # would as readily be the halves of a pipeline, and the second half is where the page
    # ends.
    text(COL1, 65, "They are alternatives, and both are priced in the same five columns.",
         size=11, fill=MUTED)

    # Above the rows rather than under them, so the marks are named before they are met,
    # and above the first divider, so they read as belonging to the page and not to row
    # sharding in particular.
    y = legend(88)
    y = sharding(divider(y + 20) + 26)
    y = tiling(divider(y + 32) + 26)
    height = round(y + 14)

    front = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{height}" '
        f'viewBox="0 0 {W} {height}">',
        "<defs>" + HATCH + head("fwd", MUTED, width=6)
        + head("down", SYNC) + head("up", SYNC, back=True) + "</defs>",
        f'<rect width="{W}" height="{height}" fill="#ffffff"/>',
    ]
    return "\n".join(front + out + ["</svg>"])


def rasterise(svg):
    """Write the PNG beside the SVG, or say why there is no new one"""
    try:
        import cairosvg
    except ImportError:
        return "no cairosvg: figure.png left as it was"
    cairosvg.svg2png(bytestring=svg.encode(), write_to=PNG, scale=RASTER)
    return f"wrote {PNG}"


if __name__ == "__main__":
    svg = draw()
    with open(OUT, "w") as handle:
        handle.write(svg)
    print(f"wrote {OUT}")
    print(rasterise(svg))
