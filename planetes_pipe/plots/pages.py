"""Shared layout and style of the diagnostic PDFs: portrait A4 pages of titled sections.

As GRAVITY's reports (gravi_visual): a title on the first page, then sections flowing
down the pages, each a bold heading, a short explanation and its content (plots or
a table); "Page n / N" and the product file at the foot of every page.

The style is a plain scientific one: narrow margins, thin frames, inward ticks on
all sides, shared axes with tick labels only on the outer panels and axis labels
once per grid, one legend per section above its panels, and ruled tables without
vertical lines. A section is drawn into a matplotlib SubFigure, without pyplot, so
no GUI backend is involved.
"""
import logging
import textwrap
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from cycler import cycler
from matplotlib import rc_context
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.collections import Collection
from matplotlib.lines import Line2D

from planetes_pipe.instrument import layout

log = logging.getLogger(__name__)

A4 = (8.27, 11.69)                         # inches, portrait
TOP, BOTTOM, SIDE = 0.6, 0.55, 0.45        # page margins, inches (sides ~1.1 cm)
TITLE_Y, FOOT_Y = 0.38, 0.3               # first-page title and footer, inches from the edge
# Dense plots are drawn as images in the PDF: point sets (markers, one per frame) above
# RASTER_MARKERS points, lines above RASTER_POINTS (spectra, ~520 points, stay vector). One
# vector path per point made a report ~0.7 MB. RASTER_DPI is matplotlib's default figure dpi,
# at which the waterfall images were always resampled.
RASTER_MARKERS, RASTER_POINTS, RASTER_DPI = 100, 600, 100
TITLE_HEIGHT, LINE_HEIGHT = 0.27, 0.14     # a section's title, and each line of its explanation, inches
WRAP = 150                                 # characters per explanation line
GRID_HEIGHT = 3.9                          # six baseline panels, inches
# Colours of the A, C, B, D outputs (in that order), then of other series.
ABCD_COLORS = ("#0C5DA5", "#FF9500", "#00B945", "#FF2C00")
COLORS = (*ABCD_COLORS, "#845B97", "#474747", "#9E9E9E")
FLAG, PASS, MARK = "#F4CCCC", "#D9EAD3", "#E8EAF6"  # table shades: failed, passed, highlighted
GUIDE = {"color": "0.55", "lw": 0.6, "ls": "--"}    # reference lines (limits, zero)
STYLE = {
    "font.size": 7.5, "axes.titlesize": 7.5, "axes.labelsize": 7.5, "legend.fontsize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "figure.labelsize": 7.5,
    "axes.linewidth": 0.6, "lines.linewidth": 0.8, "lines.markersize": 2,
    "patch.linewidth": 0.4, "axes.prop_cycle": cycler(color=COLORS),
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "xtick.minor.visible": True, "ytick.minor.visible": True,
    "xtick.major.size": 3, "ytick.major.size": 3, "xtick.minor.size": 1.5,
    "ytick.minor.size": 1.5, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.minor.width": 0.4, "ytick.minor.width": 0.4, "legend.frameon": False,
    "legend.handlelength": 1.8, "legend.columnspacing": 1.2, "axes.titlepad": 3,
    "image.cmap": "inferno", "axes.formatter.useoffset": False,
}


@dataclass(frozen=True)
class Section:
    """One titled block of a report: draw(subfigure) fills a content area `height` inches tall."""

    title: str
    draw: Callable
    height: float = 3.0
    explanation: str = ""


def baseline_name(baseline: int) -> str:
    first, second = layout.baselines[baseline]
    return f"S{first}–S{second}"


def panels(fig, rows, columns, xlabel=None, ylabel=None, *, sharex=True, sharey=False, **kwargs):
    """A grid of panels, flattened; shared axes keep tick labels only on the outer panels.

    xlabel and ylabel are written once for the whole grid.
    """
    axes = np.asarray(fig.subplots(rows, columns, sharex=sharex, sharey=sharey, squeeze=False,
                                   **kwargs))
    for (row, column), ax in np.ndenumerate(axes):
        if sharex and row < rows - 1:
            ax.tick_params(labelbottom=False)
        if sharey and column > 0:
            ax.tick_params(labelleft=False)
    if xlabel:
        fig.supxlabel(xlabel)
    if ylabel:
        fig.supylabel(ylabel)
    return axes.ravel()


def baseline_panels(fig, xlabel, ylabel, **kwargs):
    """Six panels, one per baseline in layout order, each named in its corner."""
    axes = panels(fig, 2, 3, xlabel, ylabel, **kwargs)
    for baseline, ax in enumerate(axes):
        name(ax, baseline_name(baseline))
    return axes


def name(ax, text: str, loc: str = "upper left") -> None:
    """A panel's name, inside one of its corners."""
    top = loc.startswith("upper")
    left = loc.endswith("left")
    ax.text(0.04 if left else 0.96, 0.94 if top else 0.06, text, transform=ax.transAxes,
            va="top" if top else "bottom", ha="left" if left else "right", fontsize=7, zorder=5,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6, "alpha": 0.85})


def legend(fig, entries, ncol=None) -> None:
    """One legend for a whole section, above its panels; entries: (label, Line2D properties)."""
    handles = [Line2D([], [], **style) for _, style in entries]
    fig.legend(handles, [text for text, _ in entries], loc="outside upper center",
               ncol=ncol or len(entries))


def abcd_entries(**style):
    return [(label, {"color": color, **style}) for label, color in zip(layout.abcd_order, ABCD_COLORS)]


def table(fig, cells, columns, rows, shade=()):
    """A ruled table filling a subfigure: horizontal rules only, row labels as first column.

    columns None draws no header row. shade holds ((row, column), color) for data
    cells, counted without the labels.
    """
    ax = fig.subplots()
    ax.axis("off")
    labelled = [[label, *row] for label, row in zip(rows, cells)]
    width = len(labelled[0])
    header = 0 if columns is None else 1
    drawn = ax.table(cellText=labelled, colLabels=None if columns is None else ["", *columns],
                     loc="center", cellLoc="center", bbox=[0, 0, 1, 1],
                     colWidths=[1.6 / (width + 0.6)] + [1 / (width + 0.6)] * (width - 1))
    drawn.auto_set_font_size(False)
    drawn.set_fontsize(7)
    for (row, column), cell in drawn.get_celld().items():
        cell.set_edgecolor("none")  # closed but invisible edges, so shading still fills
        if column == 0:
            cell.set_text_props(ha="left")
            cell.PAD = 0.02
        if row < header:
            cell.set_text_props(weight="bold")
    for (row, column), color in shade:
        drawn[row + header, column + 1].set_facecolor(color)
    n_rows = len(rows) + header  # rows share the height equally: rules at their borders
    for edge, width in ((0, 1.0), (header, 0.5), (n_rows, 1.0)):
        if edge or width == 1.0:
            y = 1 - edge / n_rows
            ax.plot([0, 1], [y, y], transform=ax.transAxes, color="black", lw=width, clip_on=False)
    return drawn


def table_height(n_rows: int) -> float:
    return 0.2 * (n_rows + 1) + 0.1


def summary_section(path, header, rows) -> Section:
    """The product's identity, from its header: rows of (label, value)."""
    cells = [[str(value)] for _, value in rows]
    return Section(f"Product {Path(path).name}",
                   lambda fig: table(fig, cells, None, [label for label, _ in rows]),
                   table_height(len(rows) - 1), "Identity of the product, from its header.")


def _wrap(text: str, width: int = WRAP) -> str:
    """Wrap an explanation at the given width without breaking a $formula$."""
    parts = text.split("$")
    parts[1::2] = [part.replace(" ", "\0") for part in parts[1::2]]
    return textwrap.fill("$".join(parts), width).replace("\0", " ")


def _heading(section) -> float:
    """Height [inches] of a section's title and its wrapped explanation."""
    lines = len(_wrap(section.explanation).splitlines()) if section.explanation else 0
    return TITLE_HEIGHT + LINE_HEIGHT * lines + 0.04


def _pack(sections, usable: float):
    """Sections in order onto pages: a new page when the next one does not fit."""
    pages, current, used = [], [], 0.0
    for section in sections:
        need = _heading(section) + section.height
        if current and used + need > usable:
            pages.append(current)
            current, used = [], 0.0
        current.append(section)
        used += need
    if current:
        pages.append(current)
    return pages


def _rasterize_dense(fig) -> None:
    """Draw dense point sets and lines as images (RASTER_MARKERS, RASTER_POINTS)."""
    for artist in fig.findobj(lambda a: isinstance(a, (Line2D, Collection))):
        if isinstance(artist, Line2D):
            n_points = len(artist.get_xdata())
            markers = artist.get_marker() not in (None, "", "None", " ") and artist.get_linestyle() == "None"
        else:
            n_points, markers = len(artist.get_offsets()), True
        if n_points > (RASTER_MARKERS if markers else RASTER_POINTS):
            artist.set_rasterized(True)


def write_pdf(path, title: str, sections, footer: str = "") -> Path:
    """Write the sections as a portrait A4 PDF; never overwrites."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    usable = A4[1] - TOP - BOTTOM
    pages = _pack(list(sections), usable)
    foot = FOOT_Y / A4[1]
    with rc_context(STYLE), PdfPages(path) as pdf:
        for number, group in enumerate(pages, 1):
            fig = Figure(figsize=A4, layout="constrained")
            fig.get_layout_engine().set(w_pad=0.03, h_pad=0.03, hspace=0.03, wspace=0.03)
            if number == 1:
                fig.text(0.5, 1 - TITLE_Y / A4[1], title, ha="center", va="center", fontsize=12,
                         weight="bold")
            fig.text(SIDE / A4[0], foot, footer, fontsize=6.5, color="0.4")
            fig.text(1 - SIDE / A4[0], foot, f"Page {number} / {len(pages)}", fontsize=6.5,
                     color="0.4", ha="right")
            content = fig.subfigures(1, 3, width_ratios=[SIDE, A4[0] - 2 * SIDE, SIDE])[1]
            heights = [_heading(s) + s.height for s in group]
            ratios = [TOP, *heights, max(usable - sum(heights), 0.01) + BOTTOM]
            areas = content.subfigures(len(ratios), 1, height_ratios=ratios)
            for area, section in zip(areas[1:-1], group):
                heading = _heading(section)
                head, body = area.subfigures(2, 1, height_ratios=[heading, section.height])
                head.text(0.0, 1 - 0.13 / heading, section.title, fontsize=9.5, weight="bold",
                          va="center")
                head.text(0.0, 1 - TITLE_HEIGHT / heading, _wrap(section.explanation), fontsize=7,
                          color="0.4", va="top", linespacing=1.5)
                section.draw(body)
            _rasterize_dense(fig)
            pdf.savefig(fig, dpi=RASTER_DPI)
    log.info("wrote %s (%d sections, %d page%s)", path.name, sum(map(len, pages)), len(pages),
             "" if len(pages) == 1 else "s")
    return path
