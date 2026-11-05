"""Shared layout of the diagnostic PDFs: portrait A4 pages of titled sections.

As GRAVITY's reports (gravi_visual): a title on the first page, then sections
flowing down the pages, each a bold heading, a short explanation and its content
(plots or a table); "Page n / N" and the product file at the foot of every page.
A section is drawn into a matplotlib SubFigure, without pyplot, so no GUI
backend is involved.
"""
import textwrap
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matplotlib import rc_context
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

from planetes_pipe.instrument import layout

A4 = (8.27, 11.69)                 # inches, portrait
TOP, BOTTOM, SIDE = 0.85, 0.55, 0.6  # page margins for title, footer and sides, inches
HEADING = 0.5                      # a section's title and explanation, inches
ABCD_COLORS = ("C0", "C1", "C2", "C3")  # A, C, B, D, in that order on every page
STYLE = {"font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "legend.fontsize": 6,
         "xtick.labelsize": 6, "ytick.labelsize": 6, "lines.linewidth": 0.9}


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


def baseline_grid(fig, **kwargs):
    """Six axes, one per baseline in layout order (2 rows x 3 columns)."""
    return np.asarray(fig.subplots(2, 3, **kwargs)).ravel()


def table(fig, cells, columns, rows, shade=()):
    """A text table filling a subfigure, row labels as its first column.

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
    for row in range(len(rows) + header):
        drawn[row, 0].set_text_props(ha="left", weight="bold" if row >= header else "normal")
        drawn[row, 0].PAD = 0.03
    for column in range(width if header else 0):
        drawn[0, column].set_text_props(weight="bold")
        drawn[0, column].set_facecolor("0.92")
    for (row, column), color in shade:
        drawn[row + header, column + 1].set_facecolor(color)
    return drawn


def table_height(n_rows: int) -> float:
    return 0.22 * (n_rows + 1) + 0.15


def summary_section(path, header, rows) -> Section:
    """The product's identity, from its header: rows of (label, value)."""
    cells = [[str(value)] for _, value in rows]
    return Section(f"Product: {Path(path).name}",
                   lambda fig: table(fig, cells, None, [label for label, _ in rows]),
                   table_height(len(rows) - 1), "From the product header.")


def _pack(sections, usable: float):
    """Sections in order onto pages: a new page when the next one does not fit."""
    pages, current, used = [], [], 0.0
    for section in sections:
        need = HEADING + section.height
        if current and used + need > usable:
            pages.append(current)
            current, used = [], 0.0
        current.append(section)
        used += need
    if current:
        pages.append(current)
    return pages


def write_pdf(path, title: str, sections, footer: str = "") -> Path:
    """Write the sections as a portrait A4 PDF; never overwrites."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    usable = A4[1] - TOP - BOTTOM
    pages = _pack(list(sections), usable)
    with rc_context(STYLE), PdfPages(path) as pdf:
        for number, group in enumerate(pages, 1):
            fig = Figure(figsize=A4, layout="constrained")
            fig.get_layout_engine().set(w_pad=0.04, h_pad=0.02, hspace=0.0, wspace=0.02)
            if number == 1:
                fig.text(0.5, 1 - 0.42 / A4[1], title, ha="center", va="center", fontsize=13,
                         weight="bold")
            fig.text(SIDE / A4[0], 0.3 / A4[1], f"Page {number} / {len(pages)}    {footer}",
                     fontsize=7, color="0.3")
            content = fig.subfigures(1, 3, width_ratios=[SIDE, A4[0] - 2 * SIDE, SIDE])[1]
            heights = [HEADING + s.height for s in group]
            ratios = [TOP, *heights, max(usable - sum(heights), 0.01) + BOTTOM]
            areas = content.subfigures(len(ratios), 1, height_ratios=ratios)
            for area, section in zip(areas[1:-1], group):
                head, body = area.subfigures(2, 1, height_ratios=[HEADING, section.height])
                head.text(0.0, 0.72, section.title, fontsize=10, weight="bold", va="center")
                explanation = textwrap.fill(section.explanation, 140)
                head.text(0.0, 0.3, explanation, fontsize=7, color="0.3", va="center")
                section.draw(body)
            pdf.savefig(fig)
    return path
