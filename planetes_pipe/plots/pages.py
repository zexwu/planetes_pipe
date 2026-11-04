"""Shared helpers for the diagnostic PDFs: page layout, colours, writing.

Pages are matplotlib Figures made without pyplot, so no GUI backend is involved
and nothing needs closing. A module's pages(...) yields its Figures; write_pdf
collects them into one PDF.
"""
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

from planetes_pipe.instrument import layout

ABCD_COLORS = ("C0", "C1", "C2", "C3")  # A, C, B, D, in that order on every page
PAGE_STYLE = {"font.size": 9, "figure.figsize": (11.7, 8.3)}  # A4 landscape


def page(**kwargs) -> Figure:
    """A new A4-landscape page."""
    return Figure(figsize=kwargs.pop("figsize", PAGE_STYLE["figure.figsize"]), **kwargs)


def finish(fig: Figure, title: str, top: float = 0.95) -> Figure:
    """Give a page its title and lay it out."""
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, top))
    return fig


def baseline_name(baseline: int) -> str:
    first, second = layout.baselines[baseline]
    return f"S{first}–S{second}"


def baseline_grid(fig: Figure, **kwargs):
    """Six axes, one per baseline in layout order (2 rows x 3 columns)."""
    return np.asarray(fig.subplots(2, 3, **kwargs)).ravel()


def table(ax, cells, columns, rows, shade=()):
    """A text table on an empty axes; shade holds (row, column) cells to highlight."""
    ax.axis("off")
    drawn = ax.table(cellText=cells, colLabels=columns, rowLabels=rows, loc="center")
    drawn.scale(1, 1.5)
    for (row, column), color in shade:
        drawn[row + 1, column].set_facecolor(color)
    return drawn


def write_pdf(path, figures) -> Path:
    """Write the Figures (an iterable, e.g. a pages() generator) into a new PDF."""
    import matplotlib

    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with matplotlib.rc_context({"font.size": PAGE_STYLE["font.size"]}), PdfPages(path) as pdf:
        for fig in figures:
            pdf.savefig(fig)
    return path
