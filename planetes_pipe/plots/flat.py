"""FLAT pages: the flat image with its traces, trace centres and widths, telescope fluxes."""
import numpy as np

from planetes_pipe.instrument import layout
from planetes_pipe.plots.pages import finish, page
from planetes_pipe.steps.flat import Flat


def _x_range(flat: Flat, output: int) -> slice:
    """Columns where an output's trace was fitted (all columns if not recorded)."""
    if flat.trace_x_range is None:
        return slice(None)
    first, last = flat.trace_x_range[output]
    return slice(int(first), int(last) + 1)


def pages(flat: Flat):
    """Pages for one FLAT."""
    n_outputs, width = flat.trace_y_pixels.shape
    x = np.arange(width)

    fig = page()
    ax = fig.subplots()
    low, high = np.percentile(flat.flat_map_adu, [1, 99.5])
    shown = ax.imshow(flat.flat_map_adu, origin="lower", aspect="auto", cmap="inferno",
                      vmin=low, vmax=high)
    fig.colorbar(shown, ax=ax, label="ADU")
    for output in range(n_outputs):
        columns = _x_range(flat, output)
        ax.plot(x[columns], flat.trace_y_pixels[output, columns], color="cyan", lw=0.5)
        ax.text(x[columns][-1] + 5, flat.trace_y_pixels[output, columns][-1], str(output),
                color="cyan", fontsize=6, va="center")
    ax.set(xlabel="Detector x [pixel]", ylabel="Detector y [pixel]")
    yield finish(fig, f"FLAT — combined flat with the {n_outputs} fitted traces (output numbers)")

    fig = page(figsize=(11.7, 11.7))
    axes = np.asarray(fig.subplots(6, 4)).ravel()
    for output, ax in enumerate(axes[:n_outputs]):
        columns = _x_range(flat, output)
        ax.plot(x[columns], flat.trace_y_pixels[output, columns], lw=1)
        ax.set_title(f"output {output}", fontsize=8)
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    fig.supxlabel("Detector x [pixel]")
    fig.supylabel("Trace centre y [pixel]")
    yield finish(fig, "FLAT — trace centres over the fitted columns (independent y scales)")

    if flat.profile_sigma_pixels is not None:
        fig = page()
        ax = fig.subplots()
        colors = [f"C{output % 10}" for output in range(n_outputs)]
        for output in range(n_outputs):
            columns = _x_range(flat, output)
            ax.plot(x[columns], flat.profile_sigma_pixels[output, columns], lw=0.8,
                    color=colors[output], label=str(output))
        ax.set(xlabel="Detector x [pixel]", ylabel="Profile width σ [pixel]")
        ax.legend(ncol=12, fontsize=6, title="output")
        yield finish(fig, "FLAT — fitted spatial profile width per output")

    if flat.telescope_flux_adu is not None:
        fig = page()
        ax = fig.subplots()
        labels = [f"T{t}" for t in layout.telescopes]
        bars = ax.bar(labels, flat.telescope_flux_adu, color="gray")
        ax.bar_label(bars, fmt="%.3g")
        ax.set(ylabel="Total flux per DIT [ADU]")
        yield finish(fig, "FLAT — flux of each telescope's flat (QC.FLATFLUX)")
