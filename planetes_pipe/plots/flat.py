"""FLAT sections: the flat image with its traces, trace centres and widths, telescope fluxes."""
import numpy as np

from planetes_pipe.instrument import layout
from planetes_pipe.plots.pages import Section, table, table_height
from planetes_pipe.steps.flat import Flat


def _x_range(flat: Flat, output: int) -> slice:
    """Columns where an output's trace was fitted (all columns if not recorded)."""
    if flat.trace_x_range is None:
        return slice(None)
    first, last = flat.trace_x_range[output]
    return slice(int(first), int(last) + 1)


def sections(flat: Flat) -> list[Section]:
    """Sections for one FLAT."""
    n_outputs, width = flat.trace_y_pixels.shape
    x = np.arange(width)

    def draw_image(fig):
        ax = fig.subplots()
        low, high = np.percentile(flat.flat_map_adu, [1, 99.5])
        shown = ax.imshow(flat.flat_map_adu, origin="lower", aspect="auto", vmin=low, vmax=high)
        fig.colorbar(shown, ax=ax, label="ADU", pad=0.02)
        for output in range(n_outputs):
            columns = _x_range(flat, output)
            ax.plot(x[columns], flat.trace_y_pixels[output, columns], color="cyan", lw=0.4)
            ax.text(x[columns][-1] + 5, flat.trace_y_pixels[output, columns][-1], str(output),
                    color="cyan", fontsize=5, va="center")
        ax.minorticks_off()
        ax.set(xlabel="Detector x [pixel]", ylabel="Detector y [pixel]")

    def draw_traces(fig):
        ax = fig.subplots()
        for output in range(n_outputs):
            columns = _x_range(flat, output)
            trace = flat.trace_y_pixels[output, columns]
            ax.plot(x[columns], trace - np.median(trace), color=f"C{output % 4}", lw=0.5, alpha=0.8)
        ax.set(xlabel="Detector x [pixel]", ylabel="Trace centre − median [pixel]")

    result = [Section("FLAT image and traces", draw_image, 4.4,
                      f"Mean of the four dark-subtracted telescope flats, with the {n_outputs} fitted traces and their output numbers; "
                      "colour range 1–99.5 %."),
              Section("Trace centres", draw_traces, 2.4,
                      "Cubic fit to the Gaussian centres of each output's columns, minus its median: the "
                      "curvature and tilt of all outputs on one axis.")]
    if flat.profile_sigma_pixels is not None:
        def draw_widths(fig):
            ax = fig.subplots()
            colors = [f"C{t}" for t in range(4)]
            for output in range(n_outputs):
                columns = _x_range(flat, output)
                ax.plot(x[columns], flat.profile_sigma_pixels[output, columns], lw=0.5,
                        color=colors[output % 4], alpha=0.8)
            ax.set(xlabel="Detector x [pixel]", ylabel="Profile width σ [pixel]")
        result.append(Section("Profile widths", draw_widths, 2.4,
                              "$\\sigma$ of the Gaussian fitted across each output, per column."))
    if flat.telescope_flux_adu is not None:
        cells = [[f"{f:.4g}" for f in flat.telescope_flux_adu]]
        result.append(Section("Telescope fluxes (QC.FLATFLUX)",
                              lambda fig: table(fig, cells, [f"T{t}" for t in layout.telescopes],
                                                ["ADU per DIT"]),
                              table_height(1), "Sum of all pixels of each telescope's dark-subtracted mean flat."))
    return result
