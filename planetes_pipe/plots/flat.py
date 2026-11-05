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
        shown = ax.imshow(flat.flat_map_adu, origin="lower", aspect="auto", cmap="inferno",
                          vmin=low, vmax=high)
        fig.colorbar(shown, ax=ax, label="ADU")
        for output in range(n_outputs):
            columns = _x_range(flat, output)
            ax.plot(x[columns], flat.trace_y_pixels[output, columns], color="cyan", lw=0.5)
            ax.text(x[columns][-1] + 5, flat.trace_y_pixels[output, columns][-1], str(output),
                    color="cyan", fontsize=5, va="center")
        ax.set(xlabel="Detector x [pixel]", ylabel="Detector y [pixel]")

    def draw_traces(fig):
        axes = np.asarray(fig.subplots(6, 4)).ravel()
        for output, ax in enumerate(axes[:n_outputs]):
            columns = _x_range(flat, output)
            ax.plot(x[columns], flat.trace_y_pixels[output, columns])
            ax.set_title(f"output {output}", fontsize=6, pad=2)
            ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        fig.supxlabel("Detector x [pixel]", fontsize=7)
        fig.supylabel("Trace centre y [pixel]", fontsize=7)

    result = [Section("FLAT image and traces", draw_image, 5.0,
                      f"Combined flat with the {n_outputs} fitted traces and their output numbers; "
                      "colour range 1–99.5 %."),
              Section("Trace centres", draw_traces, 7.6,
                      "Fitted centre of each output over its fitted columns (independent y scales).")]
    if flat.profile_sigma_pixels is not None:
        def draw_widths(fig):
            ax = fig.subplots()
            for output in range(n_outputs):
                columns = _x_range(flat, output)
                ax.plot(x[columns], flat.profile_sigma_pixels[output, columns], lw=0.7,
                        color=f"C{output % 10}", label=str(output))
            ax.set(xlabel="Detector x [pixel]", ylabel="Profile width σ [pixel]")
            ax.legend(ncol=12, fontsize=5, title="output", title_fontsize=6)
        result.append(Section("Profile widths", draw_widths, 2.8,
                              "Fitted spatial profile width of each output."))
    if flat.telescope_flux_adu is not None:
        cells = [[f"{f:.4g}" for f in flat.telescope_flux_adu]]
        result.append(Section("Telescope fluxes (QC.FLATFLUX)",
                              lambda fig: table(fig, cells, [f"T{t}" for t in layout.telescopes],
                                                ["ADU per DIT"]),
                              table_height(1), "Total flux of each telescope's dark-subtracted flat."))
    return result
