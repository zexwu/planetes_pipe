"""WAVE sections: the wavelength solution of every output and its line residuals."""
import numpy as np

from planetes_pipe.plots.pages import Section
from planetes_pipe.steps.wave import Wave


def sections(wave: Wave) -> list[Section]:
    """Sections for one WAVE."""
    n_outputs, width = wave.wavelength_map_m.shape
    x = np.arange(width)

    def draw_solution(fig):
        ax = fig.subplots()
        for output in range(n_outputs):
            ax.plot(x, wave.wavelength_map_m[output] * 1e9, lw=0.5)
        ax.set(xlabel="Detector x [pixel]", ylabel="Wavelength [nm]")

    result = [Section("Wavelength solution", draw_solution, 3.0,
                      f"Wavelength against detector column for the {n_outputs} outputs.")]
    if wave.fit_residual_m is not None:
        residual_nm = wave.fit_residual_m * 1e9
        rms = float(np.std(residual_nm, ddof=1)) if len(residual_nm) > 1 else float("nan")

        def draw_residuals(fig):
            scatter_ax, hist_ax = fig.subplots(1, 2, width_ratios=[2.2, 1])
            points = scatter_ax.scatter(wave.fit_x_pixels, residual_nm, s=4, c=wave.fit_y_pixels,
                                        cmap="viridis")
            fig.colorbar(points, ax=scatter_ax, label="Detector y [pixel]")
            scatter_ax.axhline(0, color="gray", lw=0.7)
            scatter_ax.set(xlabel="Accepted line x [pixel]", ylabel="Residual [nm]")
            hist_ax.hist(residual_nm, bins=50, color="gray", orientation="horizontal")
            hist_ax.set(xlabel="Lines", ylabel="Residual [nm]")

        result.append(Section("Lamp line residuals", draw_residuals, 3.0,
                              f"{len(residual_nm)} accepted lines; sample stdev {rms:.3f} nm."))
    return result
