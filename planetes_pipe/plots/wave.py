"""WAVE sections: the wavelength solution of every output and its line residuals."""
import numpy as np

from planetes_pipe.plots.pages import GUIDE, Section
from planetes_pipe.steps.wave import Wave


def sections(wave: Wave) -> list[Section]:
    """Sections for one WAVE."""
    n_outputs, width = wave.wavelength_map_m.shape
    x = np.arange(width)

    def draw_solution(fig):
        ax = fig.subplots()
        for output in range(n_outputs):
            ax.plot(x, wave.wavelength_map_m[output] * 1e9, lw=0.5, color="0.2", alpha=0.6)
        ax.set(xlabel="Detector x [pixel]", ylabel="Wavelength [nm]")

    result = [Section("Wavelength solution", draw_solution, 2.6,
                      f"Wavelength of each detector column from the lamp line fit, for the {n_outputs} outputs.")]
    if wave.fit_residual_m is not None:
        residual_nm = wave.fit_residual_m * 1e9
        rms = float(np.std(residual_nm, ddof=1)) if len(residual_nm) > 1 else float("nan")

        def draw_residuals(fig):
            scatter_ax, hist_ax = fig.subplots(1, 2, width_ratios=[3, 1], sharey=True)
            points = scatter_ax.scatter(wave.fit_x_pixels, residual_nm, s=3, c=wave.fit_y_pixels,
                                        cmap="viridis", linewidths=0)
            scatter_ax.axhline(0, **GUIDE)
            scatter_ax.set(xlabel="Accepted line x [pixel]", ylabel="Residual [nm]")
            hist_ax.hist(residual_nm, bins=50, color="0.45", orientation="horizontal",
                         histtype="stepfilled")
            hist_ax.axhline(0, **GUIDE)
            hist_ax.tick_params(labelleft=False)
            hist_ax.set(xlabel="Lines")
            fig.colorbar(points, ax=[scatter_ax, hist_ax], label="Detector y [pixel]", pad=0.02)

        result.append(Section("Lamp line residuals", draw_residuals, 2.6,
                              f"Accepted lamp lines about the fitted solution: {len(residual_nm)} lines, sample stdev {rms:.3f} nm."))
    return result
