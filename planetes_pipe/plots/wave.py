"""WAVE pages: the wavelength solution of every output and its line residuals."""
import numpy as np

from planetes_pipe.plots.pages import finish, page
from planetes_pipe.steps.wave import Wave


def pages(wave: Wave):
    """Pages for one WAVE."""
    n_outputs, width = wave.wavelength_map_m.shape
    x = np.arange(width)
    fig = page()
    ax = fig.subplots()
    for output in range(n_outputs):
        ax.plot(x, wave.wavelength_map_m[output] * 1e9, lw=0.6)
    ax.set(xlabel="Detector x [pixel]", ylabel="Wavelength [nm]")
    yield finish(fig, f"WAVE — wavelength solution of the {n_outputs} outputs")

    if wave.fit_residual_m is None:
        return
    residual_nm = wave.fit_residual_m * 1e9
    fig = page()
    top = fig.subplots(2, 1)
    scatter = top[0].scatter(wave.fit_x_pixels, residual_nm, s=6, c=wave.fit_y_pixels,
                             cmap="viridis")
    fig.colorbar(scatter, ax=top[0], label="Detector y [pixel]")
    top[0].axhline(0, color="gray", lw=0.7)
    top[0].set(xlabel="Accepted line x [pixel]", ylabel="Residual [nm]")
    top[1].hist(residual_nm, bins=60, color="gray")
    rms = float(np.std(residual_nm, ddof=1)) if len(residual_nm) > 1 else float("nan")
    top[1].set(xlabel="Residual [nm]", ylabel="Lines",
               title=f"{len(residual_nm)} accepted lines, sample stdev {rms:.3f} nm")
    yield finish(fig, "WAVE — residuals of the accepted lamp lines")
