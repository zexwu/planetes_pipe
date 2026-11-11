"""WAVE sections: summary, lamp image and spectra on the solution, the solution and its
dispersion, lines per output, and the line residuals along x, across y and per lamp line.

A product written before 0.35.0 has only the solution and its accepted residuals; the
sections that need the lamp spectra, the image or every fitted line are then left out.
"""
import numpy as np

from matplotlib import colormaps
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm, ListedColormap

from planetes_pipe.io.keywords import QC_SATURATED
from planetes_pipe.plots.pages import (
    FLAG,
    GUIDE,
    Section,
    legend,
    table,
    table_height,
)
from planetes_pipe.steps.wave import (
    LINE_MAX_FWHM,
    QC_WAVE_MAX,
    QC_WAVE_MIN,
    REFERENCE_WAVELENGTH_M,
    SURFACE_DEGREE,
    Wave,
    accepted_residual_m,
    qc,
)

ACCEPTED = {"color": "0.25", "ls": "", "marker": ".", "ms": 3}
REJECTED = {"color": "#FF2C00", "ls": "", "marker": "o", "ms": 4, "mfc": "none", "mew": 0.7}
BROAD = {"color": "#FF9500", "ls": "", "marker": "x", "ms": 4, "mew": 0.9}
OUTPUT_CMAP = "viridis"  # outputs coloured by index, bottom of the detector first

def _lines(wave: Wave):
    """Every fitted line: residual [nm], x, y, laboratory wavelength [nm], output, rejected."""
    n = len(wave.fit_residual_m)
    rejected = np.zeros(n, bool) if wave.fit_rejected is None else wave.fit_rejected
    lab = None if wave.fit_wavelength_m is None else wave.fit_wavelength_m * 1e9
    return (wave.fit_residual_m * 1e9, wave.fit_x_pixels, wave.fit_y_pixels, lab,
            wave.fit_output, rejected)


def _broad(wave: Wave):
    """Lines dropped before the fit as broader than LINE_MAX_FWHM (none before 0.43.0)."""
    if wave.fit_fwhm_pixels is None:
        return np.zeros(len(wave.fit_residual_m), bool)
    return np.nan_to_num(wave.fit_fwhm_pixels) > LINE_MAX_FWHM


def _output_colors(n_outputs):
    return colormaps[OUTPUT_CMAP](np.linspace(0, 0.95, n_outputs))


def _output_colorbar(fig, axes, n_outputs):
    """A discrete colour bar of the output index, for lines coloured by _output_colors."""
    norm = BoundaryNorm(np.arange(n_outputs + 1) - 0.5, n_outputs)
    bar = fig.colorbar(ScalarMappable(norm, ListedColormap(_output_colors(n_outputs))), ax=axes,
                       pad=0.02, label="Output")
    bar.set_ticks(np.arange(0, n_outputs, 4))


def summary(wave: Wave, header=None) -> list[tuple[str, str]]:
    """Rows of the summary table: (label, value)."""
    values = qc(wave)
    full = wave.wavelength_map_m * 1e9
    dispersion = np.abs(np.gradient(full, axis=1))
    rows = [("Common band [nm]", f"{values[QC_WAVE_MIN] * 1e9:.1f} – {values[QC_WAVE_MAX] * 1e9:.1f}"),
            ("Full range [nm]", f"{full.min():.1f} – {full.max():.1f}"),
            ("Dispersion [nm/pixel]", f"{np.median(dispersion):.3f} "
                                      f"({dispersion.min():.3f} – {dispersion.max():.3f})")]
    in_range = (REFERENCE_WAVELENGTH_M * 1e9 >= full.min()) & (REFERENCE_WAVELENGTH_M * 1e9 <= full.max())
    rows.append(("Reference lines in range", f"{int(in_range.sum())} of {len(REFERENCE_WAVELENGTH_M)}"))
    if wave.fit_residual_m is not None:
        residual, _, _, lab, output, rejected = _lines(wave)
        accepted = residual[~rejected]
        rms = np.std(accepted, ddof=1) if len(accepted) > 1 else np.nan
        broad = _broad(wave)
        rows.append(("Lines matched / too broad / rejected",
                     f"{len(residual)} / {int(broad.sum())} / {int((rejected & ~broad).sum())}"))
        if wave.fit_fwhm_pixels is not None and np.isfinite(wave.fit_fwhm_pixels).any():
            p5, p50, p95 = np.nanpercentile(wave.fit_fwhm_pixels, [5, 50, 95])
            rows.append(("Line FWHM [pixel]", f"{p50:.2f} (5–95%: {p5:.2f} – {p95:.2f}); "
                                              f"dropped above {LINE_MAX_FWHM:g}"))
        rows.append(("Residual stdev (accepted)",
                     f"{rms:.3f} nm = {rms / np.median(dispersion):.3f} pixel"))
        if output is not None:
            per_output = np.bincount(output[~rejected], minlength=full.shape[0])
            rows.append(("Accepted lines per output", f"{per_output.min()} – {per_output.max()}"))
        if lab is not None:
            used = np.unique(lab[~rejected])
            rows.append(("Lamp lines used", f"{len(used)}, {used.min():.1f} – {used.max():.1f} nm"))
    rows.append(("Surface degree", str(SURFACE_DEGREE)))
    if header is not None:
        rows.append(("Saturated lamp samples", str(QC_SATURATED.at("LAMP").get(header) or 0)))
    return rows


def sections(wave: Wave, header=None) -> list[Section]:
    """Sections for one WAVE; header adds the lamp's saturation to the summary."""
    n_outputs, width = wave.wavelength_map_m.shape
    x = np.arange(width)
    wavelength_nm = wave.wavelength_map_m * 1e9
    has_lines = wave.fit_residual_m is not None
    result = []

    rows = summary(wave, header)
    shade = [((i, 0), FLAG) for i, (label, value) in enumerate(rows)
             if label == "Saturated lamp samples" and value != "0"]
    result.append(Section("Wavelength calibration summary",
                          lambda fig: table(fig, [[v] for _, v in rows], None,
                                            [label for label, _ in rows], shade=shade),
                          table_height(len(rows) - 1),
                          "The solution's coverage and dispersion, and how well the lamp lines fit it."))

    if wave.lamp_image_adu is not None:
        def draw_image(fig):
            ax = fig.subplots()
            image = wave.lamp_image_adu
            low, high = np.percentile(image, [5, 99.7])
            shown = ax.imshow(np.arcsinh((image - low) / max(high - low, 1e-9) * 10),
                              origin="lower", aspect="auto", cmap="gray_r")
            shown.set_clim(0, np.arcsinh(10))
            if has_lines:
                _, lx, ly, _, _, rejected = _lines(wave)
                broad = _broad(wave)
                ax.plot(lx[~rejected], ly[~rejected], **{**ACCEPTED, "color": "#0C5DA5", "ms": 2})
                ax.plot(lx[rejected & ~broad], ly[rejected & ~broad], **REJECTED)
                ax.plot(lx[broad], ly[broad], **BROAD)
            ax.minorticks_off()
            ax.set(xlabel="Detector x [pixel]", ylabel="Detector y [pixel]")
            if has_lines:
                legend(fig, [("accepted line", {**ACCEPTED, "color": "#0C5DA5"}),
                             ("rejected line", REJECTED),
                             (f"too broad (FWHM > {LINE_MAX_FWHM:g} px)", BROAD)])
        result.append(Section("Lamp image with the fitted lines", draw_image, 3.2,
                              "Mean lamp frame minus its dark (arcsinh stretch), and every lamp "
                              "line matched: used, rejected beyond 3σ, or dropped as defocused."))

    if wave.lamp_spectra_adu is not None:
        spectra = wave.lamp_spectra_adu
        scale = np.percentile(spectra, 99.9, axis=1, keepdims=True)
        normalized = spectra / np.where(scale > 0, scale, 1)
        colors = _output_colors(n_outputs)
        low, high = wavelength_nm.min(), wavelength_nm.max()
        reference = REFERENCE_WAVELENGTH_M * 1e9
        reference = reference[(reference >= low) & (reference <= high)]
        used = set()
        if has_lines and wave.fit_wavelength_m is not None:
            _, _, _, lab, _, rejected = _lines(wave)
            used = set(np.round(lab[~rejected], 4))

        def draw_spectra(fig):
            middle = (low + high) / 2
            axes = fig.subplots(2, 1)
            for ax, (start, stop) in zip(axes, [(low, middle + 2), (middle - 2, high)]):
                for output in range(n_outputs):
                    ax.plot(wavelength_nm[output], normalized[output], color=colors[output],
                            lw=0.4, alpha=0.7)
                for line in reference:
                    if not start <= line <= stop:
                        continue
                    if np.round(line, 4) in used:
                        ax.axvline(line, color="#FF2C00", lw=0.6, ls="--")
                        ax.text(line, 1.02, f"{line:.1f}", rotation=90, fontsize=5.5,
                                ha="center", va="bottom", color="#FF2C00",
                                transform=ax.get_xaxis_transform())
                    else:
                        ax.axvline(line, **GUIDE)
                ax.set(xlim=(start, stop), ylim=(-0.05, 1.05), ylabel="Normalized flux")
            axes[-1].set(xlabel="Wavelength [nm]")
            _output_colorbar(fig, list(axes), n_outputs)
            legend(fig, [("reference line used", {"color": "#FF2C00", "lw": 0.6, "ls": "--"}),
                         ("reference line not used", GUIDE)])
        result.append(Section("Lamp spectra of every output on its wavelength solution",
                              draw_spectra, 4.6,
                              "Each output's lamp spectrum (normalized) against its own solution: "
                              "with a good solution the lines of all outputs coincide at their "
                              "laboratory wavelengths."))

    def draw_solution(fig):
        axes = fig.subplots(1, 3)
        colors = _output_colors(n_outputs)
        median = np.median(wavelength_nm, axis=0)
        dispersion = np.gradient(wavelength_nm, axis=1)
        for output in range(n_outputs):
            axes[0].plot(x, wavelength_nm[output], color=colors[output], lw=0.5)
            axes[1].plot(x, dispersion[output], color=colors[output], lw=0.5)
            axes[2].plot(x, wavelength_nm[output] - median, color=colors[output], lw=0.5)
        axes[0].set(xlabel="Detector x [pixel]", ylabel="Wavelength [nm]")
        axes[1].set(xlabel="Detector x [pixel]", ylabel="Dispersion [nm/pixel]")
        axes[2].axhline(0, **GUIDE)
        axes[2].set(xlabel="Detector x [pixel]", ylabel="Deviation from median output [nm]")
        _output_colorbar(fig, list(axes), n_outputs)
    result.append(Section("Wavelength solution (WAVE_DATA_SC), dispersion and spread across outputs",
                          draw_solution, 2.6,
                          f"$\\lambda_o(x)$ of each detector column for the {n_outputs} outputs, the dispersion $d\\lambda/dx$, and "
                          "the deviation $\\lambda_o(x) - \\mathrm{median}_o\\,\\lambda_o(x)$ from the other outputs."))

    if not has_lines:
        return result
    residual, lx, ly, lab, output, rejected = _lines(wave)
    accepted = accepted_residual_m(wave) * 1e9
    rms = float(np.std(accepted, ddof=1)) if len(accepted) > 1 else float("nan")
    limit = max(6 * rms, np.abs(accepted).max() * 1.2) if len(accepted) else 1.0

    if output is not None:
        def draw_counts(fig):
            ax = fig.subplots()
            kept = np.bincount(output[~rejected], minlength=n_outputs)
            lost = np.bincount(output[rejected], minlength=n_outputs)
            ax.bar(np.arange(n_outputs), kept, color="0.45")
            ax.bar(np.arange(n_outputs), lost, bottom=kept, color="#FF2C00")
            ax.set(xlabel="Output", ylabel="Lines", xticks=np.arange(0, n_outputs, 2))
            legend(fig, [("accepted", {"color": "0.45", "lw": 6}),
                         ("rejected", {"color": "#FF2C00", "lw": 6})])
        result.append(Section("Lamp lines per output", draw_counts, 1.8,
                              "Lines matched in each output and kept or rejected by the fit; an "
                              "output with few lines constrains the surface little."))

    def draw_along_x(fig):
        scatter_ax, hist_ax = fig.subplots(1, 2, width_ratios=[3, 1], sharey=True)
        points = scatter_ax.scatter(lx[~rejected], residual[~rejected], s=3, c=ly[~rejected],
                                    cmap="viridis", linewidths=0)
        scatter_ax.plot(lx[rejected], residual[rejected], **REJECTED)
        scatter_ax.axhline(0, **GUIDE)
        scatter_ax.set(xlabel="Line x [pixel]", ylabel="Residual [nm]", ylim=(-limit, limit))
        bins = np.linspace(-limit, limit, 51)
        hist_ax.hist(residual[~rejected], bins=bins, color="0.45", orientation="horizontal",
                     histtype="stepfilled")
        if rejected.any():
            hist_ax.hist(np.clip(residual[rejected], -limit, limit), bins=bins, color="#FF2C00",
                         orientation="horizontal", histtype="step")
        hist_ax.axhline(0, **GUIDE)
        hist_ax.tick_params(labelleft=False)
        hist_ax.set(xlabel="Lines")
        fig.colorbar(points, ax=[scatter_ax, hist_ax], label="Detector y [pixel]", pad=0.02)
    result.append(Section("Line residuals along the dispersion", draw_along_x, 2.6,
                          f"Residual = laboratory − fitted wavelength; fit $\\lambda(x, y)$: one polynomial of total degree "
                          f"{SURFACE_DEGREE}, lines beyond 3$\\sigma$ rejected three times. {int((~rejected).sum())} accepted "
                          f"(stdev {rms:.3f} nm), {int(rejected.sum())} rejected (red, clipped to the axis); a trend in x is a missing term."))

    def draw_across_y(fig):
        ax = fig.subplots()
        points = ax.scatter(ly[~rejected], residual[~rejected], s=3, c=lx[~rejected],
                            cmap="plasma", linewidths=0)
        ax.plot(ly[rejected], np.clip(residual[rejected], -limit, limit), **REJECTED)
        ax.axhline(0, **GUIDE)
        ax.set(xlabel="Line y [pixel]", ylabel="Residual [nm]", ylim=(-limit, limit))
        fig.colorbar(points, ax=ax, label="Detector x [pixel]", pad=0.02)
    result.append(Section("Line residuals across the outputs", draw_across_y, 2.4,
                          "The same lines against detector y; a step between outputs is an "
                          "output whose lines were matched or centred differently."))

    if lab is not None:
        def draw_per_line(fig):
            ax = fig.subplots()
            lines = np.unique(lab)
            index = np.searchsorted(lines, lab)
            spread = output / max(n_outputs - 1, 1) - 0.5 if output is not None else 0
            position = index + 0.7 * spread
            ax.plot(position[~rejected], residual[~rejected], **ACCEPTED)
            ax.plot(position[rejected], np.clip(residual[rejected], -limit, limit), **REJECTED)
            for k, line in enumerate(lines):
                kept = (lab == line) & ~rejected
                if kept.any():
                    ax.plot([k - 0.4, k + 0.4], [np.median(residual[kept])] * 2, color="#0C5DA5",
                            lw=1.4)
            ax.axhline(0, **GUIDE)
            ax.set_xticks(np.arange(len(lines)), [f"{v:.1f}" for v in lines], rotation=90,
                          fontsize=6)
            ax.minorticks_off()
            ax.set(xlabel="Laboratory wavelength [nm]", ylabel="Residual [nm]", ylim=(-limit, limit),
                   xlim=(-0.6, len(lines) - 0.4))
            legend(fig, [("accepted, outputs 0→23 left to right", ACCEPTED), ("rejected", REJECTED),
                         ("median of the line", {"color": "#0C5DA5", "lw": 1.4})])
        result.append(Section("Residual of each lamp line", draw_per_line, 2.4,
                              "Each line's residual in every output, one column per line; "
                              "a line off zero in all outputs has a wrong laboratory wavelength or "
                              "is a blend."))
    return result
