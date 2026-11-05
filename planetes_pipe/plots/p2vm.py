"""P2VM sections: a quality summary and the instrumental response, as GRAVITY's P2VM
report (gravi_visual_p2vm.py), plus diagnostics of the calibration scans.

The quality rules are GRAVITY's: each output's relative transmission within
PHOTOMETRY_RANGE, and the normalized coherence within COHERENCE_RANGE over the
central band except for ALLOWED_OUTSIDE of a spectrum. Saturated raw samples
(QC.NSAT in the product header) are a PLANETES addition.
"""
import numpy as np

from planetes_pipe.instrument import layout
from planetes_pipe.io.keywords import QC_SATURATED
from planetes_pipe.plots.pages import (
    ABCD_COLORS,
    Section,
    baseline_grid,
    baseline_name,
    table,
    table_height,
)
from planetes_pipe.steps.p2vm import P2VM

PHOTOMETRY_RANGE = (0.7, 1.3)    # relative transmission of each output
COHERENCE_RANGE = (0.80, 1.03)   # normalized coherence
CENTRAL_BAND = (0.15, 0.85)      # fraction of the wavelength channels checked
ALLOWED_OUTSIDE = 0.05           # of one spectrum's channels, over all baselines
NOMINAL_PHASE_DEG = {"C": 180.0, "B": 90.0, "D": 270.0}  # relative to A


def normalized_coherence(p2vm: P2VM) -> np.ndarray:
    """Coherence / (2 sqrt(T1 T2)) of each baseline's outputs: (baseline, A/C/B/D, wavelength)."""
    values = []
    for b, (t1, t2) in enumerate(layout.baseline_pairs):
        outputs = p2vm.baseline_outputs[b]
        norm = 2 * np.sqrt(p2vm.transmission[outputs, t1] * p2vm.transmission[outputs, t2])
        values.append(np.divide(p2vm.coherence_amplitude[outputs, b], norm,
                                out=np.full_like(norm, np.nan), where=norm > 0))
    return np.asarray(values)


def relative_output_transmission(p2vm: P2VM) -> np.ndarray:
    """Mean transmission of each output over telescopes and wavelengths / the overall mean."""
    return p2vm.transmission.mean(axis=(1, 2)) / p2vm.transmission.mean()


def central_band(p2vm: P2VM) -> slice:
    n = len(p2vm.wavelength_m)
    return slice(int(n * CENTRAL_BAND[0]), int(n * CENTRAL_BAND[1]))


def output_phases_deg(p2vm: P2VM) -> np.ndarray:
    """Phase of each baseline's outputs relative to A over the central band: (baseline, 4)."""
    band = central_band(p2vm)
    phases = [np.angle(np.exp(1j * p2vm.coherence_phase_rad[p2vm.baseline_outputs[b], b, band])
                       ).mean(axis=1) for b in range(len(layout.baselines))]
    return np.rad2deg(np.asarray(phases)) % 360


def quality(p2vm: P2VM, header=None) -> dict:
    """GRAVITY's P2VM quality checks, plus raw saturation; verdict GOOD, WARNING or BAD."""
    relative = relative_output_transmission(p2vm)
    coherence = normalized_coherence(p2vm)[..., central_band(p2vm)]
    allowed = ALLOWED_OUTSIDE * len(p2vm.wavelength_m)
    low = int(np.count_nonzero(coherence < COHERENCE_RANGE[0]))
    high = int(np.count_nonzero(coherence > COHERENCE_RANGE[1]))
    labels = [f"T{t}" for t in layout.telescopes] + [f"SC{b}" for b in layout.baselines]
    counts = {label: QC_SATURATED.at(label).get(header) for label in labels} if header is not None else {}
    saturated = {label: int(count) for label, count in counts.items() if count}
    checks = {
        f"Output transmission < {PHOTOMETRY_RANGE[0]}": int(np.any(relative < PHOTOMETRY_RANGE[0])),
        f"Output transmission > {PHOTOMETRY_RANGE[1]}": int(np.any(relative > PHOTOMETRY_RANGE[1])),
        f"Coherence < {COHERENCE_RANGE[0]} in > {allowed:.0f} channels": int(low > allowed),
        f"Coherence > {COHERENCE_RANGE[1]} in > {allowed:.0f} channels": int(high > allowed),
        "Saturated raw samples": int(bool(saturated)),
    }
    failed = sum(checks.values())
    verdict = "GOOD" if failed == 0 else ("WARNING" if failed < 3 else "BAD")
    return {"verdict": verdict, "checks": checks, "relative_transmission": relative,
            "coherence_low": low, "coherence_high": high, "saturated": saturated,
            "visibility": np.nanmean(coherence, axis=(1, 2))}


def phase_residual_rad(phase_rad, opd_m, wavelength_m: float):
    """Fringe phase minus ±2π·OPD/λ and its circular mean, wrapped to ±π.

    No unwrapping: a scan step can exceed a fringe. The sign that fits better is used.
    """
    best = None
    for sign in (1.0, -1.0):
        offset = phase_rad - sign * 2 * np.pi * opd_m / wavelength_m
        residual = np.angle(np.exp(1j * (offset - np.angle(np.exp(1j * offset).mean()))))
        if best is None or np.mean(np.abs(residual)) < np.mean(np.abs(best)):
            best = residual
    return best


BASELINE_GRID_HEIGHT = 4.2  # inches for six baseline panels on a portrait page
VERDICT_COLORS = {"GOOD": "palegreen", "WARNING": "khaki", "BAD": "salmon"}


def sections(p2vm: P2VM, header=None) -> list[Section]:
    """Sections for one P2VM, in GRAVITY's order; header adds its QC to the summary."""
    result = quality(p2vm, header)
    wavelength = p2vm.wavelength_m * 1e9
    n = len(layout.baselines)
    names = [baseline_name(b) for b in range(n)]
    out = []

    # Quality summary (GRAVITY: "P2VM overall quality" and its checks).
    def draw_verdict(fig):
        ax = fig.subplots()
        ax.axis("off")
        ax.text(0.5, 0.5, f"P2VM overall quality: {result['verdict']}", ha="center", va="center",
                fontsize=12, bbox={"facecolor": VERDICT_COLORS[result["verdict"]], "pad": 6})
    out.append(Section("Quality", draw_verdict, 0.5, "GRAVITY gravi_visual rules, plus raw "
                       "saturation: GOOD with no failed check, WARNING with 1–2, BAD with 3 or more."))

    rows = list(result["checks"])
    outputs = [(o, f"{baseline_name(b)} {label}") for b in range(n)
               for o, label in zip(p2vm.baseline_outputs[b], layout.abcd_order)]
    relative = result["relative_transmission"]
    detail = {rows[0]: sum(relative[o] < PHOTOMETRY_RANGE[0] for o, _ in outputs),
              rows[1]: sum(relative[o] > PHOTOMETRY_RANGE[1] for o, _ in outputs),
              rows[2]: result["coherence_low"], rows[3]: result["coherence_high"],
              rows[4]: ", ".join(f"{k} {v}" for k, v in result["saturated"].items()) or "none"}
    checks = [["FAIL" if failed else "ok", str(detail[name])] for name, failed in result["checks"].items()]
    out.append(Section("Quality checks", lambda fig: table(
        fig, checks, ["Result", "Value"], rows,
        shade=[((i, 0), "salmon" if f else "palegreen") for i, f in enumerate(result["checks"].values())]),
        table_height(len(rows)), "Values: outputs outside the range; coherence samples outside the "
        "range in the central band; saturated samples per raw input (QC.NSAT)."))

    phases = output_phases_deg(p2vm)
    per_baseline = [[f"{100 * result['visibility'][b]:.1f} %"] + [f"{phases[b, i]:.1f}" for i in (1, 2, 3)]
                    for b in range(n)]
    out.append(Section("Per baseline", lambda fig: table(
        fig, per_baseline, ["Visibility transmission", "C−A [deg]", "B−A [deg]", "D−A [deg]"], names,
        shade=[((b, 0), "salmon") for b in range(n) if result["visibility"][b] < COHERENCE_RANGE[0]]),
        table_height(n), f"Central band means; visibility transmission below "
        f"{100 * COHERENCE_RANGE[0]:.0f} % shaded. Nominal phases C−A 180, B−A 90, D−A 270."))

    # Photometry (GRAVITY: relative transmission per output; transmissions vs wavelength).
    def draw_relative_outputs(fig):
        ax = fig.subplots()
        order = [o for b in range(n) for o in p2vm.baseline_outputs[b]]
        values = relative[order]
        ok = (values >= PHOTOMETRY_RANGE[0]) & (values <= PHOTOMETRY_RANGE[1])
        ax.bar(range(len(order)), values, color=np.where(ok, "steelblue", "salmon"))
        for limit in PHOTOMETRY_RANGE:
            ax.axhline(limit, color="gray", ls="--", lw=0.7)
        ax.set_xticks(range(len(order)), [label for _, label in outputs], rotation=90, fontsize=5)
        ax.set(ylabel="Relative transmission", ylim=(0, max(2, values.max() * 1.1)))
    out.append(Section("Relative photometric transmission per output", draw_relative_outputs, 2.6,
                       f"Mean over telescopes and wavelengths / overall mean; allowed "
                       f"{PHOTOMETRY_RANGE[0]}–{PHOTOMETRY_RANGE[1]} (outside: red)."))

    def percent_cells():
        cells = []
        for b in range(n):
            flux = p2vm.transmission[p2vm.baseline_outputs[b]].sum(axis=(0, 2))
            cells.append([f"{100 * f / flux.sum():.1f}" for f in flux])
        return cells
    out.append(Section("Wavelength-averaged photometric transmission (%)", lambda fig: table(
        fig, percent_cells(), [f"T{t}" for t in layout.telescopes], names,
        shade=[((b, t), "lavender") for b, pair in enumerate(layout.baseline_pairs) for t in pair]),
        table_height(n), "Share of each telescope in each baseline's four outputs; the baseline's "
        "two telescopes shaded, the others are cross-talk."))

    share = p2vm.transmission / p2vm.transmission.sum(axis=1, keepdims=True)
    for values, title, ylabel, explanation, ylim in (
            (p2vm.transmission, "Photometric transmission vs. wavelength", "Transmission",
             "For each telescope and wavelength the mean over the 24 outputs is 1.", None),
            (share, "Relative photometric transmission vs. wavelength", "Share of output flux",
             "Each telescope / all four telescopes in the output; balanced = 0.5.", (0, 1))):
        def draw(fig, values=values, ylabel=ylabel, ylim=ylim):
            for b, ax in enumerate(baseline_grid(fig)):
                t1, t2 = layout.baseline_pairs[b]
                first, second = layout.baselines[b]
                for o, label, color in zip(p2vm.baseline_outputs[b], layout.abcd_order, ABCD_COLORS):
                    ax.plot(wavelength, values[o, t1], color=color, label=f"{label} T{first}")
                    ax.plot(wavelength, values[o, t2], color=color, ls="--", label=f"{label} T{second}")
                if ylim:
                    ax.axhline(0.5, color="gray", lw=0.6)
                    ax.set_ylim(*ylim)
                ax.set(title=f"{baseline_name(b)}: T{first} solid, T{second} dashed",
                       xlabel="Wavelength [nm]", ylabel=ylabel)
            fig.axes[0].legend(ncol=2)
        out.append(Section(title, draw, BASELINE_GRID_HEIGHT, explanation))

    # Coherence (GRAVITY: coherent flux matrix, visibility transmission, coherence vs wavelength).
    def matrix_cells():
        cells = []
        for b in range(n):
            flux = np.abs(p2vm.coherence_amplitude[p2vm.baseline_outputs[b]]).sum(axis=(0, 2))
            cells.append([f"{100 * f / flux.sum():.1f}" for f in flux])
        return cells
    out.append(Section("Wavelength-averaged coherent flux transmission (%)", lambda fig: table(
        fig, matrix_cells(), names, [f"outputs of {name}" for name in names],
        shade=[((b, b), "lavender") for b in range(n)]),
        table_height(n), "Coherent flux of each baseline (columns) in each baseline's outputs; "
        "close to 100 on the diagonal and 0 elsewhere."))

    coherence = normalized_coherence(p2vm)
    band = central_band(p2vm)

    def draw_visibility(fig):
        ax = fig.subplots()
        mean = np.nanmean(coherence[..., band], axis=2)
        for i, (label, color) in enumerate(zip(layout.abcd_order, ABCD_COLORS)):
            ax.bar(np.arange(n) + (i - 1.5) * 0.2, 100 * mean[:, i], 0.2, color=color, label=label)
        ax.axhline(100 * COHERENCE_RANGE[0], color="gray", ls="--", lw=0.7,
                   label=f"nominal minimum {100 * COHERENCE_RANGE[0]:.0f} %")
        ax.set_xticks(range(n), names)
        ax.set(ylabel="Visibility transmission [%]", ylim=(0, 110))
        ax.legend(ncol=5)
    out.append(Section("Wavelength-averaged visibility transmission", draw_visibility, 2.6,
                       "Normalized fringe amplitude over the central band, per output; nominal "
                       f"higher than {100 * COHERENCE_RANGE[0]:.0f} %."))

    def draw_coherence(fig):
        for b, ax in enumerate(baseline_grid(fig)):
            for values, label, color in zip(coherence[b], layout.abcd_order, ABCD_COLORS):
                ax.plot(wavelength, values, color=color, label=label)
            ax.axvspan(wavelength[band][0], wavelength[band][-1], color="gray", alpha=0.08)
            ax.axhline(COHERENCE_RANGE[0], color="gray", ls="--", lw=0.6)
            ax.set(title=baseline_name(b), xlabel="Wavelength [nm]", ylabel="Normalized coherence")
        fig.axes[0].legend()
    out.append(Section("Coherence vs. wavelength", draw_coherence, BASELINE_GRID_HEIGHT,
                       "Coherence / (2 √T1 √T2) of each output; shaded: the checked central band; "
                       f"dashed: {COHERENCE_RANGE[0]}. Instrumental response, not target visibility."))

    # Phases (GRAVITY: phase table and output phases vs wavelength).
    phase_cells = [[f"{phases[b, 1]:.1f}", f"{phases[b, 2]:.1f}", f"{phases[b, 3]:.1f}",
                    f"{(phases[b, 3] - phases[b, 2]) % 360:.1f}"] for b in range(n)]
    out.append(Section("Wavelength-averaged phases of the outputs (deg, A = 0)", lambda fig: table(
        fig, phase_cells, ["C−A", "B−A", "D−A", "D−B"], names), table_height(n),
        "Central band. Nominal: C−A 180, B−A 90, D−A 270, D−B 180."))

    out.append(_per_baseline(p2vm, np.rad2deg(p2vm.coherence_phase_rad) % 360,
                             "Output phases vs. wavelength", "Phase relative to A [deg]",
                             "Relative to output A."))
    if p2vm.dc_term_adu is not None:
        out.append(_per_baseline(p2vm, p2vm.dc_term_adu, "c-matrix vs. wavelength", "DC level [ADU]",
                                 "DC level of each output during its baseline's scan."))
    if p2vm.fit_rms_adu is not None:
        out.append(_per_baseline(p2vm, p2vm.fit_rms_adu, "Fringe-fit residual RMS", "Fit RMS [ADU]",
                                 "RMS of the scan frames about the fitted fringe."))
    out += _scan_sections(p2vm)
    return out


def _per_baseline(p2vm, values, title, ylabel, explanation) -> Section:
    wavelength = p2vm.wavelength_m * 1e9

    def draw(fig):
        for b, ax in enumerate(baseline_grid(fig)):
            for o, label, color in zip(p2vm.baseline_outputs[b], layout.abcd_order, ABCD_COLORS):
                ax.plot(wavelength, values[o, b], color=color, label=label)
            ax.set(title=baseline_name(b), xlabel="Wavelength [nm]", ylabel=ylabel)
        fig.axes[0].legend()
    return Section(title, draw, BASELINE_GRID_HEIGHT, explanation)


def _scan_sections(p2vm) -> list[Section]:
    """Calibration scan diagnostics, when the P2VM keeps them."""
    out = []
    wavelength = p2vm.wavelength_m * 1e9
    n_wave = len(p2vm.wavelength_m)
    channels = [int(n_wave * f) for f in (0.2, 0.5, 0.8)]
    if p2vm.opd_m is not None:
        def draw_delays(fig):
            for b, ax in enumerate(baseline_grid(fig)):
                ax.plot(p2vm.opd_m[b] * 1e6, label="OPD")
                if p2vm.group_delay_m is not None:
                    ax.plot(p2vm.group_delay_m[b] * 1e6, label="Group delay", alpha=0.7)
                ax.set(title=baseline_name(b), xlabel="Frame", ylabel="Delay [µm]")
            fig.axes[0].legend()
        out.append(Section("Recovered scan delays", draw_delays, BASELINE_GRID_HEIGHT,
                           "OPD of every scan frame and the group delay measured from it."))
    if p2vm.ellipse_phase_rad is None or p2vm.ellipse_amplitude is None:
        return out
    circle = np.linspace(0, 2 * np.pi, 200)

    def draw_circles(fig):
        for b, ax in enumerate(baseline_grid(fig)):
            for j, color in zip(channels, ("C0", "C2", "C3")):
                r, phi = p2vm.ellipse_amplitude[b, :, j], p2vm.ellipse_phase_rad[b, :, j]
                ax.scatter(r * np.cos(phi), r * np.sin(phi), s=1.5, color=color,
                           label=f"{wavelength[j]:.0f} nm")
            ax.plot(np.cos(circle), np.sin(circle), color="gray", lw=0.6)
            ax.set(title=baseline_name(b), xlabel="(C−A) normalized", ylabel="(D−B) normalized",
                   aspect="equal", xlim=(-1.5, 1.5), ylim=(-1.5, 1.5))
        fig.axes[0].legend()
    out.append(Section("ABCD fringe ellipse", draw_circles, 5.0,
                       "Every scan frame on the fitted unit circle at three wavelengths; a perfect "
                       "fit lies on the circle, clipping or extra signal distorts it."))
    if p2vm.opd_m is not None:
        def draw_residuals(fig):
            for b, ax in enumerate(baseline_grid(fig)):
                for j, color in zip(channels, ("C0", "C2", "C3")):
                    residual = phase_residual_rad(p2vm.ellipse_phase_rad[b, :, j], p2vm.opd_m[b],
                                                  p2vm.wavelength_m[j])
                    ax.plot(p2vm.opd_m[b] * 1e6, np.rad2deg(residual), ".", ms=1.5, color=color,
                            label=f"{wavelength[j]:.0f} nm")
                ax.axhline(0, color="gray", lw=0.6)
                ax.set(title=baseline_name(b), xlabel="Scan OPD [µm]", ylabel="Residual [deg]",
                       ylim=(-180, 180))
            fig.axes[0].legend()
        out.append(Section("Fringe phase residual along the scan", draw_residuals,
                           BASELINE_GRID_HEIGHT, "Phase of every frame minus 2π·OPD/λ, wrapped; "
                           "structure means dispersion, OPD errors or distorted fringes."))
    return out
