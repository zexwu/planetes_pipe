"""P2VM pages: a quality summary and the instrumental response, as GRAVITY's P2VM
report (gravi_visual_p2vm.py), plus diagnostics of the calibration scans.

The quality rules are GRAVITY's: each output's relative transmission within
PHOTOMETRY_RANGE, and the normalized coherence within COHERENCE_RANGE over the
central band except for ALLOWED_OUTSIDE of a spectrum. Saturated raw samples
(QC.NSAT in the product header) are a PLANETES addition.
"""
import numpy as np

from planetes_pipe.instrument import layout
from planetes_pipe.io.keywords import QC_SATURATED
from planetes_pipe.plots.pages import ABCD_COLORS, baseline_grid, baseline_name, finish, page, table
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


def pages(p2vm: P2VM, header=None):
    """Pages for one P2VM; header (the product's primary header) adds its QC to the summary."""
    yield _summary_page(p2vm, header)
    yield _relative_output_page(p2vm)
    yield from _transmission_pages(p2vm)
    yield _coherent_flux_matrix_page(p2vm)
    yield _visibility_transmission_page(p2vm)
    yield _coherence_page(p2vm)
    yield _phase_table_page(p2vm)
    yield _phase_page(p2vm)
    if p2vm.dc_term_adu is not None:
        yield _per_baseline_page(p2vm, p2vm.dc_term_adu, "DC level [ADU]",
                                 "P2VM c-matrix — DC level of each output during its baseline's scan")
    if p2vm.fit_rms_adu is not None:
        yield _per_baseline_page(p2vm, p2vm.fit_rms_adu, "Fit RMS [ADU]", "Fringe-fit residual RMS")
    if p2vm.opd_m is not None:
        yield _delay_page(p2vm)
    if p2vm.ellipse_phase_rad is not None and p2vm.ellipse_amplitude is not None:
        yield _unit_circle_page(p2vm)
        if p2vm.opd_m is not None:
            yield _phase_residual_page(p2vm)


def _summary_page(p2vm, header):
    result = quality(p2vm, header)
    colors = {"GOOD": "palegreen", "WARNING": "khaki", "BAD": "salmon"}
    fig = page()
    top, middle, bottom = fig.subplots(3, 1, height_ratios=[1, 2.2, 3])
    top.axis("off")
    top.text(0.5, 0.5, f"P2VM overall quality: {result['verdict']}", ha="center", va="center",
             fontsize=16, bbox={"facecolor": colors[result["verdict"]], "pad": 10})
    rows = list(result["checks"])
    outputs = [(o, f"{baseline_name(b)} {label}") for b in range(len(layout.baselines))
               for o, label in zip(p2vm.baseline_outputs[b], layout.abcd_order)]
    relative = result["relative_transmission"]
    low = [f"{n} {relative[o]:.2f}" for o, n in outputs if relative[o] < PHOTOMETRY_RANGE[0]]
    high = [f"{n} {relative[o]:.2f}" for o, n in outputs if relative[o] > PHOTOMETRY_RANGE[1]]
    detail = {rows[0]: f"{len(low)} outputs", rows[1]: f"{len(high)} outputs",
              rows[2]: str(result["coherence_low"]), rows[3]: str(result["coherence_high"]),
              rows[4]: ", ".join(f"{k} {v}" for k, v in result["saturated"].items()) or "none"}
    cells = [["FAIL" if failed else "ok", detail.get(name, "")] for name, failed in result["checks"].items()]
    table(middle, cells, ["Result", "Values"], rows,
          shade=[((i, 0), "salmon" if failed else "palegreen")
                 for i, failed in enumerate(result["checks"].values())])
    phases = output_phases_deg(p2vm)
    columns = ["Visibility transmission"] + [f"{label}−A [deg]" for label in "CBD"]
    cells = [[f"{100 * result['visibility'][b]:.1f} %"]
             + [f"{phases[b, i]:.1f}" for i in (1, 2, 3)] for b in range(len(layout.baselines))]
    table(bottom, cells, columns, [baseline_name(b) for b in range(len(layout.baselines))],
          shade=[((b, 0), "salmon") for b in range(len(layout.baselines))
                 if result["visibility"][b] < COHERENCE_RANGE[0]])
    bottom.set_title("Central band: mean normalized coherence and output phases", fontsize=9)
    return finish(fig, "P2VM quality (GRAVITY gravi_visual rules; saturation added)")


def _relative_output_page(p2vm):
    relative = relative_output_transmission(p2vm)
    fig = page()
    ax = fig.subplots()
    names = [f"{baseline_name(b)} {label}" for b in range(len(layout.baselines))
             for label in layout.abcd_order]
    order = [o for b in range(len(layout.baselines)) for o in p2vm.baseline_outputs[b]]
    values = relative[order]
    ok = (values >= PHOTOMETRY_RANGE[0]) & (values <= PHOTOMETRY_RANGE[1])
    ax.bar(range(len(order)), values, color=np.where(ok, "steelblue", "salmon"))
    for limit in PHOTOMETRY_RANGE:
        ax.axhline(limit, color="gray", ls="--", lw=0.8)
    ax.set_xticks(range(len(order)), names, rotation=90, fontsize=7)
    ax.set(ylabel="Relative transmission", ylim=(0, max(2, values.max() * 1.1)))
    return finish(fig, f"Relative photometric transmission per output (allowed "
                       f"{PHOTOMETRY_RANGE[0]}–{PHOTOMETRY_RANGE[1]})")


def _transmission_pages(p2vm):
    wavelength = p2vm.wavelength_m * 1e9
    share = p2vm.transmission / p2vm.transmission.sum(axis=1, keepdims=True)
    for values, ylabel, title, ylim in (
            (p2vm.transmission, "Transmission", "P2VM transmission — share of each telescope in "
             "each output; for every telescope and wavelength the mean over the 24 outputs is 1",
             None),
            (share, "Share of output flux", "P2VM relative transmission — each telescope / all "
             "four telescopes in the output (balanced = 0.5)", (0, 1))):
        fig = page()
        for b, ax in enumerate(baseline_grid(fig)):
            t1, t2 = layout.baseline_pairs[b]
            first, second = layout.baselines[b]
            for o, label, color in zip(p2vm.baseline_outputs[b], layout.abcd_order, ABCD_COLORS):
                ax.plot(wavelength, values[o, t1], color=color, label=f"{label} T{first}")
                ax.plot(wavelength, values[o, t2], color=color, ls="--", label=f"{label} T{second}")
            if ylim:
                ax.axhline(0.5, color="gray", lw=0.7)
                ax.set_ylim(*ylim)
            ax.set(title=f"{baseline_name(b)}: T{first} solid, T{second} dashed",
                   xlabel="Wavelength [nm]", ylabel=ylabel)
        fig.axes[0].legend(fontsize=7, ncol=2)
        yield finish(fig, title)

    fig = page(figsize=(11.7, 4))
    cells = []
    for b in range(len(layout.baselines)):
        flux = p2vm.transmission[p2vm.baseline_outputs[b]].sum(axis=(0, 2))
        cells.append([f"{100 * f / flux.sum():.1f}" for f in flux])
    table(fig.subplots(), cells, [f"T{t}" for t in layout.telescopes],
          [baseline_name(b) for b in range(len(layout.baselines))],
          shade=[((b, t), "lavender") for b, pair in enumerate(layout.baseline_pairs) for t in pair])
    yield finish(fig, "Wavelength-averaged photometric transmission per baseline (%)\nshare of "
                      "each telescope in the four outputs; its two telescopes shaded", top=0.85)


def _coherent_flux_matrix_page(p2vm):
    n = len(layout.baselines)
    cells, shade = [], []
    for b in range(n):
        flux = np.abs(p2vm.coherence_amplitude[p2vm.baseline_outputs[b]]).sum(axis=(0, 2))
        cells.append([f"{100 * f / flux.sum():.1f}" for f in flux])
        shade.append(((b, b), "lavender"))
    fig = page(figsize=(11.7, 4))
    table(fig.subplots(), cells, [baseline_name(b) for b in range(n)],
          [f"outputs of {baseline_name(b)}" for b in range(n)], shade=shade)
    return finish(fig, "Wavelength-averaged coherent flux transmission (%)\ncoherent flux of "
                       "each baseline (columns) in each baseline's outputs; ~100 on the diagonal, "
                       "0 elsewhere", top=0.85)


def _visibility_transmission_page(p2vm):
    coherence = normalized_coherence(p2vm)[..., central_band(p2vm)]
    mean = np.nanmean(coherence, axis=2)  # (baseline, output)
    fig = page()
    ax = fig.subplots()
    width = 0.2
    for i, (label, color) in enumerate(zip(layout.abcd_order, ABCD_COLORS)):
        ax.bar(np.arange(len(layout.baselines)) + (i - 1.5) * width, 100 * mean[:, i], width,
               color=color, label=label)
    ax.axhline(100 * COHERENCE_RANGE[0], color="gray", ls="--", lw=0.8,
               label=f"nominal minimum {100 * COHERENCE_RANGE[0]:.0f} %")
    ax.set_xticks(range(len(layout.baselines)),
                  [baseline_name(b) for b in range(len(layout.baselines))])
    ax.set(ylabel="Visibility transmission [%]", ylim=(0, 110))
    ax.legend(ncol=5)
    return finish(fig, "Wavelength-averaged visibility transmission (central band) — instrumental "
                       "transfer function of the normalized fringe amplitude")


def _coherence_page(p2vm):
    wavelength = p2vm.wavelength_m * 1e9
    band = central_band(p2vm)
    coherence = normalized_coherence(p2vm)
    fig = page()
    for b, ax in enumerate(baseline_grid(fig)):
        for values, label, color in zip(coherence[b], layout.abcd_order, ABCD_COLORS):
            ax.plot(wavelength, values, color=color, label=label)
        ax.axvspan(wavelength[band][0], wavelength[band][-1], color="gray", alpha=0.08)
        ax.axhline(COHERENCE_RANGE[0], color="gray", ls="--", lw=0.8)
        ax.set(title=baseline_name(b), xlabel="Wavelength [nm]", ylabel="Normalized coherence")
    fig.axes[0].legend()
    return finish(fig, "P2VM coherence — instrumental response, not target visibility "
                       f"(shaded: checked band; dashed: {COHERENCE_RANGE[0]})")


def _phase_table_page(p2vm):
    phases = output_phases_deg(p2vm)
    columns = [f"{label}−A" for label in "CBD"] + ["D−B"]
    cells = [[f"{phases[b, 1]:.1f}", f"{phases[b, 2]:.1f}", f"{phases[b, 3]:.1f}",
              f"{(phases[b, 3] - phases[b, 2]) % 360:.1f}"] for b in range(len(layout.baselines))]
    fig = page(figsize=(11.7, 4))
    table(fig.subplots(), cells, columns, [baseline_name(b) for b in range(len(layout.baselines))])
    return finish(fig, "Wavelength-averaged phases of the outputs (deg, A = 0, central band); "
                       "nominal C−A 180, B−A 90, D−A 270, D−B 180", top=0.9)


def _phase_page(p2vm):
    wavelength = p2vm.wavelength_m * 1e9
    fig = page()
    for b, ax in enumerate(baseline_grid(fig)):
        for o, label, color in zip(p2vm.baseline_outputs[b], layout.abcd_order, ABCD_COLORS):
            ax.plot(wavelength, np.rad2deg(p2vm.coherence_phase_rad[o, b]) % 360, color=color,
                    label=label)
        ax.set(title=baseline_name(b), xlabel="Wavelength [nm]", ylabel="Phase relative to A [deg]")
    fig.axes[0].legend()
    return finish(fig, "P2VM phase — instrumental response, not target visibility")


def _per_baseline_page(p2vm, values, ylabel, title):
    wavelength = p2vm.wavelength_m * 1e9
    fig = page()
    for b, ax in enumerate(baseline_grid(fig)):
        for o, label, color in zip(p2vm.baseline_outputs[b], layout.abcd_order, ABCD_COLORS):
            ax.plot(wavelength, values[o, b], color=color, label=label)
        ax.set(title=baseline_name(b), xlabel="Wavelength [nm]", ylabel=ylabel)
    fig.axes[0].legend()
    return finish(fig, title)


def _delay_page(p2vm):
    fig = page()
    for b, ax in enumerate(baseline_grid(fig)):
        ax.plot(p2vm.opd_m[b] * 1e6, label="OPD")
        if p2vm.group_delay_m is not None:
            ax.plot(p2vm.group_delay_m[b] * 1e6, label="Group delay", alpha=0.7)
        ax.set(title=baseline_name(b), xlabel="Frame", ylabel="Delay [µm]")
    fig.axes[0].legend()
    return finish(fig, "Recovered scan delays")


def _sample_channels(p2vm):
    n = len(p2vm.wavelength_m)
    return [int(n * f) for f in (0.2, 0.5, 0.8)]


def _unit_circle_page(p2vm):
    wavelength = p2vm.wavelength_m * 1e9
    circle = np.linspace(0, 2 * np.pi, 200)
    fig = page()
    for b, ax in enumerate(baseline_grid(fig)):
        for j, color in zip(_sample_channels(p2vm), ("C0", "C2", "C3")):
            r, phi = p2vm.ellipse_amplitude[b, :, j], p2vm.ellipse_phase_rad[b, :, j]
            ax.scatter(r * np.cos(phi), r * np.sin(phi), s=3, color=color,
                       label=f"{wavelength[j]:.0f} nm")
        ax.plot(np.cos(circle), np.sin(circle), color="gray", lw=0.7)
        ax.set(title=baseline_name(b), xlabel="(C−A) normalized", ylabel="(D−B) normalized",
               aspect="equal", xlim=(-1.6, 1.6), ylim=(-1.6, 1.6))
    fig.axes[0].legend(fontsize=7)
    return finish(fig, "ABCD fringe ellipse fitted to the unit circle, every scan frame — a "
                       "perfect fit lies on the circle; clipping or extra signal distorts it")


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


def _phase_residual_page(p2vm):
    wavelength = p2vm.wavelength_m * 1e9
    fig = page()
    for b, ax in enumerate(baseline_grid(fig)):
        for j, color in zip(_sample_channels(p2vm), ("C0", "C2", "C3")):
            residual = phase_residual_rad(p2vm.ellipse_phase_rad[b, :, j], p2vm.opd_m[b],
                                          p2vm.wavelength_m[j])
            ax.plot(p2vm.opd_m[b] * 1e6, np.rad2deg(residual), ".", ms=2, color=color,
                    label=f"{wavelength[j]:.0f} nm")
        ax.axhline(0, color="gray", lw=0.7)
        ax.set(title=baseline_name(b), xlabel="Scan OPD [µm]", ylabel="Phase residual [deg]",
               ylim=(-180, 180))
    fig.axes[0].legend(fontsize=7)
    return finish(fig, "Fringe phase of every scan frame minus 2π·OPD/λ (wrapped) — structure "
                       "along the scan means dispersion, OPD errors or distorted fringes")
