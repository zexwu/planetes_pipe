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
    FLAG,
    GRID_HEIGHT,
    GUIDE,
    MARK,
    PASS,
    Section,
    abcd_entries,
    baseline_name,
    baseline_panels,
    definitions_section,
    legend,
    table,
    table_height,
)
from planetes_pipe.steps.p2vm import P2VM, normalized_coherence

PHOTOMETRY_RANGE = (0.7, 1.3)    # relative transmission of each output
COHERENCE_RANGE = (0.80, 1.03)   # normalized coherence
CENTRAL_BAND = (0.15, 0.85)      # fraction of the wavelength channels checked
ALLOWED_OUTSIDE = 0.05           # of one spectrum's channels, over all baselines
NOMINAL_PHASE_DEG = {"C": 180.0, "B": 90.0, "D": 270.0}  # relative to A


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


DEFINITIONS = (
    ("Symbols", [r"Output $o$, telescope $t$, baseline $b$ of telescopes $i,j$, wavelength $\lambda$."]),
    (r"Transmission $T_{ot}(\lambda)$",
     [r"$F_{ot}\,/\,\langle F_{ot}\rangle_o$, with $F_{ot}$ the median flat spectrum of telescope $t$ in output $o$.",
      "1 = the output gets the average share of that telescope's light at that wavelength."]),
    ("Relative transmission", [r"$\langle T_{ot}\rangle_{t,\lambda}$, per output; the mean over all outputs is 1."]),
    ("Share of output flux", [r"$T_{ot}\,/\,\sum_t T_{ot}$."]),
    ("Scan fit", [r"$I_o = c_{ob} + a_{ob}\cos(\phi - \theta_{ob})$ in the scan of $b$, with $\phi$ the measured fringe phase."]),
    ("Telescope flux $P_t$", [r"From the fitted DC levels: $c_{ob} = \sum_t T_{ot}\,P_t$."]),
    ("Coherence $C_{ob}$", [r"$a_{ob}\,/\,\sqrt{P_i P_j}$."]),
    ("Normalized coherence $V_{ob}$",
     [r"$C_{ob}\,/\,(2\sqrt{T_{oi}T_{oj}}) = a_{ob}\,/\,(2\sqrt{T_{oi}P_i\;T_{oj}P_j})$.",
      "Fringe contrast of the output during the scan; 1 = full contrast."]),
    ("Visibility transmission", [r"$V_{ob}$ averaged over the central band; per baseline also over A, C, B, D."]),
    ("Output phase", [r"$\theta_{ob} - \theta_{Ab}$, relative to output A of the baseline."]),
    ("Central band", [f"The middle {100 * (CENTRAL_BAND[1] - CENTRAL_BAND[0]):.0f} % of the wavelength channels."]),
)
VERDICT_COLORS = {"GOOD": PASS, "WARNING": "#FFF2CC", "BAD": FLAG}
CHANNEL_COLORS = ("#0C5DA5", "#00B945", "#FF2C00")  # three sample wavelengths


def sections(p2vm: P2VM, header=None) -> list[Section]:
    """Sections for one P2VM, in GRAVITY's order; header adds its QC to the summary."""
    result = quality(p2vm, header)
    wavelength = p2vm.wavelength_m * 1e9
    n = len(layout.baselines)
    names = [baseline_name(b) for b in range(n)]
    out = [definitions_section(DEFINITIONS)]

    # Quality summary (GRAVITY: "P2VM overall quality" and its checks).
    def draw_verdict(fig):
        ax = fig.subplots()
        ax.axis("off")
        ax.text(0.0, 0.5, f"P2VM overall quality:  {result['verdict']}", ha="left", va="center",
                fontsize=11, weight="bold",
                bbox={"facecolor": VERDICT_COLORS[result["verdict"]], "edgecolor": "none", "pad": 5})
    out.append(Section("Quality", draw_verdict, 0.4, "GRAVITY gravi_visual rules, plus raw "
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
        shade=[((i, 0), FLAG if f else PASS) for i, f in enumerate(result["checks"].values())]),
        table_height(len(rows)), "Values: outputs outside the range; coherence samples outside the "
        "range in the central band; saturated samples per raw input (QC.NSAT)."))

    phases = output_phases_deg(p2vm)
    per_baseline = [[f"{100 * result['visibility'][b]:.1f} %"] + [f"{phases[b, i]:.1f}" for i in (1, 2, 3)]
                    for b in range(n)]
    out.append(Section("Per baseline", lambda fig: table(
        fig, per_baseline, ["Vis. transm.", "C−A [deg]", "B−A [deg]", "D−A [deg]"], names,
        shade=[((b, 0), FLAG) for b in range(n) if result["visibility"][b] < COHERENCE_RANGE[0]]),
        table_height(n), f"Central band means; Vis. transm. is the visibility transmission, below "
        f"{100 * COHERENCE_RANGE[0]:.0f} % shaded. Nominal phases C−A 180, B−A 90, D−A 270."))

    # Photometry (GRAVITY: relative transmission per output; transmissions vs wavelength).
    def draw_relative_outputs(fig):
        ax = fig.subplots()
        order = [o for b in range(n) for o in p2vm.baseline_outputs[b]]
        values = relative[order]
        ok = (values >= PHOTOMETRY_RANGE[0]) & (values <= PHOTOMETRY_RANGE[1])
        ax.bar(range(len(order)), values, 0.7, color=np.where(ok, ABCD_COLORS[0], ABCD_COLORS[3]))
        for limit in PHOTOMETRY_RANGE:
            ax.axhline(limit, **GUIDE)
        ax.set_xticks([4 * b + 1.5 for b in range(n)], names, minor=False)
        ax.set_xticks(range(len(order)), minor=True)
        ax.tick_params(axis="x", which="major", length=0, top=False)
        for b in range(1, n):
            ax.axvline(4 * b - 0.5, color="0.85", lw=0.5)
        ax.set(ylabel="Relative transmission", xlim=(-0.6, len(order) - 0.4),
               ylim=(0, max(1.6, values.max() * 1.1)))
    out.append(Section("Relative photometric transmission per output", draw_relative_outputs, 2.2,
                       f"$\\langle T_{{ot}}\\rangle_{{t,\\lambda}}$ of outputs A, C, B, D of each baseline; "
                       f" allowed {PHOTOMETRY_RANGE[0]}–{PHOTOMETRY_RANGE[1]} (outside: red)."))

    def percent_cells():
        cells = []
        for b in range(n):
            flux = p2vm.transmission[p2vm.baseline_outputs[b]].sum(axis=(0, 2))
            cells.append([f"{100 * f / flux.sum():.1f}" for f in flux])
        return cells
    out.append(Section("Wavelength-averaged photometric transmission (%)", lambda fig: table(
        fig, percent_cells(), [f"T{t}" for t in layout.telescopes], names,
        shade=[((b, t), MARK) for b, pair in enumerate(layout.baseline_pairs) for t in pair]),
        table_height(n), "Share of each telescope in $\\sum T_{ot}$ over the baseline's four outputs and all wavelengths; the baseline's "
        "two telescopes shaded, the others are cross-talk."))

    share = p2vm.transmission / p2vm.transmission.sum(axis=1, keepdims=True)
    pair_entries = abcd_entries() + [("first telescope", {"color": "0.3"}),
                                     ("second telescope", {"color": "0.3", "ls": "--"})]
    for values, title, ylabel, explanation, ylim in (
            (p2vm.transmission, "Photometric transmission vs. wavelength", "Transmission",
             "$T_{ot}(\\lambda)$: for each telescope and wavelength the mean over the 24 outputs is 1. Solid: the "
             "baseline's first telescope, dashed: its second.", None),
            (share, "Relative photometric transmission vs. wavelength", "Share of output flux",
             "$T_{ot}\\,/\\,\\sum_t T_{ot}$; balanced = 0.5. Solid: first "
             "telescope, dashed: second.", (0, 1))):
        def draw(fig, values=values, ylabel=ylabel, ylim=ylim):
            axes = baseline_panels(fig, "Wavelength [nm]", ylabel, sharey=ylim is not None)
            for b, ax in enumerate(axes):
                t1, t2 = layout.baseline_pairs[b]
                for o, color in zip(p2vm.baseline_outputs[b], ABCD_COLORS):
                    ax.plot(wavelength, values[o, t1], color=color)
                    ax.plot(wavelength, values[o, t2], color=color, ls="--")
                if ylim:
                    ax.axhline(0.5, **GUIDE)
                    ax.set_ylim(*ylim)
            legend(fig, pair_entries)
        out.append(Section(title, draw, GRID_HEIGHT, explanation))

    # Coherence (GRAVITY: coherent flux matrix, visibility transmission, coherence vs wavelength).
    def matrix_cells():
        cells = []
        for b in range(n):
            flux = np.abs(p2vm.coherence_amplitude[p2vm.baseline_outputs[b]]).sum(axis=(0, 2))
            cells.append([f"{100 * f / flux.sum():.1f}" for f in flux])
        return cells
    out.append(Section("Wavelength-averaged coherent flux transmission (%)", lambda fig: table(
        fig, matrix_cells(), names, [f"outputs of {name}" for name in names],
        shade=[((b, b), MARK) for b in range(n)]),
        table_height(n), "Share of $|C_{ob}|$: each baseline (columns) in each baseline's outputs (rows); "
        "close to 100 on the diagonal and 0 elsewhere."))

    coherence = normalized_coherence(p2vm)
    band = central_band(p2vm)

    def draw_visibility(fig):
        ax = fig.subplots()
        mean = np.nanmean(coherence[..., band], axis=2)
        for i, color in enumerate(ABCD_COLORS):
            ax.bar(np.arange(n) + (i - 1.5) * 0.2, 100 * mean[:, i], 0.18, color=color)
        ax.axhline(100 * COHERENCE_RANGE[0], **GUIDE)
        ax.set_xticks(range(n), names)
        ax.set_xticks([], minor=True)
        ax.tick_params(axis="x", length=0, top=False)
        ax.set(ylabel="Visibility transmission [%]", ylim=(0, 110))
        legend(fig, abcd_entries(lw=4) + [(f"minimum {100 * COHERENCE_RANGE[0]:.0f} %", GUIDE)])
    out.append(Section("Wavelength-averaged visibility transmission", draw_visibility, 2.2,
                       "$V_{ob}$ averaged over the central band, per output; nominal "
                       f"higher than {100 * COHERENCE_RANGE[0]:.0f} %."))

    def draw_coherence(fig):
        axes = baseline_panels(fig, "Wavelength [nm]", "Normalized coherence", sharey=True)
        for b, ax in enumerate(axes):
            for values, color in zip(coherence[b], ABCD_COLORS):
                ax.plot(wavelength, values, color=color)
            ax.axvspan(wavelength[band][0], wavelength[band][-1], color="0.5", alpha=0.08, lw=0)
            ax.axhline(COHERENCE_RANGE[0], **GUIDE)
        legend(fig, abcd_entries() + [(f"{COHERENCE_RANGE[0]}", GUIDE)])
    out.append(Section("Coherence vs. wavelength", draw_coherence, GRID_HEIGHT,
                       "$V_{ob}(\\lambda)$ of each output; shaded: the checked central band. "
                       "Instrumental response, not target visibility."))

    # Phases (GRAVITY: phase table and output phases vs wavelength).
    phase_cells = [[f"{phases[b, 1]:.1f}", f"{phases[b, 2]:.1f}", f"{phases[b, 3]:.1f}",
                    f"{(phases[b, 3] - phases[b, 2]) % 360:.1f}"] for b in range(n)]
    out.append(Section("Wavelength-averaged phases of the outputs (deg, A = 0)", lambda fig: table(
        fig, phase_cells, ["C−A", "B−A", "D−A", "D−B"], names), table_height(n),
        "$\\theta_{ob} - \\theta_{Ab}$ over the central band. Nominal: C−A 180, B−A 90, D−A 270, D−B 180."))

    def draw_phases(fig):
        axes = baseline_panels(fig, "Wavelength [nm]", "Phase relative to A [deg]", sharey=True)
        for b, ax in enumerate(axes):
            for o, color in zip(p2vm.baseline_outputs[b], ABCD_COLORS):
                ax.plot(wavelength, np.rad2deg(p2vm.coherence_phase_rad[o, b]) % 360, color=color)
        axes[0].set(ylim=(-10, 370), yticks=[0, 90, 180, 270, 360])
        legend(fig, abcd_entries())
    out.append(Section("Output phases vs. wavelength", draw_phases, GRID_HEIGHT,
                       "$\\theta_{ob} - \\theta_{Ab}$; nominal C 180, B 90, D 270."))
    if p2vm.dc_term_adu is not None:
        out.append(_per_baseline(p2vm, p2vm.dc_term_adu, "c-matrix vs. wavelength", "DC level [ADU]",
                                 "$c_{ob}$: DC level of each output during its baseline's scan."))
    if p2vm.fit_rms_adu is not None:
        out.append(_per_baseline(p2vm, p2vm.fit_rms_adu, "Fringe-fit residual RMS", "Fit RMS [ADU]",
                                 "RMS of the scan frames about the fitted $c_{ob} + a_{ob}\\cos(\\phi - \\theta_{ob})$."))
    out += _scan_sections(p2vm)
    return out


def _per_baseline(p2vm, values, title, ylabel, explanation) -> Section:
    wavelength = p2vm.wavelength_m * 1e9

    def draw(fig):
        for b, ax in enumerate(baseline_panels(fig, "Wavelength [nm]", ylabel)):
            for o, color in zip(p2vm.baseline_outputs[b], ABCD_COLORS):
                ax.plot(wavelength, values[o, b], color=color)
        legend(fig, abcd_entries())
    return Section(title, draw, GRID_HEIGHT, explanation)


def _scan_sections(p2vm) -> list[Section]:
    """Calibration scan diagnostics, when the P2VM keeps them."""
    out = []
    wavelength = p2vm.wavelength_m * 1e9
    n_wave = len(p2vm.wavelength_m)
    channels = [int(n_wave * f) for f in (0.2, 0.5, 0.8)]
    channel_entries = [(f"{wavelength[j]:.0f} nm", {"color": c, "ls": "", "marker": "o", "ms": 3})
                       for j, c in zip(channels, CHANNEL_COLORS)]
    if p2vm.opd_m is not None:
        def draw_delays(fig):
            for b, ax in enumerate(baseline_panels(fig, "Scan frame", "Delay [µm]")):
                ax.plot(p2vm.opd_m[b] * 1e6, color=ABCD_COLORS[0])
                if p2vm.group_delay_m is not None:
                    ax.plot(p2vm.group_delay_m[b] * 1e6, color=ABCD_COLORS[3], lw=0.6)
            entries = [("OPD", {"color": ABCD_COLORS[0]})]
            if p2vm.group_delay_m is not None:
                entries.append(("group delay", {"color": ABCD_COLORS[3]}))
            legend(fig, entries)
        out.append(Section("Recovered scan delays", draw_delays, GRID_HEIGHT,
                           "OPD recovered from the fringe phase of every scan frame, and the group delay measured from the spectrum."))
    if p2vm.ellipse_phase_rad is None or p2vm.ellipse_amplitude is None:
        return out
    circle = np.linspace(0, 2 * np.pi, 200)

    def draw_circles(fig):
        axes = baseline_panels(fig, "(C−A) normalized", "(D−B) normalized", sharey=True)
        for b, ax in enumerate(axes):
            for j, color in zip(channels, CHANNEL_COLORS):
                r, phi = p2vm.ellipse_amplitude[b, :, j], p2vm.ellipse_phase_rad[b, :, j]
                ax.scatter(r * np.cos(phi), r * np.sin(phi), s=1, color=color, linewidths=0)
            ax.plot(np.cos(circle), np.sin(circle), color="0.3", lw=0.6)
            ax.set_aspect("equal")
        axes[0].set(xlim=(-1.5, 1.5), ylim=(-1.5, 1.5))
        legend(fig, channel_entries)
    out.append(Section("ABCD fringe ellipse", draw_circles, 4.4,
                       "Every scan frame on the fitted unit circle at three wavelengths; a perfect "
                       "fit lies on the circle, clipping or extra signal distorts it."))
    if p2vm.opd_m is not None:
        def draw_residuals(fig):
            axes = baseline_panels(fig, "Scan OPD [µm]", "Phase residual [deg]", sharex=False,
                                   sharey=True)
            for b, ax in enumerate(axes):
                for j, color in zip(channels, CHANNEL_COLORS):
                    residual = phase_residual_rad(p2vm.ellipse_phase_rad[b, :, j], p2vm.opd_m[b],
                                                  p2vm.wavelength_m[j])
                    ax.plot(p2vm.opd_m[b] * 1e6, np.rad2deg(residual), ".", ms=1.2, color=color)
                ax.axhline(0, **GUIDE)
            axes[0].set(ylim=(-180, 180), yticks=[-180, -90, 0, 90, 180])
            legend(fig, channel_entries)
        out.append(Section("Fringe phase residual along the scan", draw_residuals,
                           GRID_HEIGHT, "Phase of every frame minus 2π·OPD/λ, wrapped; "
                           "structure means dispersion, OPD errors or distorted fringes."))
    return out
