"""SINGLE_SCI_P2VMRED sections, following GRAVITY's gravi_visual_p2vmreduced.

How a quantity is shown depends on what the comparison needs:

- one axis, coloured by telescope or baseline: series with one unit and separated
  levels, compared with each other (flux, coherent flux, SNR, spectra);
- waterfall (wavelength against time): anything that varies in both (flux, VISAMP,
  differential phase);
- rows with a shared time axis and their own y scale: signals that drift by different
  amounts and would hide each other on one axis (phase, group delay).
"""
import numpy as np

from planetes_pipe.instrument import layout
from planetes_pipe.plots.pages import (
    COLORS,
    GRID_HEIGHT,
    Section,
    baseline_name,
    definitions_section,
    legend,
    name,
    panels,
)
from planetes_pipe.steps.science import GAIN_E_PER_ADU, TRIPLETS, P2VMRed, closure, coherent_snr

STACK_HEIGHT = 4.4                       # six rows sharing a time axis, inches
WATERFALL_HEIGHT = 4.2

DEFINITIONS = (
    ("Reduction", [r"At each wavelength, $(P_t,\,C_{ij})$ = P2VM matrix $\times$ the 24 output fluxes of the frame."]),
    ("Flux $P_t$ (FLUX)", ["Photometric flux of telescope $t$, in ADU/DIT."]),
    ("Coherent flux $C_{ij}$ (VISDATA)", ["Complex coherent flux of the baseline of telescopes $i,j$, in ADU/DIT."]),
    ("VISAMP", [r"$|C_{ij}|\,/\,\sqrt{P_i P_j}$, relative to the calibration: 1 = as coherent as the P2VM source."]),
    ("F1F2", [r"$P_i P_j$, in the product."]),
    ("Group delay GD (GDELAY)", [r"The delay that maximizes $|\sum_\lambda C_{ij}(\lambda)\,\exp(-2\pi i\,\mathrm{GD}/\lambda)|$."]),
    ("Differential phase", [r"$\arg[\,C_{ij}\exp(-2\pi i\,\mathrm{GD}/\lambda)\,]$, minus its mean over wavelength."]),
    ("Closure phase (T3PHI)", [r"$\arg(C_{ij}\,C_{jk}\,C_{ik}^{*})$ of telescopes $i,j,k$."]),
    ("Errors (FLUXERR, VISERR)",
     [rf"Output variance: signal / gain + read noise from the DARK RMS (gain {GAIN_E_PER_ADU:.0f} e$^-$/ADU), through P2VM$^2$."]),
    ("SNR", [r"$|\sum_\lambda C_{ij}|\,/\,\sqrt{\sum_\lambda |\sigma_{C_{ij}}|^2}$ of the frame."]),
    ("CHI2", ["Reduced $\\chi^2$ of the outputs against the P2VM model of the reduced channels."]),
)


def sections(reduced: P2VMRed, full: bool = False) -> list[Section]:
    """Sections for one P2VMRED; full=True adds differential phase every 10th frame and channel.

    Needs a complete P2VMRed (with group delay and differential phase).
    """
    unit = "s" if reduced.time_s is not None else "frame"
    out = [definitions_section(DEFINITIONS),
           Section("SC Average spectrum (OI_FLUX)", lambda fig: _spectrum(fig, reduced), 2.4,
                   "In ADU/DIT vs. wavelength, averaged over frames."),
           Section("SC flux vs. time (OI_FLUX wavelength averaged)", lambda fig: _flux(fig, reduced), 2.4,
                   f"In ADU/DIT vs. {unit}."),
           Section("SC flux waterfall per telescope (OI_FLUX)", lambda fig: _flux_waterfall(fig, reduced),
                   WATERFALL_HEIGHT, "In ADU/DIT. The average over the sequence has been subtracted "
                   "and is printed; same colour scale for all telescopes."),
           Section("SC coherent flux vs. time (VISDATA wavelength averaged)",
                   lambda fig: _coherent_flux(fig, reduced), 2.4,
                   f"$|\\langle C_{{ij}}\\rangle_\\lambda|$ in ADU/DIT vs. {unit}."),
           ]
    if reduced.coherent_err_adu is not None:
        out.append(Section("SC coherent flux SNR vs. time (OI_VIS/SNR)", lambda fig: _snr(fig, reduced), 2.4,
                           f"SNR of each frame vs. {unit}."))
    out += [Section("SC visibility amplitude waterfall (VISAMP)", lambda fig: _visibility(fig, reduced),
                    WATERFALL_HEIGHT, "$|C_{ij}|\\,/\\,\\sqrt{P_i P_j}$; flagged samples masked."),
            Section("SC visibility amplitude vs. wavelength (VISAMP averaged over frames)",
                    lambda fig: _mean_visibility(fig, reduced), 2.4, "Unflagged frames."),
            Section("arg(VISDATA) vs. time", lambda fig: _phase(fig, reduced), STACK_HEIGHT,
                    f"$\\arg\\langle C_{{ij}}\\rangle_\\lambda$ in degrees vs. {unit}, unwrapped; each row has its own scale."),
            Section("SC group delay vs. time (GDELAY)", lambda fig: _group_delay(fig, reduced), STACK_HEIGHT,
                    f"GD in $\\mu$m vs. {unit}; each row has its own scale."),
            Section("SC differential phase waterfall (DIFFPHASE)", lambda fig: _phase_waterfall(fig, reduced),
                    WATERFALL_HEIGHT, "In degrees; colour range 1–99 %, symmetric about zero."),
            Section("SC closure phase vs. time (OI_T3 wavelength averaged)", lambda fig: _closure(fig, reduced),
                    2.4, f"Circular mean of T3PHI over wavelength, in degrees vs. {unit}.")]
    if reduced.chi2 is not None:
        out.append(Section("SC reduced chi2 vs. wavelength (CHI2)", lambda fig: _chi2(fig, reduced), 2.0,
                           "Median over frames; about 1 when the P2VM model and noise model describe the outputs."))
    if full:
        n_frames, _, n_wave = reduced.differential_phase_deg.shape
        for frame in range(0, n_frames, 10):
            out.append(Section(f"Differential phase, frame {frame}",
                               lambda fig, f=frame: _spectral_phase(fig, reduced, f), GRID_HEIGHT))
        for channel in range(0, n_wave, 10):
            out.append(Section(f"Differential phase vs. group delay, channel {channel}",
                               lambda fig, c=channel: _phase_vs_gd(fig, reduced, c), GRID_HEIGHT))
    return out


def _time(reduced):
    """x values and label of a per-frame plot: seconds when the product records them."""
    if reduced.time_s is not None:
        return reduced.time_s, "Time [s]"
    return np.arange(len(reduced.coherent_flux_adu)), "Frame"


def _wavelength_nm(reduced):
    return reduced.wavelength_m * 1e9


def _telescope_entries():
    return [(f"T{t}", {"color": color}) for t, color in zip(layout.telescopes, COLORS)]


def _baseline_entries():
    return [(baseline_name(b), {"color": COLORS[b]}) for b in range(len(layout.baselines))]


def _overlay(fig, x, series, xlabel, ylabel, entries, **limits):
    """One axis, one coloured line per series; series is (series, x)."""
    ax = fig.subplots()
    for values, (_, style) in zip(series, entries):
        ax.plot(x, values, lw=0.6, color=style["color"])
    ax.set(xlabel=xlabel, ylabel=ylabel, xlim=(x[0], x[-1]), **limits)
    legend(fig, entries)
    return ax


def _waterfall(fig, reduced, images, labels, rows, columns, colorbar, **image_options):
    """Wavelength horizontal, time vertical; one colour scale and bar for the panels."""
    wavelength, time = _wavelength_nm(reduced), _time(reduced)
    extent = (wavelength[0], wavelength[-1], time[0][0], time[0][-1])
    axes = panels(fig, rows, columns, "Wavelength [nm]", time[1], sharey=True)
    for ax, image, label in zip(axes, images, labels):
        shown = ax.imshow(image, origin="lower", aspect="auto", extent=extent,
                          interpolation="nearest", **image_options)
        ax.minorticks_off()
        name(ax, label)
    fig.colorbar(shown, ax=list(axes), label=colorbar, pad=0.02)


def _stack(fig, reduced, series, ylabel):
    """Six rows sharing the time axis, each with its own y scale."""
    x, xlabel = _time(reduced)
    axes = panels(fig, len(layout.baselines), 1, xlabel, ylabel)
    for b, ax in enumerate(axes):
        ax.plot(x, series[:, b], lw=0.5, color=COLORS[b])
        ax.set_xlim(x[0], x[-1])
        name(ax, baseline_name(b))


def _spectrum(fig, reduced):
    _overlay(fig, _wavelength_nm(reduced), reduced.photometric_flux_adu.mean(axis=0),
             "Wavelength [nm]", "Flux [ADU/DIT]", _telescope_entries())


def _flux(fig, reduced):
    x, xlabel = _time(reduced)
    flux = reduced.photometric_flux_adu.mean(axis=-1).T
    _overlay(fig, x, flux, xlabel, "Flux [ADU/DIT]", _telescope_entries(), ylim=(0, flux.max() * 1.1))


def _flux_waterfall(fig, reduced):
    flux = reduced.photometric_flux_adu
    means = flux.mean(axis=(0, 2), keepdims=True)
    relative = flux - means
    low, high = np.nanpercentile(relative, [1, 99])
    labels = [f"T{t}  mean {m:.3g}" for t, m in zip(layout.telescopes, means.ravel())]
    _waterfall(fig, reduced, relative.transpose(1, 0, 2), labels, 2, 2, "Flux − mean [ADU/DIT]",
               vmin=low, vmax=high, cmap="cubehelix")


def _coherent_flux(fig, reduced):
    x, xlabel = _time(reduced)
    _overlay(fig, x, np.abs(reduced.coherent_flux_adu.mean(axis=-1)).T, xlabel,
             "|Coherent flux| [ADU/DIT]", _baseline_entries(), ylim=(0, None))


def _snr(fig, reduced):
    x, xlabel = _time(reduced)
    ax = _overlay(fig, x, coherent_snr(reduced).T, xlabel, "SNR", _baseline_entries())
    ax.set_yscale("log")


def _visibility(fig, reduced):
    visamp = np.ma.array(reduced.visibility_amplitude, mask=reduced.flags)
    _waterfall(fig, reduced, visamp.transpose(1, 0, 2),
               [baseline_name(b) for b in range(len(layout.baselines))], 2, 3, "VISAMP",
               vmin=0, vmax=1.2, cmap="cubehelix")


def _mean_visibility(fig, reduced):
    visamp = np.ma.array(reduced.visibility_amplitude, mask=reduced.flags).mean(axis=0)
    _overlay(fig, _wavelength_nm(reduced), visamp, "Wavelength [nm]", "VISAMP", _baseline_entries(),
             ylim=(0, 1.3))


def _phase(fig, reduced):
    phase = np.unwrap(np.angle(reduced.coherent_flux_adu.mean(axis=-1), deg=True), period=360, axis=0)
    _stack(fig, reduced, phase, "arg(coherent flux) [deg]")


def _group_delay(fig, reduced):
    _stack(fig, reduced, reduced.group_delay_m * 1e6, r"Group delay [$\mu$m]")


def _phase_waterfall(fig, reduced):
    phase = reduced.differential_phase_deg
    limit = max(np.abs(np.nanpercentile(phase, [1, 99])))
    _waterfall(fig, reduced, phase.transpose(1, 0, 2),
               [baseline_name(b) for b in range(len(layout.baselines))], 2, 3,
               "Differential phase [deg]", vmin=-limit, vmax=limit, cmap="RdBu_r")


def _closure(fig, reduced):
    x, xlabel = _time(reduced)
    phase = np.angle(np.exp(1j * np.deg2rad(closure(reduced)[1])).mean(axis=-1), deg=True)
    entries = [("T" + "-".join(t), {"color": COLORS[n]}) for n, t in enumerate(TRIPLETS)]
    _overlay(fig, x, phase.T, xlabel, "Closure phase [deg]", entries, ylim=(-180, 180),
             yticks=[-180, -90, 0, 90, 180])


def _chi2(fig, reduced):
    ax = fig.subplots()
    ax.plot(_wavelength_nm(reduced), np.nanmedian(reduced.chi2, axis=0), color=COLORS[0])
    ax.set(xlabel="Wavelength [nm]", ylabel="Reduced chi2")
    ax.set_yscale("log")


def _baselines(fig, xlabel, ylabel, **kwargs):
    return panels(fig, 2, layout.n_baselines // 2, xlabel, ylabel, sharey=True, **kwargs)


def _spectral_phase(fig, reduced, frame):
    wavelength = _wavelength_nm(reduced)
    for baseline, ax in enumerate(_baselines(fig, "Wavelength [nm]", "Differential phase [deg]")):
        ax.plot(wavelength, reduced.differential_phase_deg[frame, baseline])
        name(ax, baseline_name(baseline))
        ax.set_ylim(-10, 10)


def _phase_vs_gd(fig, reduced, channel):
    for baseline, ax in enumerate(_baselines(fig, r"Group delay [$\mu$m]", "Differential phase [deg]",
                                             sharex=False)):
        ax.scatter(reduced.group_delay_m[:, baseline] * 1e6,
                   reduced.differential_phase_deg[:, baseline, channel], s=4, alpha=0.7)
        name(ax, baseline_name(baseline))
        ax.set_ylim(-10, 10)
