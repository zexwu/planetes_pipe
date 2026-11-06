"""SINGLE_SCI_P2VMRED sections: per-frame fluxes, coherent flux, VISAMP, phase, group delay."""
import numpy as np
from matplotlib import colormaps

from planetes_pipe.instrument import layout
from planetes_pipe.plots.pages import (
    GRID_HEIGHT,
    Section,
    baseline_name,
    definitions_section,
    name,
    panels,
)
from planetes_pipe.steps.science import P2VMRed

DEFINITIONS = (
    ("Reduction", [r"At each wavelength, $(P_t,\,C_{ij})$ = P2VM matrix $\times$ the 24 output fluxes of the frame."]),
    ("Flux $P_t$", ["Photometric flux of telescope $t$, in ADU."]),
    ("Coherent flux $C_{ij}$", ["Complex coherent flux of the baseline of telescopes $i,j$, in ADU."]),
    ("VISAMP", [r"$|C_{ij}|\,/\,\sqrt{P_i P_j}$, relative to the calibration: 1 = as coherent as the P2VM source."]),
    ("Group delay GD", [r"The delay that maximizes $|\sum_\lambda C_{ij}(\lambda)\,\exp(-2\pi i\,\mathrm{GD}/\lambda)|$."]),
    ("Differential phase", [r"$\arg[\,C_{ij}\exp(-2\pi i\,\mathrm{GD}/\lambda)\,]$, minus its mean over wavelength."]),
)


def sections(reduced: P2VMRed, full: bool = False) -> list[Section]:
    """Sections for one P2VMRED; full=True adds differential phase every 10th frame and channel.

    Needs a complete P2VMRed (with group delay and differential phase).
    """
    out = [definitions_section(DEFINITIONS),
           Section("SC flux vs. time (OI_FLUX wavelength averaged)", lambda fig: _flux(fig, reduced), 3.4,
                   "In ADU/DIT vs. frame: $\\langle P_t\\rangle_\\lambda$ minus its mean over frames (printed)."),
           Section("SC coherent flux vs. time (VISDATA wavelength averaged)", lambda fig: _coherent_flux(fig, reduced), GRID_HEIGHT,
                   "In ADU/DIT vs. frame: $|\\langle C_{ij}\\rangle_\\lambda|$ minus its mean over frames (printed)."),
           Section("SC visibility amplitude (VISAMP)", lambda fig: _visibility(fig, reduced), GRID_HEIGHT,
                   "$|C_{ij}|\\,/\\,\\sqrt{P_i P_j}$ of every frame (colour: frame order); flagged samples masked."),
           Section("arg(VISDATA) vs. time", lambda fig: _phase(fig, reduced), GRID_HEIGHT,
                   "$\\arg\\langle C_{ij}\\rangle_\\lambda$ in degrees vs. frame, unwrapped over frames."),
           Section("SC group delay vs. time (GDELAY)", lambda fig: _group_delay(fig, reduced), GRID_HEIGHT,
                   "GD of every frame vs. frame, as defined above.")]
    if full:
        n_frames, _, n_wave = reduced.differential_phase_deg.shape
        for frame in range(0, n_frames, 10):
            out.append(Section(f"Differential phase, frame {frame}",
                               lambda fig, f=frame: _spectral_phase(fig, reduced, f), GRID_HEIGHT))
        for channel in range(0, n_wave, 10):
            out.append(Section(f"Differential phase vs. group delay, channel {channel}",
                               lambda fig, c=channel: _phase_vs_gd(fig, reduced, c), GRID_HEIGHT))
    return out


def _grid(fig, rows, columns, xlabel, ylabel, sharex=True, sharey=True):
    return panels(fig, rows, columns, xlabel, ylabel, sharex=sharex, sharey=sharey)


def _baselines(fig, xlabel, ylabel, **kwargs):
    return _grid(fig, 2, layout.n_baselines // 2, xlabel, ylabel, **kwargs)


def _mean_label(ax, mean: float) -> None:
    name(ax, f"mean {mean:.2f}", "upper right")


def _flux(fig, reduced):
    mean_flux = reduced.photometric_flux_adu.mean(axis=-1)
    axes = _grid(fig, layout.n_telescopes // 2, 2, "Frame", "Flux − mean [ADU]")
    for telescope, ax in enumerate(axes[: layout.n_telescopes]):
        mean = mean_flux[:, telescope].mean()
        ax.plot(mean_flux[:, telescope] - mean)
        name(ax, f"T{layout.telescopes[telescope]}")
        _mean_label(ax, mean)


def _coherent_flux(fig, reduced):
    mean_coherent = abs(reduced.coherent_flux_adu.mean(axis=-1))
    for baseline, ax in enumerate(_baselines(fig, "Frame", "|Coherent flux| − mean [ADU]")):
        mean = mean_coherent[:, baseline].mean()
        ax.plot(mean_coherent[:, baseline] - mean)
        name(ax, baseline_name(baseline))
        _mean_label(ax, mean)


def _visibility(fig, reduced):
    visamp = np.ma.array(reduced.visibility_amplitude, mask=reduced.flags)
    colors = colormaps["viridis"](np.linspace(0, 1, len(visamp)))
    for baseline, ax in enumerate(_baselines(fig, "Wavelength index", "VISAMP")):
        ax.set_prop_cycle(color=colors)
        ax.plot(visamp[:, baseline, :].T, alpha=0.3, lw=0.3)
        name(ax, baseline_name(baseline))
        ax.set_ylim(0, 1.5)


def _phase(fig, reduced):
    mean_phase = np.unwrap(np.angle(reduced.coherent_flux_adu.mean(axis=-1), deg=True),
                           period=360, axis=0)
    for baseline, ax in enumerate(_baselines(fig, "Frame", "arg(coherent flux) [deg]")):
        ax.plot(mean_phase[:, baseline])
        name(ax, baseline_name(baseline))


def _group_delay(fig, reduced):
    for baseline, ax in enumerate(_baselines(fig, "Frame", "Group delay [m]", sharey=False)):
        ax.plot(reduced.group_delay_m[:, baseline])
        name(ax, baseline_name(baseline))


def _spectral_phase(fig, reduced, frame):
    for baseline, ax in enumerate(_baselines(fig, "Wavelength index", "Differential phase [deg]")):
        ax.plot(reduced.differential_phase_deg[frame, baseline])
        name(ax, baseline_name(baseline))
        ax.set_ylim(-10, 10)


def _phase_vs_gd(fig, reduced, channel):
    for baseline, ax in enumerate(_baselines(fig, "Group delay [m]", "Differential phase [deg]",
                                             sharex=False)):
        ax.scatter(reduced.group_delay_m[:, baseline],
                   reduced.differential_phase_deg[:, baseline, channel], s=4, alpha=0.7)
        name(ax, baseline_name(baseline))
        ax.set_ylim(-10, 10)
