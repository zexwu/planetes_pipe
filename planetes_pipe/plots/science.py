"""SINGLE_SCI_P2VMRED pages: per-frame fluxes, coherent flux, VISAMP, phase, group delay."""
import numpy as np
from matplotlib import colormaps

from planetes_pipe.instrument import layout
from planetes_pipe.plots.pages import finish, page
from planetes_pipe.steps.science import P2VMRed


def pages(reduced: P2VMRed, label: str = "PLANETES", full: bool = False):
    """Pages for one P2VMRED; full=True adds differential phase every 10th frame and channel.

    Needs a complete P2VMRed (with group delay and differential phase).
    """
    yield _flux_page(reduced, label)
    yield _coherent_flux_page(reduced, label)
    yield _visibility_page(reduced, label)
    yield _phase_page(reduced, label)
    yield _group_delay_page(reduced, label)
    if full:
        n_frames, _, n_wave = reduced.differential_phase_deg.shape
        for frame in range(0, n_frames, 10):
            yield _spectral_phase_page(reduced, label, frame)
        for channel in range(0, n_wave, 10):
            yield _phase_vs_group_delay_page(reduced, label, channel)


def _grid(rows, columns, xlabel, ylabel, sharex=True, sharey=True):
    fig = page()
    axes = np.asarray(fig.subplots(rows, columns, sharex=sharex, sharey=sharey)).ravel()
    for ax in axes:
        ax.set(xlabel=xlabel, ylabel=ylabel)
        ax.label_outer()
    return fig, axes


def _baselines(xlabel, ylabel, **kwargs):
    return _grid(2, layout.n_baselines // 2, xlabel, ylabel, **kwargs)


def _annotate_mean(ax, mean: float) -> None:
    ax.text(0.5, 0.8, f"Mean: {mean:.2f}", transform=ax.transAxes, ha="center", va="center")


def _flux_page(reduced, label):
    fig, axes = _grid(layout.n_telescopes // 2, 2, "#Frame", "Flux [adu]")
    mean_flux = reduced.photometric_flux_adu.mean(axis=-1)
    for telescope, ax in enumerate(axes[: layout.n_telescopes]):
        mean = mean_flux[:, telescope].mean()
        ax.plot(mean_flux[:, telescope] - mean)
        ax.set_title(f"Telescope {layout.telescopes[telescope]}")
        _annotate_mean(ax, mean)
    return finish(fig, f"{label}\nwavelength-averaged flux per telescope", top=0.93)


def _coherent_flux_page(reduced, label):
    fig, axes = _baselines("#Frame", "abs(coherent flux) [adu]")
    mean_coherent = abs(reduced.coherent_flux_adu.mean(axis=-1))
    for baseline, ax in enumerate(axes):
        mean = mean_coherent[:, baseline].mean()
        ax.plot(mean_coherent[:, baseline] - mean)
        ax.set_title(f"Baseline {layout.baselines[baseline]}")
        _annotate_mean(ax, mean)
    return finish(fig, f"{label}\nwavelength-averaged coherent flux per baseline", top=0.93)


def _visibility_page(reduced, label):
    fig, axes = _baselines("Wavelength Index", "VISAMP")
    visamp = np.ma.array(reduced.visibility_amplitude, mask=reduced.flags)
    colors = colormaps["viridis"](np.linspace(0, 1, len(visamp)))
    for baseline, ax in enumerate(axes):
        ax.set_prop_cycle(color=colors)
        ax.plot(visamp[:, baseline, :].T, alpha=0.3, lw=0.3)
        ax.set_title(f"Baseline {layout.baselines[baseline]}")
        ax.set_ylim(0, 1.5)
    return finish(fig, f"{label}\nVISAMP per baseline", top=0.93)


def _phase_page(reduced, label):
    fig, axes = _baselines("#Frame", "arg(coherent flux) [deg]")
    mean_phase = np.unwrap(np.angle(reduced.coherent_flux_adu.mean(axis=-1), deg=True),
                           period=360, axis=0)
    for baseline, ax in enumerate(axes):
        ax.plot(mean_phase[:, baseline])
        ax.set_title(f"Baseline {layout.baselines[baseline]}")
    return finish(fig, f"{label}\nwavelength-averaged phase per baseline", top=0.93)


def _group_delay_page(reduced, label):
    fig, axes = _baselines("#Frame", "Group Delay [m]", sharey=False)
    for baseline, ax in enumerate(axes):
        ax.plot(reduced.group_delay_m[:, baseline])
        ax.set_title(f"Baseline {layout.baselines[baseline]}")
    return finish(fig, f"{label}\nGroup Delay per baseline", top=0.93)


def _spectral_phase_page(reduced, label, frame):
    fig, axes = _baselines("Wavelength Index", "Differential Phase [deg]")
    for baseline, ax in enumerate(axes):
        ax.plot(reduced.differential_phase_deg[frame, baseline])
        ax.set_title(f"Baseline {layout.baselines[baseline]}")
        ax.set_ylim(-10, 10)
    return finish(fig, f"{label}\ndifferential phase per baseline @ DIT {frame}", top=0.93)


def _phase_vs_group_delay_page(reduced, label, channel):
    fig, axes = _baselines("Group Delay [m]", "Differential Phase [deg]", sharex=False)
    for baseline, ax in enumerate(axes):
        ax.scatter(reduced.group_delay_m[:, baseline],
                   reduced.differential_phase_deg[:, baseline, channel], s=5, alpha=0.7)
        ax.set_title(f"Baseline {layout.baselines[baseline]}")
        ax.set_ylim(-10, 10)
    return finish(fig, f"{label}\ndifferential phase per baseline", top=0.93)
