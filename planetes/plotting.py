"""Diagnostic plots for reduced targets."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from . import log
from .instrument import BASELINES, N_BASELINES, N_TELESCOPES, TELESCOPES
from .products import P2VMRed


def _make_figure_grid(rows: int, columns: int, xlabel: str, ylabel: str, **kwargs):
    """Create a labelled grid of diagnostic axes."""
    options = {"sharex": True, "sharey": True, **kwargs}
    fig, axes = plt.subplots(rows, columns, **options)
    axes = np.asarray(axes).ravel()
    for axis in axes:
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.label_outer()
    return fig, axes


def save_reduction_plots(
    reduced: P2VMRed,
    output: Path,
    *,
    object_label: str = "PLANETES",
    full: bool = False,
) -> None:
    """Write the existing diagnostic plots from an in-memory reduced product."""
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing diagnostic: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    flux_adu = reduced.photometric_flux_adu
    coherent_adu = reduced.coherent_flux_adu
    visamp = reduced.visibility_amplitude
    flags = reduced.flags
    phase_deg = reduced.differential_phase_deg
    gd_m = reduced.group_delay_m

    log.info(f"--- Step: PLOTTING {object_label} ---")

    with PdfPages(output) as pdf:
        mean_flux = flux_adu.mean(axis=-1)
        frame_samples = np.arange(0, phase_deg.shape[0], 10)
        wave_samples = np.arange(0, phase_deg.shape[-1], 10)

        def save(fig) -> None:
            pdf.savefig(fig)
            plt.close(fig)

        def plot_flux() -> None:
            fig, axes = _make_figure_grid(
                N_TELESCOPES // 2, 2, xlabel="#Frame", ylabel="Flux [adu]"
            )
            for telescope in range(N_TELESCOPES):
                mean = mean_flux[:, telescope].mean()
                axes[telescope].plot(mean_flux[:, telescope] - mean)
                axes[telescope].set_title(f"Telescope {TELESCOPES[telescope]}")
                axes[telescope].text(
                    0.5,
                    0.8,
                    f"Mean: {mean:.2f}",
                    transform=axes[telescope].transAxes,
                    ha="center",
                    va="center",
                )
            fig.suptitle(f"{object_label}\nwavelength-averaged flux per telescope")
            save(fig)

        def plot_coherent_flux() -> None:
            fig, axes = _make_figure_grid(
                2, N_BASELINES // 2, xlabel="#Frame", ylabel="abs(coherent flux) [adu]"
            )
            mean_coherent = abs(coherent_adu.mean(axis=-1))
            for baseline in range(N_BASELINES):
                mean = mean_coherent[:, baseline].mean()
                axes[baseline].plot(mean_coherent[:, baseline] - mean)
                axes[baseline].set_title(f"Baseline {BASELINES[baseline]}")
                axes[baseline].text(
                    0.5,
                    0.8,
                    f"Mean: {mean:.2f}",
                    transform=axes[baseline].transAxes,
                    ha="center",
                    va="center",
                )
            fig.suptitle(
                f"{object_label}\nwavelength-averaged coherent flux per baseline"
            )
            save(fig)

        def plot_visibility() -> None:
            fig, axes = _make_figure_grid(
                2, N_BASELINES // 2, xlabel="Wavelength Index", ylabel="VISAMP"
            )
            plotted = np.ma.array(visamp, mask=flags)
            colors = plt.cm.viridis(np.linspace(0, 1, len(plotted)))
            for baseline in range(N_BASELINES):
                axes[baseline].set_prop_cycle(color=colors)
                axes[baseline].plot(plotted[:, baseline, :].T, alpha=0.3, lw=0.3)
                axes[baseline].set_title(f"Baseline {BASELINES[baseline]}")
                axes[baseline].set_ylim(0, 1.5)
            fig.suptitle(f"{object_label}\nVISAMP per baseline")
            save(fig)

        def plot_phase() -> None:
            fig, axes = _make_figure_grid(
                2, N_BASELINES // 2, xlabel="#Frame", ylabel="arg(coherent flux) [deg]"
            )
            mean_phase = np.angle(coherent_adu.mean(axis=-1), deg=True)
            mean_phase = np.unwrap(mean_phase, period=360, axis=0)
            for baseline in range(N_BASELINES):
                axes[baseline].plot(mean_phase[:, baseline])
                axes[baseline].set_title(f"Baseline {BASELINES[baseline]}")
            fig.suptitle(f"{object_label}\nwavelength-averaged phase per baseline")
            save(fig)

        def plot_group_delay() -> None:
            fig, axes = _make_figure_grid(
                2,
                N_BASELINES // 2,
                xlabel="#Frame",
                ylabel="Group Delay [m]",
                sharey=False,
            )
            for baseline in range(N_BASELINES):
                axes[baseline].plot(gd_m[:, baseline])
                axes[baseline].set_title(f"Baseline {BASELINES[baseline]}")
            fig.suptitle(f"{object_label}\nGroup Delay per baseline")
            save(fig)

        def plot_spectral_phase() -> None:
            for frame in frame_samples:
                fig, axes = _make_figure_grid(
                    2,
                    N_BASELINES // 2,
                    xlabel="Wavelength Index",
                    ylabel="Differential Phase [deg]",
                )
                for baseline in range(N_BASELINES):
                    axes[baseline].plot(phase_deg[frame, baseline])
                    axes[baseline].set_title(f"Baseline {BASELINES[baseline]}")
                    axes[baseline].set_ylim(-10, 10)
                fig.suptitle(
                    f"{object_label}\ndifferential phase per baseline @ DIT {frame}"
                )
                save(fig)

        def plot_phase_vs_group_delay() -> None:
            for wavelength in wave_samples:
                fig, axes = _make_figure_grid(
                    2,
                    N_BASELINES // 2,
                    xlabel="Group Delay [m]",
                    ylabel="Differential Phase [deg]",
                    sharex=False,
                )
                for baseline in range(N_BASELINES):
                    axes[baseline].scatter(
                        gd_m[:, baseline],
                        phase_deg[:, baseline, wavelength],
                        s=5,
                        alpha=0.7,
                    )
                    axes[baseline].set_title(f"Baseline {BASELINES[baseline]}")
                    axes[baseline].set_ylim(-10, 10)
                fig.suptitle(f"{object_label}\ndifferential phase per baseline")
                save(fig)

        plots = [
            plot_flux,
            plot_coherent_flux,
            plot_visibility,
            plot_phase,
            plot_group_delay,
        ]
        if full:
            plots.extend([plot_spectral_phase, plot_phase_vs_group_delay])
        for plot in plots:
            plot()

    log.info(f"--- Step: PLOTTING {object_label} [DONE] ---")
