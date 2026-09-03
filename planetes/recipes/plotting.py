"""Diagnostic plots for reduced targets."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from .. import log
from ..config import PipelineConfig
from ..instrument import BASELINES, N_BASELINES, N_TELESCOPES, TELESCOPES
from ..product import ReducedProduct, read_product

plt.rcParams.update(
    {
        "font.size": 12,
        "figure.figsize": (12, 8),
        "image.cmap": "inferno",
        "image.origin": "lower",
        "image.interpolation": "nearest",
        "image.aspect": "auto",
        "figure.autolayout": False,
        "figure.constrained_layout.use": False,
        "figure.dpi": 300,
    }
)


def genfig(rows: int, columns: int, xlabel: str, ylabel: str, **kwargs):
    """Create a labelled grid of diagnostic axes."""
    options = {"sharex": True, "sharey": True, **kwargs}
    fig, axes = plt.subplots(rows, columns, **options)
    axes = np.asarray(axes).ravel()
    for axis in axes:
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.label_outer()
    return fig, axes


def summary_plot(
    config: PipelineConfig,
    target_name: str,
    *,
    full: bool = False,
) -> None:
    """Write summary diagnostics for one reduced target."""
    target = config.target(target_name)
    reduced = read_product(
        config.output_dir / f"{target_name}_reduced.npz", ReducedProduct
    )
    object_label = str(target.data)
    fluxdata = reduced.fluxdata
    coherent_flux = reduced.coherent_flux
    visamp = reduced.visamp
    flag = reduced.flag
    visphi = reduced.visphi
    gdelay = reduced.gdelay

    log.info(f"--- Step: PLOTTING {target_name} ---")
    plt.rcParams["font.size"] = 14

    with PdfPages(config.output_dir / f"{target_name}.pdf") as pdf:
        mean_flux = fluxdata.mean(axis=-1)
        frame_samples = np.arange(0, visphi.shape[0], 10)
        wave_samples = np.arange(0, visphi.shape[-1], 10)

        def save(fig) -> None:
            pdf.savefig(fig)
            plt.close(fig)

        def plot_flux() -> None:
            fig, axes = genfig(
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
            fig, axes = genfig(
                2,
                N_BASELINES // 2,
                xlabel="#Frame",
                ylabel="abs(coherent flux) [adu]",
            )
            mean_coherent_flux = abs(coherent_flux.mean(axis=-1))
            for baseline in range(N_BASELINES):
                mean = mean_coherent_flux[:, baseline].mean()
                axes[baseline].plot(mean_coherent_flux[:, baseline] - mean)
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
            fig, axes = genfig(
                2, N_BASELINES // 2, xlabel="Wavelength Index", ylabel="VISAMP"
            )
            plotted = np.ma.array(visamp, mask=flag)
            colors = plt.cm.viridis(np.linspace(0, 1, len(plotted)))
            for baseline in range(N_BASELINES):
                axes[baseline].set_prop_cycle(color=colors)
                axes[baseline].plot(plotted[:, baseline, :].T, alpha=0.3, lw=0.3)
                axes[baseline].set_title(f"Baseline {BASELINES[baseline]}")
                axes[baseline].set_ylim(0, 1.5)
            fig.suptitle(f"{object_label}\nVISAMP per baseline")
            save(fig)

        def plot_phase() -> None:
            fig, axes = genfig(
                2,
                N_BASELINES // 2,
                xlabel="#Frame",
                ylabel="arg(coherent flux) [deg]",
            )
            mean_phase = np.angle(coherent_flux.mean(axis=-1), deg=True)
            mean_phase = np.unwrap(mean_phase, period=360, axis=0)
            for baseline in range(N_BASELINES):
                axes[baseline].plot(mean_phase[:, baseline])
                axes[baseline].set_title(f"Baseline {BASELINES[baseline]}")
            fig.suptitle(f"{object_label}\nwavelength-averaged phase per baseline")
            save(fig)

        def plot_group_delay() -> None:
            fig, axes = genfig(
                2,
                N_BASELINES // 2,
                xlabel="#Frame",
                ylabel="Group Delay [um]",
                sharey=False,
            )
            for baseline in range(N_BASELINES):
                axes[baseline].plot(gdelay[:, baseline])
                axes[baseline].set_title(f"Baseline {BASELINES[baseline]}")
            fig.suptitle(f"{object_label}\nGroup Delay per baseline")
            save(fig)

        def plot_spectral_phase() -> None:
            for frame in frame_samples:
                fig, axes = genfig(
                    2,
                    N_BASELINES // 2,
                    xlabel="Wavelength Index",
                    ylabel="Phase [deg]",
                )
                for baseline in range(N_BASELINES):
                    axes[baseline].plot(visphi[frame, baseline])
                    axes[baseline].set_title(f"Baseline {BASELINES[baseline]}")
                    axes[baseline].set_ylim(-10, 10)
                fig.suptitle(f"{object_label}\nvisphi per baseline @ DIT {frame}")
                save(fig)

        def plot_phase_vs_group_delay() -> None:
            for wavelength in wave_samples:
                fig, axes = genfig(
                    2,
                    N_BASELINES // 2,
                    xlabel="Group Delay [um]",
                    ylabel="Phase [deg]",
                    sharex=False,
                )
                for baseline in range(N_BASELINES):
                    axes[baseline].scatter(
                        gdelay[:, baseline],
                        visphi[:, baseline, wavelength],
                        s=5,
                        alpha=0.7,
                    )
                    axes[baseline].set_title(f"Baseline {BASELINES[baseline]}")
                    axes[baseline].set_ylim(-10, 10)
                fig.suptitle(f"{object_label}\nvisphi per baseline")
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

    log.info(f"--- Step: PLOTTING {target_name} [DONE] ---")
