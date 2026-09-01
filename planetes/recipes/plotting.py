import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from .. import log
from ..config import PipelineConfig
from ..product import ObjectPreprocProduct, ReducedProduct, read_product
pltkwargs = {
    "font.size": 12,
    "figure.figsize": (12, 8),
    "image.cmap": "inferno",
    "image.origin": "lower",
    "image.interpolation": "nearest",
    "image.aspect": "auto",
    # "savefig.bbox": "tight",
    "figure.autolayout": False,
    "figure.constrained_layout.use": False,
    "figure.dpi": 300,
}
plt.rcParams.update(pltkwargs)


def genfig(m, n, xlabel, ylabel, **kwargs):
    kwargs = {**dict(sharex=True, sharey=True), **kwargs}
    fig, axs = plt.subplots(m, n, **kwargs)
    axs = axs.flatten()
    [ax.set_xlabel(xlabel) for ax in axs]
    [ax.set_ylabel(ylabel) for ax in axs]
    [ax.label_outer() for ax in axs]
    return fig, axs

def summary_plot(
    config: PipelineConfig,
    target_name: str,
    *,
    full: bool | None = None,
) -> None:
    """Write summary diagnostics for one reduced target."""
    if target_name not in config.targets:
        available = ", ".join(config.targets)
        raise ValueError(f"Unknown target {target_name!r}; available targets: {available}")
    if full is None:
        full = config.parameters.plot.full
    object_label = str(config.targets[target_name].data)
    instrument = config.instrument
    log.info(f"--- Step: PLOTTING {target_name} ---")
    plt.rcParams["font.size"] = 14

    reduced = read_product(
        config.output_dir / f"{target_name}_reduced.npz", ReducedProduct
    )
    p2vmred = reduced.p2vmred
    coherent_flux = reduced.coherent_flux
    visamp = reduced.visamp
    flag = reduced.flag
    visphi = reduced.visphi
    gdelay = reduced.gdelay
    bsl_to_reg = reduced.bsl_to_reg

    pdf = PdfPages(config.output_dir / f"{target_name}.pdf")
    mean_flux = p2vmred[:, : instrument.n_telescopes, :].mean(axis=-1)
    frame_samples = np.arange(0, visphi.shape[0], 10)
    wave_samples = np.arange(0, visphi.shape[-1], 10)

    def plot_flux():
        fig, axs = genfig(instrument.n_telescopes // 2, 2,
                          xlabel="#Frame", ylabel="Flux [adu]")
        for tel in range(instrument.n_telescopes):
            _mean = mean_flux[:, tel].mean()
            axs[tel].plot(mean_flux[:, tel] - _mean)
            axs[tel].set_title(f"Telescope {instrument.telescopes[tel]}")
            axs[tel].text(0.5, 0.8, f"Mean: {_mean:.2f}",
                          transform=axs[tel].transAxes, ha="center", va="center")
        fig.suptitle(f"{object_label}\n"\
                      "wavelength-averaged flux per telescope")
        pdf.savefig(fig); plt.close(fig)


    def plot_cohflux():
        fig, axs = genfig(2, instrument.n_baselines // 2,
                          xlabel="#Frame", ylabel="abs(coherent flux) [adu]")
        mean_cohflux = abs(coherent_flux.mean(axis=-1))
        for bsl in range(instrument.n_baselines):
            _mean = mean_cohflux[:, bsl].mean()
            axs[bsl].plot(mean_cohflux[:, bsl] - _mean)
            axs[bsl].set_title(f"Baseline {instrument.baselines[bsl]}")
            axs[bsl].text(0.5, 0.8, f"Mean: {_mean:.2f}",
                          transform=axs[bsl].transAxes, ha="center", va="center")
        fig.suptitle(f"{object_label}\n"\
                      "wavelength-averaged coherent flux per baseline")
        pdf.savefig(fig); plt.close(fig)


    def plot_phase():
        fig, axs = genfig(2, instrument.n_baselines // 2,
                          xlabel="#Frame", ylabel="arg(coherent flux) [deg]")
        mean_phase = np.angle(coherent_flux.mean(axis=-1), deg=True)
        mean_phase = np.unwrap(mean_phase, period=360, axis=0)
        for bsl in range(instrument.n_baselines):
            axs[bsl].plot(mean_phase[:, bsl])
            axs[bsl].set_title(f"Baseline {instrument.baselines[bsl]}")
        fig.suptitle(f"{object_label}\n"\
                      "wavelength-averaged phase per baseline")
        pdf.savefig(fig); plt.close(fig)

    def plot_gdelay():
        fig, axs = genfig(2, instrument.n_baselines // 2,
                          xlabel="#Frame", ylabel="Group Delay [um]", sharey=False)
        for bsl in range(instrument.n_baselines):
            axs[bsl].plot(gdelay[:, bsl])
            axs[bsl].set_title(f"Baseline {instrument.baselines[bsl]}")
        fig.suptitle(f"{object_label}\nGroup Delay per baseline")
        pdf.savefig(fig); plt.close(fig)

    def plot_visphi():
        for dit in frame_samples:
            fig, axs = genfig(2, instrument.n_baselines // 2,
                              xlabel="Wavelength Index", ylabel="Phase [deg]")
            for bsl in range(instrument.n_baselines):
                axs[bsl].plot(visphi[dit, bsl, :])
                axs[bsl].set_title(f"Baseline {instrument.baselines[bsl]}")
                axs[bsl].set_ylim(-10, 10)
            fig.suptitle(f"{object_label}\n"\
                          f"visphi per baseline @ DIT {dit}")
            pdf.savefig(fig); plt.close(fig)

    def plot_gdelay_visphi():
        for iwl in wave_samples:
            fig, axs = genfig(2, instrument.n_baselines // 2,
                              xlabel="Group Delay [um]",
                              ylabel="Phase [deg]",
                              sharex=False)
            for bsl in range(instrument.n_baselines):
                axs[bsl].scatter(gdelay[:, bsl], visphi[:, bsl, iwl], s=5, alpha=0.7)
                axs[bsl].set_title(f"Baseline {instrument.baselines[bsl]}")
                axs[bsl].set_ylim(-10, 10)
            fig.suptitle(f"{object_label}\nvisphi per baseline")
            pdf.savefig(fig); plt.close(fig)


    def plot_visamp():
        fig, axs = genfig(2, instrument.n_baselines // 2,
                          xlabel="Wavelength Index", ylabel="VISAMP")
        plotted_visamp = np.ma.array(visamp, mask=flag)
        n_dit = plotted_visamp.shape[0]
        colors = plt.cm.viridis(np.linspace(0, 1, n_dit))
        for bsl in range(instrument.n_baselines):
            axs[bsl].set_prop_cycle(color=colors)
            axs[bsl].plot(plotted_visamp[:, bsl, :].T, alpha=0.3, lw=0.3)
            axs[bsl].set_title(f"Baseline {instrument.baselines[bsl]}")
            axs[bsl].set_ylim(0, 1.5)
        fig.suptitle(f"{object_label}\nVISAMP per baseline")
        pdf.savefig(fig); plt.close(fig)


    def plot_ABCD():
        preproc = read_product(
            config.output_dir / f"{target_name}_preproc.npz",
            ObjectPreprocProduct,
        )
        spec = preproc.spec.copy()
        spec_flat = preproc.spec_flat
        spec /= spec_flat

        for bsl in range(instrument.n_baselines):
            fig, axs = genfig(2, 2,
                              xlabel="Wavelength Index", ylabel="Normalized Flux")
            regs = bsl_to_reg[bsl]
            colors = plt.cm.viridis(np.linspace(0, 1, len(frame_samples)))
            for i in range(4):
                axs[i].set_prop_cycle(color=colors)
                axs[i].plot(spec[regs[i], frame_samples, :].T, alpha=0.5, lw=0.3)
                axs[i].set_title(f"OUTPUT {regs[i]}-{['A','C','B','D'][i]}")
            fig.suptitle(f"{object_label}\n"\
                         f"Baseline {instrument.baselines[bsl]} Extracted Spectra")
            pdf.savefig(fig); plt.close(fig)

    plots = [plot_flux, plot_cohflux, plot_visamp, plot_phase, plot_gdelay]
    if full:
        plots.extend([plot_visphi, plot_gdelay_visphi, plot_ABCD])
    for plot in plots:
        plot()
    pdf.close()

    return


def colored_text(text: str, color: str="white", bold: bool=False) -> str:
    # Dictionary mapping color names to ANSI codes
    colors = {
        "red": "\033[91m", "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "magenta": "\033[95m",
        "cyan": "\033[96m",
        "white": "\033[97m",
        "reset": "\033[0m"
    }

    code = colors.get(color.lower(), colors["white"])
    if bold:
        return f"\033[1m{code}{text}{colors['reset']}\033[0m"
    return f"{code}{text}{colors['reset']}"
