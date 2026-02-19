import matplotlib.pyplot as plt
import numpy as np

from . import PipelineContext, arg, command, log

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

@command("plot", "Summary plots for reduced data.")
@arg("--object", type=str, default=None, help="target to reduce (object name)")
def summary_plot(ctx: PipelineContext, **kwargs):
    obj = kwargs.get("object", None)
    if not obj: return
    obj_list = ctx.conf["object"] if obj == "all" else [obj]
    for obj in obj_list:
        summary_plot_single(ctx, object=obj)

def summary_plot_single(ctx: PipelineContext, **kwargs):
    obj = kwargs.get("object", None)
    log.info(f"--- Step: PLOTTING {obj} ---")
    plt.rcParams["font.size"] = 14

    with ctx.load_product(("reduced", obj)) as d:
        p2vmred = d["p2vmred"]
        visdata = d["visdata"]
        visphi = d["visphi"]
        gdelay = d["gdelay"]
        bsl_to_reg = d["bsl_to_reg"][()]
        bsl_to_tel = d["bsl_to_tel"][()]

    pdf = ctx.plot_ctx(f"{obj}.pdf")
    mean_flux = p2vmred[:, :ctx.n_tel, :].mean(axis=-1)

    def plot_flux():
        fig, axs = genfig(ctx.n_tel//2, 2,
                          xlabel="#Frame", ylabel="Flux [adu]")
        for tel in range(ctx.n_tel):
            _mean = mean_flux[:, tel].mean()
            axs[tel].plot(mean_flux[:, tel] - _mean)
            axs[tel].set_title(f"Telescope {ctx.telescopes[tel]}")
            axs[tel].text(0.5, 0.8, f"Mean: {_mean:.2f}",
                          transform=axs[tel].transAxes, ha="center", va="center")
        fig.suptitle(f"{ctx.conf['object'][obj]}\n"\
                      "wavelength-averaged flux per telescope")
        pdf.savefig(fig); plt.close(fig)


    def plot_cohflux():
        fig, axs = genfig(2, ctx.n_bsl//2,
                          xlabel="#Frame", ylabel="abs(VISDATA) [adu]")
        mean_cohflux = abs(visdata.mean(axis=-1))
        for bsl in range(ctx.n_bsl):
            _mean = mean_cohflux[:, bsl].mean()
            axs[bsl].plot(mean_cohflux[:, bsl] - _mean)
            axs[bsl].set_title(f"Baseline {ctx.baselines[bsl]}")
            axs[bsl].text(0.5, 0.8, f"Mean: {_mean:.2f}",
                          transform=axs[bsl].transAxes, ha="center", va="center")
        fig.suptitle(f"{ctx.conf['object'][obj]}\n"\
                      "wavelength-averaged coherent flux per baseline")
        pdf.savefig(fig); plt.close(fig)


    def plot_phase():
        fig, axs = genfig(2, ctx.n_bsl//2,
                          xlabel="#Frame", ylabel="arg(VISDATA) [deg]")
        mean_phase = np.angle(visdata.mean(axis=-1), deg=True)
        mean_phase = np.unwrap(mean_phase, period=360, axis=0)
        for bsl in range(ctx.n_bsl):
            axs[bsl].plot(mean_phase[:, bsl])
            axs[bsl].set_title(f"Baseline {ctx.baselines[bsl]}")
        fig.suptitle(f"{ctx.conf['object'][obj]}\n"\
                      "wavelength-averaged phase per baseline")
        pdf.savefig(fig); plt.close(fig)

    def plot_gdelay():
        fig, axs = genfig(2, ctx.n_bsl//2,
                          xlabel="#Frame", ylabel="Group Delay [um]", sharey=False)
        for bsl in range(ctx.n_bsl):
            axs[bsl].plot(gdelay[:, bsl])
            axs[bsl].set_title(f"Baseline {ctx.baselines[bsl]}")
        fig.suptitle(f"{ctx.conf['object'][obj]}\nGroup Delay per baseline")
        pdf.savefig(fig); plt.close(fig)

    def plot_visphi():
        for dit in np.arange(0, visphi.shape[0]-1, 10):
            fig, axs = genfig(2, ctx.n_bsl//2,
                              xlabel="Wavelength Index", ylabel="Phase [deg]")
            for bsl in range(ctx.n_bsl):
                axs[bsl].plot(visphi[dit, bsl, :])
                axs[bsl].set_title(f"Baseline {ctx.baselines[bsl]}")
                axs[bsl].set_ylim(-10, 10)
            fig.suptitle(f"{ctx.conf['object'][obj]}\n"\
                          f"visphi per baseline @ DIT {dit}")
            pdf.savefig(fig); plt.close(fig)

    def plot_gdelay_visphi():
        for iwl in np.arange(0, visphi.shape[-1]-1, 100):
            fig, axs = genfig(2, ctx.n_bsl//2,
                              xlabel="Group Delay [um]",
                              ylabel="Phase [deg]",
                              sharex=False)
            for bsl in range(ctx.n_bsl):
                axs[bsl].scatter(gdelay[:, bsl], visphi[:, bsl, iwl], s=5, alpha=0.7)
                axs[bsl].set_title(f"Baseline {ctx.baselines[bsl]}")
                axs[bsl].set_ylim(-10, 10)
            fig.suptitle(f"{ctx.conf['object'][obj]}\nvisphi per baseline")
            pdf.savefig(fig); plt.close(fig)


    def plot_visamp():
        fig, axs = genfig(2, ctx.n_bsl//2,
                          xlabel="Wavelength Index", ylabel="VISAMP")
        visamp = np.abs(visdata)
        n_dit = visamp.shape[0]
        colors = plt.cm.viridis(np.linspace(0, 1, n_dit))
        for bsl in range(ctx.n_bsl):
            tel1, tel2 = bsl_to_tel[bsl]
            visamp[:, bsl, :] /= (p2vmred[:, tel1, :] * p2vmred[:, tel2, :]) ** 0.5
            axs[bsl].set_prop_cycle(color=colors)
            axs[bsl].plot(visamp[:, bsl, :].T, alpha=0.3, lw=0.3)
            axs[bsl].set_title(f"Baseline {ctx.baselines[bsl]}")
            axs[bsl].set_ylim(0, 1.5)
        fig.suptitle(f"{ctx.conf['object'][obj]}\nVISAMP per baseline")
        pdf.savefig(fig); plt.close(fig)


    def plot_ABCD():
        with ctx.load_product(("preproc", obj)) as d:
            spec = d["spec"]
            spec_flat = d["spec_flat"]
        spec /= spec_flat

        for bsl in range(ctx.n_bsl):
            fig, axs = genfig(2, 2,
                              xlabel="Wavelength Index", ylabel="Normalized Flux")
            regs = bsl_to_reg[bsl]
            n_dit = spec.shape[1]
            colors = plt.cm.viridis(np.linspace(0, 1, n_dit))
            for i in range(4):
                axs[i].set_prop_cycle(color=colors)
                axs[i].plot(spec[regs[i], :, :].T, alpha=0.5, lw=0.3)
                axs[i].set_title(f"OUTPUT {regs[i]}-{['A','C','B','D'][i]}")
            fig.suptitle(f"{ctx.conf['object'][obj]}\n"\
                         f"Baseline {ctx.baselines[bsl]} Extracted Spectra")
            pdf.savefig(fig); plt.close(fig)

    [fn() for fn in [plot_flux, plot_cohflux, plot_visamp, plot_phase, plot_gdelay,
                     plot_visphi, plot_gdelay_visphi, plot_ABCD]]
    pdf.close()

    return

def bold(text):
    return f"\033[1m{text}\033[0m"

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
