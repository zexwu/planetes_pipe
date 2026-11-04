"""Diagnostic PDF reports for calibration products and per-frame science observables."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from planetes_pipe import log
from planetes_pipe.dark import Bad, Dark
from planetes_pipe.flat import Flat
from planetes_pipe.instrument import layout
from planetes_pipe.p2vm import P2VM
from planetes_pipe.science import P2VMRed
from planetes_pipe.wave import Wave


def save_calibration_plots(
    dark: Dark, bad: Bad, flat: Flat, wave: Wave, p2vm: P2VM, output: Path,
) -> None:
    """Write product maps, fitted traces, wavelength and instrumental P2VM diagnostics.

    Trace plots cover each output's P2VM wavelength interval only. Optional
    residual and OPD pages are included when saved in the P2VM product.
    """
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    wavelength = p2vm.wavelength_m * 1e9
    band = (wave.wavelength_map_m >= p2vm.wavelength_m.min()) & (
        wave.wavelength_map_m <= p2vm.wavelength_m.max())
    x = np.arange(flat.trace_y_pixels.shape[1])
    with plt.rc_context({'font.size': 9, 'figure.figsize': (11.7, 8.3)}), PdfPages(output) as pdf:
        def save(fig, title):
            fig.suptitle(title)
            fig.tight_layout(rect=(0, 0, 1, .95))
            pdf.savefig(fig)
            plt.close(fig)

        fig, axes = plt.subplots(1, 3)
        for ax, image, title, unit in zip(
            axes, (dark.mean_adu, flat.flat_map_adu, bad.bad_pixel_flags),
            ('Dark', 'Combined flat', 'Bad-pixel flags'), ('ADU', 'ADU', 'Bit mask'),
        ):
            im = ax.imshow(image, origin='lower', aspect='auto', cmap='inferno')
            fig.colorbar(im, ax=ax, label=unit, shrink=.65)
            ax.set(title=title, xlabel='Detector x [pixel]', ylabel='Detector y [pixel]')
        save(fig, 'Calibration detector maps')

        fig, axes = plt.subplots(6, 4, figsize=(11.7, 11.7))
        for b, pair in enumerate(layout.baselines):
            for ax, o, label in zip(axes[b], p2vm.baseline_outputs[b], layout.abcd_order):
                ax.plot(x[band[o]], flat.trace_y_pixels[o, band[o]], lw=1)
                ax.set_title(f'S{pair[0]}–S{pair[1]} {label} / output {o}')
                ax.ticklabel_format(axis='y', style='plain', useOffset=False)
        fig.supxlabel('Detector x [pixel] — P2VM band only')
        fig.supylabel('Trace centre y [pixel]')
        save(fig, 'Retained extraction traces — independent axis scales')

        fig, axes = plt.subplots(2, 1)
        for o in range(len(band)):
            axes[0].plot(x[band[o]], wave.wavelength_map_m[o, band[o]] * 1e9, lw=.6)
        axes[0].set(xlabel='Detector x [pixel]', ylabel='Wavelength [nm]')
        if wave.fit_residual_m is not None:
            axes[1].scatter(wave.fit_x_pixels, wave.fit_residual_m * 1e9, s=5)
            axes[1].axhline(0, color='gray', lw=.7)
        axes[1].set(xlabel='Accepted line x [pixel]', ylabel='Wavelength residual [nm]')
        save(fig, 'Wavelength solution — all accepted line residuals')

        for quantity in ('coherence', 'phase'):
            fig, axes = plt.subplots(2, 3)
            for b, (ax, pair) in enumerate(zip(axes.flat, layout.baselines)):
                outputs = p2vm.baseline_outputs[b]
                if quantity == 'phase':
                    values = np.rad2deg(p2vm.coherence_phase_rad[outputs, b]) % 360
                    unit = 'Phase relative to A [deg]'
                else:
                    t1, t2 = layout.baseline_pairs[b]
                    norm = 2 * np.sqrt(p2vm.transmission[outputs, t1] * p2vm.transmission[outputs, t2])
                    values = np.divide(p2vm.coherence_amplitude[outputs, b], norm,
                                       out=np.full_like(norm, np.nan), where=norm > 0)
                    unit = 'Normalized coherence'
                for value, label in zip(values, layout.abcd_order):
                    ax.plot(wavelength, value, label=label)
                ax.set(title=f'S{pair[0]}–S{pair[1]}', xlabel='Wavelength [nm]', ylabel=unit)
            axes[0, 0].legend()
            save(fig, f'P2VM {quantity} — instrumental response, not target visibility')

        if p2vm.fit_rms_adu is not None:
            fig, axes = plt.subplots(2, 3)
            for b, (ax, pair) in enumerate(zip(axes.flat, layout.baselines)):
                for o, label in zip(p2vm.baseline_outputs[b], layout.abcd_order):
                    ax.plot(wavelength, p2vm.fit_rms_adu[o, b], label=label)
                ax.set(title=f'S{pair[0]}–S{pair[1]}', xlabel='Wavelength [nm]', ylabel='Fit RMS [ADU]')
            axes[0, 0].legend()
            save(fig, 'Fringe-fit residual RMS')

        if p2vm.opd_m is not None:
            fig, axes = plt.subplots(2, 3)
            for b, (ax, pair) in enumerate(zip(axes.flat, layout.baselines)):
                ax.plot(p2vm.opd_m[b] * 1e6, label='OPD')
                if p2vm.group_delay_m is not None:
                    ax.plot(p2vm.group_delay_m[b] * 1e6, label='Group delay', alpha=.7)
                ax.set(title=f'S{pair[0]}–S{pair[1]}', xlabel='Frame', ylabel='Delay [µm]')
            axes[0, 0].legend()
            save(fig, 'Recovered scan delays')


def _make_figure_grid(rows: int, columns: int, xlabel: str, ylabel: str, **kwargs):
    """Create a labelled grid of diagnostic axes."""
    options = {"sharex": True, "sharey": True, "figsize": (11.7, 8.3), **kwargs}
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
    """Write a multipage PDF of per-frame instrumental reduction diagnostics.

    Parameters
    ----------
    reduced : P2VMRed
        Complete product including GD [m] and differential phase [deg]. A live
        preview with those fields set to None is insufficient for these plots.
    output : Path
        New PDF destination; an existing file raises FileExistsError.
    object_label : str, optional
        Display label used in page titles; no physical target inference is made.
    full : bool, optional
        False writes flux, coherent flux, VISAMP, phase and GD summary pages.
        True adds differential-phase pages sampled every tenth frame/channel.

    Notes
    -----
    The VISAMP page masks flagged samples. Other diagnostic pages show their
    stored arrays without applying a new rejection rule. Wavelength-dependent
    plots currently use channel index on the x axis. Returns None.
    """
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
            fig.tight_layout(rect=(0, 0, 1, .93))
            pdf.savefig(fig)
            plt.close(fig)

        def plot_flux() -> None:
            fig, axes = _make_figure_grid(
                layout.n_telescopes // 2, 2, xlabel="#Frame", ylabel="Flux [adu]"
            )
            for telescope in range(layout.n_telescopes):
                mean = mean_flux[:, telescope].mean()
                axes[telescope].plot(mean_flux[:, telescope] - mean)
                axes[telescope].set_title(f"Telescope {layout.telescopes[telescope]}")
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
                2, layout.n_baselines // 2, xlabel="#Frame", ylabel="abs(coherent flux) [adu]"
            )
            mean_coherent = abs(coherent_adu.mean(axis=-1))
            for baseline in range(layout.n_baselines):
                mean = mean_coherent[:, baseline].mean()
                axes[baseline].plot(mean_coherent[:, baseline] - mean)
                axes[baseline].set_title(f"Baseline {layout.baselines[baseline]}")
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
                2, layout.n_baselines // 2, xlabel="Wavelength Index", ylabel="VISAMP"
            )
            plotted = np.ma.array(visamp, mask=flags)
            colors = plt.cm.viridis(np.linspace(0, 1, len(plotted)))
            for baseline in range(layout.n_baselines):
                axes[baseline].set_prop_cycle(color=colors)
                axes[baseline].plot(plotted[:, baseline, :].T, alpha=0.3, lw=0.3)
                axes[baseline].set_title(f"Baseline {layout.baselines[baseline]}")
                axes[baseline].set_ylim(0, 1.5)
            fig.suptitle(f"{object_label}\nVISAMP per baseline")
            save(fig)

        def plot_phase() -> None:
            fig, axes = _make_figure_grid(
                2, layout.n_baselines // 2, xlabel="#Frame", ylabel="arg(coherent flux) [deg]"
            )
            mean_phase = np.angle(coherent_adu.mean(axis=-1), deg=True)
            mean_phase = np.unwrap(mean_phase, period=360, axis=0)
            for baseline in range(layout.n_baselines):
                axes[baseline].plot(mean_phase[:, baseline])
                axes[baseline].set_title(f"Baseline {layout.baselines[baseline]}")
            fig.suptitle(f"{object_label}\nwavelength-averaged phase per baseline")
            save(fig)

        def plot_group_delay() -> None:
            fig, axes = _make_figure_grid(
                2,
                layout.n_baselines // 2,
                xlabel="#Frame",
                ylabel="Group Delay [m]",
                sharey=False,
            )
            for baseline in range(layout.n_baselines):
                axes[baseline].plot(gd_m[:, baseline])
                axes[baseline].set_title(f"Baseline {layout.baselines[baseline]}")
            fig.suptitle(f"{object_label}\nGroup Delay per baseline")
            save(fig)

        def plot_spectral_phase() -> None:
            for frame in frame_samples:
                fig, axes = _make_figure_grid(
                    2,
                    layout.n_baselines // 2,
                    xlabel="Wavelength Index",
                    ylabel="Differential Phase [deg]",
                )
                for baseline in range(layout.n_baselines):
                    axes[baseline].plot(phase_deg[frame, baseline])
                    axes[baseline].set_title(f"Baseline {layout.baselines[baseline]}")
                    axes[baseline].set_ylim(-10, 10)
                fig.suptitle(
                    f"{object_label}\ndifferential phase per baseline @ DIT {frame}"
                )
                save(fig)

        def plot_phase_vs_group_delay() -> None:
            for wavelength in wave_samples:
                fig, axes = _make_figure_grid(
                    2,
                    layout.n_baselines // 2,
                    xlabel="Group Delay [m]",
                    ylabel="Differential Phase [deg]",
                    sharex=False,
                )
                for baseline in range(layout.n_baselines):
                    axes[baseline].scatter(
                        gd_m[:, baseline],
                        phase_deg[:, baseline, wavelength],
                        s=5,
                        alpha=0.7,
                    )
                    axes[baseline].set_title(f"Baseline {layout.baselines[baseline]}")
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
