"""Diagnostic PDF reports for calibration products and per-frame science observables."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from planetes_pipe import log
from planetes_pipe.instrument import layout
from planetes_pipe.steps.dark import Bad, Dark
from planetes_pipe.steps.flat import Flat
from planetes_pipe.steps.p2vm import P2VM
from planetes_pipe.steps.science import P2VMRed
from planetes_pipe.steps.wave import Wave

# Line colours of the A, C, B, D outputs, as matplotlib's default cycle draws them.
ABCD_COLORS = ('C0', 'C1', 'C2', 'C3')


def save_calibration_plots(
    dark: Dark, bad: Bad, flat: Flat, wave: Wave, p2vm: P2VM, output: Path,
) -> None:
    """Write product maps, fitted traces, wavelength and instrumental P2VM diagnostics.

    P2VM pages: transmission, normalized coherence and phase per baseline and output.

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
        fig, axes = plt.subplots(1, 3)
        for ax, image, title, unit in zip(
            axes, (dark.mean_adu, flat.flat_map_adu, bad.bad_pixel_flags),
            ('Dark', 'Combined flat', 'Bad-pixel flags'), ('ADU', 'ADU', 'Bit mask'),
        ):
            im = ax.imshow(image, origin='lower', aspect='auto', cmap='inferno')
            fig.colorbar(im, ax=ax, label=unit, shrink=.65)
            ax.set(title=title, xlabel='Detector x [pixel]', ylabel='Detector y [pixel]')
        _save(pdf, fig, 'Calibration detector maps', top=0.95)

        fig, axes = plt.subplots(6, 4, figsize=(11.7, 11.7))
        for b, pair in enumerate(layout.baselines):
            for ax, o, label in zip(axes[b], p2vm.baseline_outputs[b], layout.abcd_order):
                ax.plot(x[band[o]], flat.trace_y_pixels[o, band[o]], lw=1)
                ax.set_title(f'S{pair[0]}–S{pair[1]} {label} / output {o}')
                ax.ticklabel_format(axis='y', style='plain', useOffset=False)
        fig.supxlabel('Detector x [pixel] — P2VM band only')
        fig.supylabel('Trace centre y [pixel]')
        _save(pdf, fig, 'Retained extraction traces — independent axis scales', top=0.95)

        fig, axes = plt.subplots(2, 1)
        for o in range(len(band)):
            axes[0].plot(x[band[o]], wave.wavelength_map_m[o, band[o]] * 1e9, lw=.6)
        axes[0].set(xlabel='Detector x [pixel]', ylabel='Wavelength [nm]')
        if wave.fit_residual_m is not None:
            axes[1].scatter(wave.fit_x_pixels, wave.fit_residual_m * 1e9, s=5)
            axes[1].axhline(0, color='gray', lw=.7)
        axes[1].set(xlabel='Accepted line x [pixel]', ylabel='Wavelength residual [nm]')
        _save(pdf, fig, 'Wavelength solution — all accepted line residuals', top=0.95)

        fig, axes = plt.subplots(2, 3)
        for b, (ax, pair) in enumerate(zip(axes.flat, layout.baselines)):
            t1, t2 = layout.baseline_pairs[b]
            for o, label, color in zip(p2vm.baseline_outputs[b], layout.abcd_order, ABCD_COLORS):
                ax.plot(wavelength, p2vm.transmission[o, t1], color=color, label=f'{label} T{pair[0]}')
                ax.plot(wavelength, p2vm.transmission[o, t2], color=color, ls='--',
                        label=f'{label} T{pair[1]}')
            ax.set(title=f'S{pair[0]}–S{pair[1]}: T{pair[0]} solid, T{pair[1]} dashed',
                   xlabel='Wavelength [nm]', ylabel='Transmission')
        axes[0, 0].legend(fontsize=7, ncol=2)
        _save(pdf, fig, 'P2VM transmission — share of each telescope in each output; for every '
                        'telescope and wavelength the mean over the 24 outputs is 1', top=0.95)

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
            _save(pdf, fig, f'P2VM {quantity} — instrumental response, not target visibility', top=0.95)

        if p2vm.fit_rms_adu is not None:
            fig, axes = plt.subplots(2, 3)
            for b, (ax, pair) in enumerate(zip(axes.flat, layout.baselines)):
                for o, label in zip(p2vm.baseline_outputs[b], layout.abcd_order):
                    ax.plot(wavelength, p2vm.fit_rms_adu[o, b], label=label)
                ax.set(title=f'S{pair[0]}–S{pair[1]}', xlabel='Wavelength [nm]', ylabel='Fit RMS [ADU]')
            axes[0, 0].legend()
            _save(pdf, fig, 'Fringe-fit residual RMS', top=0.95)

        if p2vm.opd_m is not None:
            fig, axes = plt.subplots(2, 3)
            for b, (ax, pair) in enumerate(zip(axes.flat, layout.baselines)):
                ax.plot(p2vm.opd_m[b] * 1e6, label='OPD')
                if p2vm.group_delay_m is not None:
                    ax.plot(p2vm.group_delay_m[b] * 1e6, label='Group delay', alpha=.7)
                ax.set(title=f'S{pair[0]}–S{pair[1]}', xlabel='Frame', ylabel='Delay [µm]')
            axes[0, 0].legend()
            _save(pdf, fig, 'Recovered scan delays', top=0.95)


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

    Pages: flux per telescope, coherent flux, VISAMP (flagged samples masked),
    phase and group delay per baseline. full=True adds differential-phase pages
    for every tenth frame and wavelength channel. Needs a complete P2VMRed (with
    GD and differential phase); an existing output raises FileExistsError.
    """
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing diagnostic: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    log.info(f"--- Step: PLOTTING {object_label} ---")
    n_frames, _, n_wave = reduced.differential_phase_deg.shape
    with PdfPages(output) as pdf:
        _save(pdf, _flux_page(reduced, object_label))
        _save(pdf, _coherent_flux_page(reduced, object_label))
        _save(pdf, _visibility_page(reduced, object_label))
        _save(pdf, _phase_page(reduced, object_label))
        _save(pdf, _group_delay_page(reduced, object_label))
        if full:
            for frame in range(0, n_frames, 10):
                _save(pdf, _spectral_phase_page(reduced, object_label, frame))
            for channel in range(0, n_wave, 10):
                _save(pdf, _phase_vs_group_delay_page(reduced, object_label, channel))
    log.info(f"--- Step: PLOTTING {object_label} [DONE] ---")


def _save(pdf, fig, title: str | None = None, top: float = 0.93) -> None:
    """Add one figure as a PDF page and close it."""
    if title is not None:
        fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, top))
    pdf.savefig(fig)
    plt.close(fig)


def _baseline_grid(xlabel: str, ylabel: str, **kwargs):
    return _make_figure_grid(2, layout.n_baselines // 2, xlabel=xlabel, ylabel=ylabel, **kwargs)


def _annotate_mean(axis, mean: float) -> None:
    axis.text(0.5, 0.8, f"Mean: {mean:.2f}", transform=axis.transAxes, ha="center", va="center")


def _flux_page(reduced: P2VMRed, label: str):
    fig, axes = _make_figure_grid(layout.n_telescopes // 2, 2, xlabel="#Frame", ylabel="Flux [adu]")
    mean_flux = reduced.photometric_flux_adu.mean(axis=-1)
    for telescope, axis in enumerate(axes[: layout.n_telescopes]):
        mean = mean_flux[:, telescope].mean()
        axis.plot(mean_flux[:, telescope] - mean)
        axis.set_title(f"Telescope {layout.telescopes[telescope]}")
        _annotate_mean(axis, mean)
    fig.suptitle(f"{label}\nwavelength-averaged flux per telescope")
    return fig


def _coherent_flux_page(reduced: P2VMRed, label: str):
    fig, axes = _baseline_grid("#Frame", "abs(coherent flux) [adu]")
    mean_coherent = abs(reduced.coherent_flux_adu.mean(axis=-1))
    for baseline, axis in enumerate(axes):
        mean = mean_coherent[:, baseline].mean()
        axis.plot(mean_coherent[:, baseline] - mean)
        axis.set_title(f"Baseline {layout.baselines[baseline]}")
        _annotate_mean(axis, mean)
    fig.suptitle(f"{label}\nwavelength-averaged coherent flux per baseline")
    return fig


def _visibility_page(reduced: P2VMRed, label: str):
    fig, axes = _baseline_grid("Wavelength Index", "VISAMP")
    visamp = np.ma.array(reduced.visibility_amplitude, mask=reduced.flags)
    colors = plt.cm.viridis(np.linspace(0, 1, len(visamp)))
    for baseline, axis in enumerate(axes):
        axis.set_prop_cycle(color=colors)
        axis.plot(visamp[:, baseline, :].T, alpha=0.3, lw=0.3)
        axis.set_title(f"Baseline {layout.baselines[baseline]}")
        axis.set_ylim(0, 1.5)
    fig.suptitle(f"{label}\nVISAMP per baseline")
    return fig


def _phase_page(reduced: P2VMRed, label: str):
    fig, axes = _baseline_grid("#Frame", "arg(coherent flux) [deg]")
    mean_phase = np.angle(reduced.coherent_flux_adu.mean(axis=-1), deg=True)
    mean_phase = np.unwrap(mean_phase, period=360, axis=0)
    for baseline, axis in enumerate(axes):
        axis.plot(mean_phase[:, baseline])
        axis.set_title(f"Baseline {layout.baselines[baseline]}")
    fig.suptitle(f"{label}\nwavelength-averaged phase per baseline")
    return fig


def _group_delay_page(reduced: P2VMRed, label: str):
    fig, axes = _baseline_grid("#Frame", "Group Delay [m]", sharey=False)
    for baseline, axis in enumerate(axes):
        axis.plot(reduced.group_delay_m[:, baseline])
        axis.set_title(f"Baseline {layout.baselines[baseline]}")
    fig.suptitle(f"{label}\nGroup Delay per baseline")
    return fig


def _spectral_phase_page(reduced: P2VMRed, label: str, frame: int):
    fig, axes = _baseline_grid("Wavelength Index", "Differential Phase [deg]")
    for baseline, axis in enumerate(axes):
        axis.plot(reduced.differential_phase_deg[frame, baseline])
        axis.set_title(f"Baseline {layout.baselines[baseline]}")
        axis.set_ylim(-10, 10)
    fig.suptitle(f"{label}\ndifferential phase per baseline @ DIT {frame}")
    return fig


def _phase_vs_group_delay_page(reduced: P2VMRed, label: str, channel: int):
    fig, axes = _baseline_grid("Group Delay [m]", "Differential Phase [deg]", sharex=False)
    for baseline, axis in enumerate(axes):
        axis.scatter(reduced.group_delay_m[:, baseline],
                     reduced.differential_phase_deg[:, baseline, channel], s=5, alpha=0.7)
        axis.set_title(f"Baseline {layout.baselines[baseline]}")
        axis.set_ylim(-10, 10)
    fig.suptitle(f"{label}\ndifferential phase per baseline")
    return fig
