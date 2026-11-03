"""Calibration product diagnostics; raw lamp and scan cubes are not reread."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from planetes_pipe.instrument import layout
from planetes_pipe.dark import Bad, Dark
from planetes_pipe.flat import Flat
from planetes_pipe.p2vm import P2VM
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
