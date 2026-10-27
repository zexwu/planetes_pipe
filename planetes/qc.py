"""Applicable GRAVITY QC definitions; unavailable physics never becomes a zero.

References (GRAVITY 1.9.6): gravi_calib.c:183,1470-1519,2158;
gravi_wave.c:2027-2045,2266; gravi_p2vm.c:1180-1303.
"""

import numpy as np

from .calibration.p2vm import build_v2pm_matrix
from .instrument import BASELINES, TELESCOPES
from .products import (
    BadPixelMap,
    DarkCalibration,
    FlatCalibration,
    P2VMCalibration,
    VisibilityObservables,
    WavelengthCalibration,
)


def measure_quality(product):
    """Return scientific metric -> (value, unit/definition comment), and missing reasons."""
    values, missing = {}, {}

    def put(key, number, comment):
        if np.isfinite(number):
            values[key] = (
                int(number) if isinstance(number, (int, np.integer)) else float(number),
                comment,
            )
        else:
            missing[key] = "unavailable: non-finite statistic"

    if isinstance(product, DarkCalibration):
        put(
            "dark_median_adu",
            np.median(product.mean_adu),
            "[ADU] Spatial median of temporal mean",
        )
        put(
            "dark_rms_median_adu",
            np.median(product.temporal_rms_adu),
            "[ADU] Spatial median of temporal population RMS",
        )
        missing["readout_noise"] = (
            "not applicable: raw temporal RMS is not independently measured readout noise"
        )
    elif isinstance(product, BadPixelMap):
        flags = product.bad_pixel_flags
        put(
            "bad_pixel_count",
            int(np.count_nonzero(flags)),
            "Number of actual flagged detector pixels",
        )
        put(
            "bad_dark_pixel_count",
            int(np.count_nonzero(flags & 3)),
            "Number with retained hot or dead bits",
        )
        put(
            "bad_rms_pixel_count",
            int(np.count_nonzero(flags & 4)),
            "Number with retained noisy bit",
        )
        missing["bad_flat_pixel_count"] = (
            "not applicable: retained mask has no flat-based rejection rule"
        )
    elif isinstance(product, FlatCalibration):
        for i, flux in enumerate(product.telescope_flux_adu):
            put(
                f"flat_flux_t{TELESCOPES[i]}_adu",
                flux,
                f"[ADU/DIT] Mean-frame total flux, T{TELESCOPES[i]}",
            )
        if not product.telescope_flux_adu.size:
            missing["flat_flux"] = "unavailable: individual flat images not retained"
        for reg in range(0, len(product.trace_y_pixels), 12):
            trace = (
                product.trace_y_pixels[reg] + 1
            )  # reference uses FITS/CPL one-based centers
            for suffix, number in [
                ("MED", np.median(trace)),
                ("LEFT", trace[len(trace) // 6]),
                ("RIGHT", trace[5 * len(trace) // 6]),
            ]:
                put(
                    f"profile_center_region{reg+1}_{suffix.lower()}_pixels",
                    np.floor(number * 1e6) * 1e-6,
                    "[pixel] One-based spatial trace center",
                )
            if product.profile_sigma_pixels.size:
                widths = product.profile_sigma_pixels[reg]
                valid = widths[np.isfinite(widths)]
                if len(valid):
                    put(
                        f"profile_sigma_region{reg+1}_pixels",
                        np.floor(np.median(valid) * 1e6) * 1e-6,
                        f"[pixel] Fitted Gaussian sigma; N={len(valid)}",
                    )
                else:
                    missing[f"profile_sigma_region{reg+1}_pixels"] = (
                        "unavailable: no accepted Gaussian fits"
                    )
            else:
                missing[f"profile_sigma_region{reg+1}_pixels"] = (
                    "unavailable: fitted sigma not provided; aperture width is different"
                )
    elif isinstance(product, WavelengthCalibration):
        low = np.max(np.min(product.wavelength_map_um, axis=1))
        high = np.min(np.max(product.wavelength_map_um, axis=1))
        for name, number in [("min", low), ("max", high)]:
            put(
                f"wavelength_{name}_m",
                number / 1e6,
                "[m] Coverage shared by every detector output",
            )
            put(
                f"wavelength_{name}_um",
                number,
                "[um] Coverage shared by every detector output",
            )
        residual = product.fit_residual_um
        if residual.size >= 2:
            rms = np.std(residual, ddof=1)
            put(
                "wavelength_residual_std_m",
                rms / 1e6,
                f"[m] Sample stdev of accepted lamp fit residuals; N={len(residual)}",
            )
            put(
                "wavelength_residual_std_um",
                rms,
                f"[um] Sample stdev of accepted lamp fit residuals; N={len(residual)}",
            )
        else:
            missing["wavelength_residual_std"] = (
                "unavailable: accepted fit residuals not provided"
            )
        missing["reference_wavelength"] = (
            "not applicable: GRAVITY K-band reference lines are outside PLANETES coverage"
        )
    elif isinstance(product, P2VMCalibration):
        coh_mean = []
        coh_std = []
        phase_std = []
        trans_mean = []
        for b, regions in product.baseline_regions.items():
            t1, t2 = product.baseline_telescopes[b]
            a = product.coherence_amplitude[regions, b, :]
            phase = product.coherence_phase_rad[regions, b, :]
            first = product.transmission[regions, t1, :]
            second = product.transmission[regions, t2, :]
            coh_mean.extend(a.mean(axis=1))
            coh_std.extend(a.std(axis=1, ddof=1))
            phase_std.extend(phase.std(axis=1, ddof=1))
            trans_mean.extend((first.mean(axis=1) + second.mean(axis=1)) / 2)
            if (first > 0).all() and (second > 0).all():
                ratio = a / np.sqrt(first) / np.sqrt(second) / 2
                ordered = np.sort(ratio, axis=1)
                nw = ratio.shape[-1]
                # gravi_array_get_quantile chooses floor(p*N), not numpy linear interpolation.
                put(
                    f"p2vm_coherence_mean_b{BASELINES[b]}",
                    ratio.mean(),
                    "Mean normalized coherence over four outputs and wavelengths",
                )
                put(
                    f"p2vm_coherence_min_b{BASELINES[b]}",
                    ordered[:, int(0.05 * nw)].min(),
                    "Minimum across outputs of spectral 5-percent order statistic",
                )
                put(
                    f"p2vm_coherence_max_b{BASELINES[b]}",
                    ordered[:, int(0.95 * nw)].max(),
                    "Maximum across outputs of spectral 95-percent order statistic",
                )
            else:
                missing[f"p2vm_coherence_mean_b{BASELINES[b]}"] = (
                    "unavailable: nonpositive associated transmission"
                )
        tm = np.asarray(trans_mean)
        if tm.mean() > 0:
            put(
                "p2vm_coherence_mean",
                np.mean(coh_mean) / (2 * tm.mean()),
                "Mean spectral coherence / twice mean associated transmission",
            )
            put(
                "p2vm_coherence_spectral_std",
                np.mean(coh_std) / (2 * tm.mean()),
                "Mean spectral sample stdev / twice mean transmission",
            )
        put(
            "p2vm_phase_spectral_std_rad",
            np.mean(phase_std),
            "[rad] Mean spectral sample stdev across detector regions",
        )
        for name, number in [("mean", tm.mean()), ("min", tm.min()), ("max", tm.max())]:
            put(
                f"p2vm_transmission_{name}",
                number / 2,
                "Associated transmission averaged over wavelength, divided by two",
            )
        missing["p2vm_chi_square"] = (
            "unavailable: retained fit has no propagated variance; fit RMS is retained separately"
        )
        missing["p2vm_group_delay_rejection"] = (
            "not applicable: coarse initialization bounds are not GRAVITY GD rejection limits"
        )
    elif isinstance(product, VisibilityObservables):
        missing["averaged_visibility"] = (
            "not applicable: GRAVITY final-VIS QC is measured after its averaging; this product is per-frame instrumental P2VMRED"
        )
        missing["rejected_frame_ratio"] = (
            "not applicable: wavelength-sample invalid flags are not GRAVITY frame rejection percentages"
        )
        missing["snr_and_flux_rate"] = (
            "unavailable: no propagated error model/electron gain and possibly no measured exposure timing"
        )
    return values, missing


def diagnostics(product):
    """Explicit project diagnostics, not advertised as GRAVITY/ESO QC keywords."""
    if isinstance(product, P2VMCalibration):
        matrix = build_v2pm_matrix(
            product.transmission,
            product.coherence_amplitude,
            product.coherence_phase_rad,
        )
        return {
            "p2vm_condition_max": float(
                np.max(np.linalg.cond(np.moveaxis(matrix, -1, 0)))
            )
        }
    if isinstance(product, VisibilityObservables):
        return {
            "visibility_flag_fraction": float(np.mean(product.flags)),
            "mean_visibility_amplitude": float(
                np.nanmean(
                    np.where(product.flags, np.nan, product.visibility_amplitude)
                )
            ),
        }
    return {}
