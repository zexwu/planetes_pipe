"""Applicable GRAVITY QC definitions; unavailable physics never becomes a zero.

References (GRAVITY 1.9.6): gravi_calib.c:183,1470-1519,2158;
gravi_wave.c:2027-2045,2266; gravi_p2vm.c:1180-1303.
"""

import numpy as np
from .models import (
    DarkCalibration,
    BadPixelMap,
    FlatCalibration,
    WavelengthCalibration,
    P2VMCalibration,
    VisibilityObservables,
)
from .instrument import BASELINES, TELESCOPES
from .p2vm import build_v2pm_matrix


def measure_quality(product):
    """Return keyword -> (measurement, definition/unit comment), and missing reasons."""
    values, missing = {}, {}

    def put(key, number, comment):
        if np.isfinite(number):
            values["ESO QC " + key] = (
                int(number) if isinstance(number, (int, np.integer)) else float(number),
                comment,
            )
        else:
            missing["ESO QC " + key] = "unavailable: non-finite statistic"

    if isinstance(product, DarkCalibration):
        put(
            "MEDIANDARK SC",
            np.median(product.mean_adu),
            "[ADU] Spatial median of temporal mean",
        )
        put(
            "DARKRMS SC",
            np.median(product.temporal_rms_adu),
            "[ADU] Spatial median of temporal population RMS",
        )
        missing["RON"] = (
            "not applicable: raw temporal RMS is not independently measured readout noise"
        )
    elif isinstance(product, BadPixelMap):
        flags = product.bad_pixel_flags
        put(
            "BADPIX SC",
            int(np.count_nonzero(flags)),
            "Number of actual flagged detector pixels",
        )
        put(
            "BADPIX_DARK SC",
            int(np.count_nonzero(flags & 3)),
            "Number with retained hot or dead bits",
        )
        put(
            "BADPIX_RMS SC",
            int(np.count_nonzero(flags & 4)),
            "Number with retained noisy bit",
        )
        missing["ESO QC BADPIX_FLAT SC"] = (
            "not applicable: retained mask has no flat-based rejection rule"
        )
    elif isinstance(product, FlatCalibration):
        for i, flux in enumerate(product.telescope_flux_adu):
            put(
                f"FLATFLUX SC{i+1}",
                flux,
                f"[ADU/DIT] Mean-frame total flux, T{TELESCOPES[i]}",
            )
        if not product.telescope_flux_adu.size:
            missing["FLATFLUX"] = "unavailable: individual flat images not retained"
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
                    f"PROFILE_CENTER SC{reg+1} {suffix}",
                    np.floor(number * 1e6) * 1e-6,
                    "[pixel] One-based spatial trace center",
                )
            if product.profile_sigma_pixels.size:
                widths = product.profile_sigma_pixels[reg]
                valid = widths[np.isfinite(widths)]
                if len(valid):
                    put(
                        f"PROFILE_WIDTH SC{reg+1} MED",
                        np.floor(np.median(valid) * 1e6) * 1e-6,
                        f"[pixel] Fitted Gaussian sigma; N={len(valid)}",
                    )
                else:
                    missing[f"PROFILE_WIDTH SC{reg+1}"] = (
                        "unavailable: no accepted Gaussian fits"
                    )
            else:
                missing[f"PROFILE_WIDTH SC{reg+1}"] = (
                    "unavailable: fitted sigma not provided; aperture width is different"
                )
    elif isinstance(product, WavelengthCalibration):
        low = np.max(np.min(product.wavelength_map_um, axis=1))
        high = np.min(np.max(product.wavelength_map_um, axis=1))
        for name, number in [("MINWAVE", low), ("MAXWAVE", high)]:
            put(
                name + " SC",
                number / 1e6,
                "[m] Coverage shared by every detector output",
            )
            put(
                name + " UM SC", number, "[um] Coverage shared by every detector output"
            )
        residual = product.fit_residual_um
        if residual.size >= 2:
            rms = np.std(residual, ddof=1)
            put(
                "RMSWAVE SC",
                rms / 1e6,
                f"[m] Sample stdev of accepted lamp fit residuals; N={len(residual)}",
            )
            put(
                "RMSWAVE UM SC",
                rms,
                f"[um] Sample stdev of accepted lamp fit residuals; N={len(residual)}",
            )
        else:
            missing["RMSWAVE"] = "unavailable: accepted fit residuals not provided"
        missing["REFWAVE"] = (
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
                    f"P2VM_COHERENCE_SC{BASELINES[b]}",
                    ratio.mean(),
                    "Mean normalized coherence over four outputs and wavelengths",
                )
                put(
                    f"P2VM_MINCOHERENCE_SC{BASELINES[b]}",
                    ordered[:, int(0.05 * nw)].min(),
                    "Minimum across outputs of spectral 5-percent order statistic",
                )
                put(
                    f"P2VM_MAXCOHERENCE_SC{BASELINES[b]}",
                    ordered[:, int(0.95 * nw)].max(),
                    "Maximum across outputs of spectral 95-percent order statistic",
                )
            else:
                missing[f"P2VM_COHERENCE_SC{BASELINES[b]}"] = (
                    "unavailable: nonpositive associated transmission"
                )
        tm = np.asarray(trans_mean)
        if tm.mean() > 0:
            put(
                "P2VM_COHERENCE_AVG_SC",
                np.mean(coh_mean) / (2 * tm.mean()),
                "Mean spectral coherence / twice mean associated transmission",
            )
            put(
                "P2VM_COHERENCERMS_AVG_SC",
                np.mean(coh_std) / (2 * tm.mean()),
                "Mean spectral sample stdev / twice mean transmission",
            )
        put(
            "P2VM_PHASERMS_AVG_SC",
            np.mean(phase_std),
            "[rad] Mean spectral sample stdev across detector regions",
        )
        for name, number in [("AVG", tm.mean()), ("MIN", tm.min()), ("MAX", tm.max())]:
            put(
                "P2VM_TRANS_" + name + "_SC",
                number / 2,
                "Associated transmission averaged over wavelength, divided by two",
            )
        missing["P2VM CHI2"] = (
            "unavailable: retained fit has no propagated variance; fit RMS is in PLANETES_P2VM_FIT_RMS"
        )
        missing["P2VM GD MAX"] = (
            "not applicable: coarse initialization bounds are not GRAVITY GD rejection limits"
        )
    elif isinstance(product, VisibilityObservables):
        missing["VISAMP/VISPHI AVG"] = (
            "not applicable: GRAVITY final-VIS QC is measured after its averaging; this product is per-frame instrumental P2VMRED"
        )
        missing["REJECTED RATIO"] = (
            "not applicable: wavelength-sample invalid flags are not GRAVITY frame rejection percentages"
        )
        missing["SNR/FLUX RATE"] = (
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
