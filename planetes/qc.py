"""Small descriptive QC set using retained GRAVITY 1.9.6 definitions.

No thresholds or rejection rules are applied. Unmeasured statistics are omitted.
"""

import numpy as np

from .instrument import BASELINES, BASELINE_PAIRS, TELESCOPES
from .products import Bad, Dark, Flat, P2VM, P2VMRed, Wave


def measure_quality(product):
    """Measure the available descriptive QC statistics for one product.

    Parameters
    ----------
    product : Dark, Bad, Flat, Wave, P2VM or P2VMRed
        In-memory scientific product. Only the implemented calibration statistics
        are returned; P2VMRed currently has no registered ESO QC metrics.

    Returns
    -------
    quality : dict
        metric -> (numeric value, definition/unit comment), ready for the FITS
        writer. Unknown, unsupported and nonfinite measurements are omitted.
        Keyword names and any metre/micrometre aliases belong to io.keywords.

    Notes
    -----
    No thresholds, rejection, masks or weights are changed. DARK RMS is a temporal
    population RMS; the WAVE fit-residual statistic is a sample standard deviation
    with ddof=1. Full metric definitions are listed in docs/QC_PARAMETERS.md.
    """
    values = {}
    if isinstance(product, Dark):
        values["dark_median_adu"] = (
            np.median(product.mean_adu),
            "[ADU] Spatial median of temporal mean",
        )
        if product.temporal_rms_adu is not None:
            values["dark_rms_median_adu"] = (
                np.median(product.temporal_rms_adu),
                "[ADU] Spatial median of temporal population RMS",
            )
    elif isinstance(product, Bad):
        flags = product.bad_pixel_flags
        values.update(
            bad_pixel_count=(
                int(np.count_nonzero(flags)),
                "Number of flagged detector pixels",
            ),
            bad_dark_pixel_count=(
                int(np.count_nonzero(flags & 3)),
                "Number with hot or dead bits",
            ),
            bad_rms_pixel_count=(
                int(np.count_nonzero(flags & 4)),
                "Number with noisy bit",
            ),
        )
    elif isinstance(product, Flat) and product.telescope_flux_adu is not None:
        for telescope, flux in zip(TELESCOPES, product.telescope_flux_adu, strict=True):
            values[f"flat_flux_t{telescope}_adu"] = (
                flux,
                f"[ADU/DIT] Mean-frame total flux, T{telescope}",
            )
    elif isinstance(product, Wave):
        for name, number in (
            ("min", np.max(np.min(product.wavelength_map_m, axis=1))),
            ("max", np.min(np.max(product.wavelength_map_m, axis=1))),
        ):
            values[f"wavelength_{name}_m"] = (
                number,
                "[m] Coverage shared by every detector output",
            )
        if product.fit_residual_m is not None and product.fit_residual_m.size >= 2:
            values["wavelength_residual_std_m"] = (
                np.std(product.fit_residual_m, ddof=1),
                f"[m] Sample stdev of accepted lamp fit residuals; N={len(product.fit_residual_m)}",
            )
    elif isinstance(product, P2VM):
        for baseline, outputs in product.baseline_outputs.items():
            t1, t2 = BASELINE_PAIRS[baseline]
            first = product.transmission[outputs, t1, :]
            second = product.transmission[outputs, t2, :]
            if (first > 0).all() and (second > 0).all():
                ratio = (
                    product.coherence_amplitude[outputs, baseline, :]
                    / np.sqrt(first)
                    / np.sqrt(second)
                    / 2
                )
                values[f"p2vm_coherence_mean_b{BASELINES[baseline]}"] = (
                    ratio.mean(),
                    "Mean normalized coherence over four outputs and wavelengths",
                )
    return {
        key: (number if isinstance(number, int) else float(number), comment)
        for key, (number, comment) in values.items()
        if np.isfinite(number)
    }


def diagnostics(product):
    """Summarize optional numerical diagnostics without assigning ESO QC keywords.

    Parameters
    ----------
    product : P2VM, P2VMRed or other product
        Calibration inverse or reduced instrumental observables to inspect.

    Returns
    -------
    values : dict of str to float
        P2VM: largest wavelength-wise condition number of the stored inverse.
        P2VMRed: flagged fraction and mean VISAMP over unflagged samples; an empty
        valid set yields NaN. Other products return an empty dictionary.

    Notes
    -----
    These summaries do not reject data or alter products and are not automatically
    written to FITS. They are distinct from the registered measure_quality metrics.
    """
    if isinstance(product, P2VM):
        return {
            "p2vm_condition_max": float(
                np.max(np.linalg.cond(np.moveaxis(product.matrix, -1, 0)))
            )
        }
    if isinstance(product, P2VMRed):
        return {
            "visibility_flag_fraction": float(np.mean(product.flags)),
            "mean_visibility_amplitude": float(
                np.nanmean(
                    np.where(product.flags, np.nan, product.visibility_amplitude)
                )
            ),
        }
    return {}
