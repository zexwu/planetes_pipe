"""Descriptive product diagnostics; these measurements do not reject or alter data."""

import numpy as np

from .models import (
    DarkCalibration,
    P2VMCalibration,
    VisibilityObservables,
    WavelengthCalibration,
)
from .p2vm import build_v2pm_matrix


def measure_quality(product: object) -> dict[str, float]:
    """Measure bad-pixel fraction, coverage, matrix conditioning, or visibility flags."""
    if isinstance(product, DarkCalibration):
        return {
            "bad_pixel_fraction": float(
                np.count_nonzero(product.bad_pixel_flags) / product.bad_pixel_flags.size
            )
        }
    if isinstance(product, WavelengthCalibration):
        return {
            "wavelength_min_um": float(np.min(product.wavelength_map_um)),
            "wavelength_max_um": float(np.max(product.wavelength_map_um)),
        }
    if isinstance(product, P2VMCalibration):
        matrix = build_v2pm_matrix(
            product.transmission,
            product.coherence_amplitude,
            product.coherence_phase_rad,
        )
        condition = float(np.max(np.linalg.cond(np.moveaxis(matrix, -1, 0))))
        return {"p2vm_condition_max": condition} if np.isfinite(condition) else {}
    if isinstance(product, VisibilityObservables):
        return {
            "visibility_flag_fraction": float(
                np.count_nonzero(product.flags) / product.flags.size
            )
        }
    return {}
