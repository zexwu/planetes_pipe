"""
Science data reduction module for the PLANETES pipeline.

This module handles the reduction of science target data, including
visibility extraction, group delay computation, and phase calculation.
"""

from collections.abc import Mapping

import numpy as np
from numpy.typing import NDArray

from .. import log
from ..instrument import IMAG_SLICE, N_BASELINES, N_TELESCOPES, REAL_SLICE
from ..numerics import compute_gdelay
from ..product import (
    P2VMProduct,
    ReducedProduct,
)

GROUP_DELAY_RANGE_UM = (-100.0, 100.0)


def compute_reduce(spectra: NDArray, p2vm: P2VMProduct) -> ReducedProduct:
    """Apply P2VM and calculate the retained instrumental visibility observables."""
    log.info("Applying P2VM to extract visibility for science spectra...")
    p2vmred = np.einsum("box,ofx->fbx", p2vm.p2vm, spectra)
    photometric_flux_adu = p2vmred[:, :N_TELESCOPES, :]
    coherent_flux_adu = p2vmred[:, REAL_SLICE, :] + 1j * p2vmred[:, IMAG_SLICE, :]
    visibility_amplitude, flags = normalize_visibility(
        coherent_flux_adu,
        photometric_flux_adu,
        p2vm.baseline_telescopes,
    )

    group_delay_um = np.zeros((len(coherent_flux_adu), N_BASELINES))
    log.info("Computing group delay for science spectra...")
    for baseline in range(N_BASELINES):
        group_delay_um[:, baseline] = compute_gdelay(
            coherent_flux_adu[:, baseline],
            p2vm.wavelength_um,
            search_range=GROUP_DELAY_RANGE_UM,
        )

    differential_phase_deg = coherent_flux_adu * np.exp(
        -2j * np.pi * group_delay_um[:, :, None] / p2vm.wavelength_um[None, :]
    )
    differential_phase_deg *= differential_phase_deg.mean(axis=2)[:, :, None].conj()
    differential_phase_deg = np.angle(differential_phase_deg, deg=True)
    differential_phase_deg -= differential_phase_deg.mean(axis=2)[:, :, None]
    differential_phase_deg = (differential_phase_deg + 180) % 360 - 180

    product = ReducedProduct(
        photometric_flux_adu=photometric_flux_adu,
        coherent_flux_adu=coherent_flux_adu,
        visibility_amplitude=visibility_amplitude,
        flags=flags,
        wavelength_um=p2vm.wavelength_um,
        group_delay_um=group_delay_um,
        differential_phase_deg=differential_phase_deg,
        baseline_regions=p2vm.baseline_regions,
        baseline_telescopes=p2vm.baseline_telescopes,
    )
    log.info("Reduction for science spectra completed.")

    log.info("--- Step: REDUCE science spectra [DONE] ---")
    return product


def normalize_visibility(
    coherent_flux: NDArray,
    flux: NDArray,
    baseline_telescopes: Mapping[int, tuple[int, int]],
) -> tuple[NDArray, NDArray]:
    """Normalize coherent flux and flag non-positive or non-finite denominators."""
    visamp = np.full(coherent_flux.shape, np.nan, dtype=float)
    flag = np.ones(coherent_flux.shape, dtype=bool)

    for baseline, (telescope_1, telescope_2) in baseline_telescopes.items():
        flux_product = flux[:, telescope_1, :] * flux[:, telescope_2, :]
        valid = (
            (flux_product > 0)
            & np.isfinite(flux_product)
            & np.isfinite(coherent_flux[:, baseline, :])
        )
        visamp[:, baseline, :][valid] = np.abs(
            coherent_flux[:, baseline, :][valid]
        ) / np.sqrt(flux_product[valid])
        flag[:, baseline, :] = ~valid

    return visamp, flag
