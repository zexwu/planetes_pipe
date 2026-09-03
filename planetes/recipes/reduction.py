"""
Science data reduction module for the PLANETES pipeline.

This module handles the reduction of science target data, including
visibility extraction, group delay computation, and phase calculation.
"""

from collections.abc import Mapping

import numpy as np
from numpy.typing import NDArray

from .. import log
from ..config import PipelineConfig
from ..instrument import IMAG_SLICE, N_BASELINES, N_TELESCOPES, REAL_SLICE
from ..numerics import compute_gdelay
from ..product import (
    FlatProduct,
    P2VMProduct,
    ReducedProduct,
    WaveProduct,
    read_product,
    write_product,
)
from .preproc import preprocess_target

GROUP_DELAY_RANGE_UM = (-100.0, 100.0)


def run_reduce(config: PipelineConfig, target_name: str) -> ReducedProduct:
    """Preprocess and reduce one configured science target."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.copy_used_config(config.output_dir / "config_used.yaml")
    log.info(f"--- Step: REDUCE {target_name} ---")
    flat = read_product(config.output_dir / "flat.npz", FlatProduct)
    wave = read_product(config.output_dir / "wave.npz", WaveProduct)
    p2vm = read_product(config.output_dir / "p2vm.npz", P2VMProduct)
    spectra = preprocess_target(config, target_name, flat, wave)

    log.info(f"Applying P2VM to extract visibility for {target_name}...")
    p2vmred = np.einsum("box,ofx->fbx", p2vm.p2vm, spectra)
    fluxdata = p2vmred[:, :N_TELESCOPES, :]
    coherent_flux = p2vmred[:, REAL_SLICE, :] + 1j * p2vmred[:, IMAG_SLICE, :]
    visamp, flag = normalize_visibility(coherent_flux, fluxdata, p2vm.bsl_to_tel)

    gdelay = np.zeros((len(coherent_flux), N_BASELINES))
    log.info(f"Computing group delay for {target_name}...")
    for baseline in range(N_BASELINES):
        gdelay[:, baseline] = compute_gdelay(
            coherent_flux[:, baseline],
            p2vm.wl_grid,
            search_range=GROUP_DELAY_RANGE_UM,
        )

    visphi = coherent_flux * np.exp(
        -2j * np.pi * gdelay[:, :, None] / p2vm.wl_grid[None, :]
    )
    visphi *= visphi.mean(axis=2)[:, :, None].conj()
    visphi = np.angle(visphi, deg=True)
    visphi -= visphi.mean(axis=2)[:, :, None]
    visphi = (visphi + 180) % 360 - 180

    product = ReducedProduct(
        fluxdata=fluxdata,
        coherent_flux=coherent_flux,
        visamp=visamp,
        flag=flag,
        wl_grid=p2vm.wl_grid,
        gdelay=gdelay,
        visphi=visphi,
        bsl_to_reg=p2vm.bsl_to_reg,
        bsl_to_tel=p2vm.bsl_to_tel,
    )
    write_product(config.output_dir / f"{target_name}_reduced.npz", product)
    log.info(f"Reduction for {target_name} completed.")

    log.info(f"--- Step: REDUCE {target_name} [DONE] ---")
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
