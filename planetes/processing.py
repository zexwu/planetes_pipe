"""Detector-spectrum extraction shared by calibration and target recipes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from . import log
from .algorithms.extraction import extract_spec_sparse, twopx_interp
from .cube import iter_cube
from .instrument import BASELINES, N_OUTPUTS, N_TELESCOPES, TELESCOPES
from .numerics import frame_mean
from .product import CalibrationSpectra, FlatProduct, WaveProduct

WAVELENGTH_MIN_UM = 1.15
WAVELENGTH_MAX_UM = 1.30
WAVELENGTH_CHANNELS = 151


@dataclass(frozen=True, slots=True)
class _ExtractionModel:
    flat: FlatProduct
    wavelength_map_um: NDArray
    detector_flat_spectra: NDArray
    resampled_flat_spectra: NDArray
    wavelength_um: NDArray


def extract_calibration_spectra(
    flat: FlatProduct,
    wave: WaveProduct,
    *,
    telescope_flats: Mapping[str, Path],
    baseline_cubes: Mapping[str, Path],
    phase_reference: Path | None = None,
    phase_dark: Path | None = None,
) -> CalibrationSpectra:
    """Read and extract explicitly selected cubes without loading a run configuration."""
    log.info("Extracting calibration spectra...")
    model = _extraction_model(flat, wave)

    spec_bsl = np.asarray(
        [
            _extract_resampled_spectra(
                baseline_cubes[name],
                model.flat.dark_map_adu,
                model,
            )
            for name in BASELINES
        ]
    )
    spec_tel = np.asarray(
        [
            _extract_resampled_spectra(
                telescope_flats[name],
                model.flat.dark_map_adu,
                model,
            )
            for name in TELESCOPES
        ]
    )

    phase_reference_spectra = np.empty(
        (N_OUTPUTS, 0, len(model.wavelength_um)), dtype=np.float64
    )
    if phase_reference is not None:
        dark = model.flat.dark_map_adu
        if phase_dark is not None:
            log.info(f"    Building phase-reference dark map from {phase_dark}...")
            dark = frame_mean(iter_cube(phase_dark))
        phase_reference_spectra = _extract_resampled_spectra(
            phase_reference,
            dark,
            model,
        )

    tel_regs = {}
    for telescope, flux in enumerate(spec_tel):
        mean_flux = flux.mean(axis=(1, 2))
        active = np.argsort(mean_flux)[-N_OUTPUTS // 2 :]
        tel_regs[telescope] = np.sort(active).tolist()

    bsl_regs = {}
    for baseline, flux in enumerate(spec_bsl):
        std_flux = flux.std(axis=1).mean(axis=-1)
        active = np.argsort(std_flux)[-4:]
        bsl_regs[baseline] = np.sort(active).tolist()

    bsl_to_reg = {}
    bsl_to_tel = {}
    for telescope_1 in range(N_TELESCOPES):
        for telescope_2 in range(N_TELESCOPES):
            if telescope_1 >= telescope_2:
                continue
            baseline_name = TELESCOPES[telescope_1] + TELESCOPES[telescope_2]
            baseline = BASELINES.index(baseline_name)
            from_telescopes = set(tel_regs[telescope_1]) & set(tel_regs[telescope_2])
            from_modulation = set(bsl_regs[baseline])
            if from_telescopes != from_modulation:
                raise ValueError(
                    f"Region mapping mismatch for baseline {baseline_name}: "
                    f"{sorted(from_telescopes)} from flats, "
                    f"{sorted(from_modulation)} from modulation"
                )
            bsl_to_reg[baseline] = sorted(bsl_regs[baseline])
            bsl_to_tel[baseline] = (telescope_1, telescope_2)

    spectra = CalibrationSpectra(
        telescope_spectra=spec_tel,
        baseline_spectra=spec_bsl,
        phase_reference=phase_reference_spectra,
        baseline_regions=bsl_to_reg,
        baseline_telescopes=bsl_to_tel,
        wavelength_um=model.wavelength_um,
    )
    log.info("Calibration spectra extracted.")
    return spectra


def extract_target_spectra(
    data: Path,
    flat: FlatProduct,
    wave: WaveProduct,
    *,
    dark: Path | None = None,
) -> NDArray:
    """Read and extract one science cube using explicit calibration objects."""
    log.info(f"Extracting spectra from {data}...")
    model = _extraction_model(flat, wave)
    dark_map = model.flat.dark_map_adu
    if dark is not None:
        log.info(f"    Building target dark map from {dark}...")
        dark_map = frame_mean(iter_cube(dark))
    spectra = _extract_resampled_spectra(data, dark_map, model)
    log.info(f"Target spectra extracted: {data}")
    return spectra


def _extraction_model(flat: FlatProduct, wave: WaveProduct) -> _ExtractionModel:
    wl_grid = np.linspace(
        WAVELENGTH_MIN_UM,
        WAVELENGTH_MAX_UM,
        WAVELENGTH_CHANNELS,
    )
    spec_flat_detector = extract_spec_sparse(
        flat.flat_map_adu[None, :, :],
        flat.profile_y_pixels,
        flat.profile_x_pixels,
    )
    spec_flat_detector = np.clip(spec_flat_detector, a_min=1e-8, a_max=None)
    spec_flat = 1 / twopx_interp(
        1 / spec_flat_detector,
        wave.wavelength_map_um,
        wl_grid,
    )
    return _ExtractionModel(
        flat=flat,
        wavelength_map_um=wave.wavelength_map_um,
        detector_flat_spectra=spec_flat_detector,
        resampled_flat_spectra=spec_flat,
        wavelength_um=wl_grid,
    )


def _extract_resampled_spectra(
    path: Path,
    dark: NDArray,
    model: _ExtractionModel,
) -> NDArray:
    """Dark-subtract, flat-correct, extract, and resample one detector cube."""
    log.info(f"    Reducing detector cube to calibrated spectra: {path}")
    aligned_chunks = []
    for data in iter_cube(path):
        spec = extract_spec_sparse(
            data - dark,
            model.flat.profile_y_pixels,
            model.flat.profile_x_pixels,
        )
        spec_aligned = twopx_interp(
            spec / model.detector_flat_spectra,
            model.wavelength_map_um,
            model.wavelength_um,
        )
        spec_aligned *= model.resampled_flat_spectra
        aligned_chunks.append(spec_aligned)
    return np.concatenate(aligned_chunks, axis=1)
