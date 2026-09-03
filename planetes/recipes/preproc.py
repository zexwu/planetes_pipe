"""Detector-spectrum extraction shared by calibration and target recipes."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numba import njit, prange
from numpy.typing import NDArray

from .. import log
from ..config import PipelineConfig
from ..cube import iter_cube
from ..instrument import BASELINES, N_OUTPUTS, N_TELESCOPES, TELESCOPES
from ..numerics import frame_mean
from ..product import FlatProduct, WaveProduct

WAVELENGTH_MIN_UM = 1.15
WAVELENGTH_MAX_UM = 1.30
WAVELENGTH_CHANNELS = 151


@dataclass(frozen=True, slots=True)
class CalibrationSpectra:
    """Extracted calibration cubes passed directly to the P2VM recipe."""

    telescope_spectra: NDArray
    baseline_spectra: NDArray
    phase_reference: NDArray
    baseline_regions: Mapping[int, list[int]]
    baseline_telescopes: Mapping[int, tuple[int, int]]
    wavelength_um: NDArray


@dataclass(frozen=True, slots=True)
class _ExtractionModel:
    flat: FlatProduct
    wavelength_map_um: NDArray
    detector_flat_spectra: NDArray
    resampled_flat_spectra: NDArray
    wavelength_um: NDArray


def preprocess_calibration(
    config: PipelineConfig,
    flat: FlatProduct,
    wave: WaveProduct,
) -> CalibrationSpectra:
    """Extract and resample the P2VM calibration cubes."""
    log.info("Extracting calibration spectra...")
    model = _extraction_model(flat, wave)
    calibration = config.calibration

    spec_bsl = np.asarray(
        [
            _extract_resampled_spectra(
                calibration.p2vm[name],
                model.flat.dark_map_adu,
                model,
            )
            for name in BASELINES
        ]
    )
    spec_tel = np.asarray(
        [
            _extract_resampled_spectra(
                calibration.flats[name],
                model.flat.dark_map_adu,
                model,
            )
            for name in TELESCOPES
        ]
    )

    phase_reference_spectra = np.empty(
        (N_OUTPUTS, 0, len(model.wavelength_um)), dtype=np.float64
    )
    if phase_reference := calibration.phase_reference:
        phase_dark = model.flat.dark_map_adu
        if phase_reference.dark is not None:
            log.info(
                f"    Building phase-reference dark map from {phase_reference.dark}..."
            )
            phase_dark = frame_mean(iter_cube(phase_reference.dark))
        phase_reference_spectra = _extract_resampled_spectra(
            phase_reference.data,
            phase_dark,
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


def preprocess_target(
    config: PipelineConfig,
    target_name: str,
    flat: FlatProduct,
    wave: WaveProduct,
) -> NDArray:
    """Extract and resample one science target cube."""
    log.info(f"Extracting spectra for target {target_name}...")
    model = _extraction_model(flat, wave)
    target = config.target(target_name)
    dark = model.flat.dark_map_adu
    if target.dark is not None:
        log.info(f"    Building target dark map from {target.dark}...")
        dark = frame_mean(iter_cube(target.dark))
    spectra = _extract_resampled_spectra(
        target.data,
        dark,
        model,
    )
    log.info(f"Target spectra extracted: {target_name}")
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


@njit(parallel=True, fastmath=True, cache=True)
def extract_spec_sparse(cube: NDArray, ys: NDArray, xs: NDArray) -> NDArray:
    """
    Extract spectra from a cube using sparse pixel indices per output channel.

    Args:
        cube (NDArray): Input cube with shape `(n_frame, ny, nx)`.
        ys (NDArray): Per-output Y indices; negative values are invalid sentinels.
        xs (NDArray): Per-output X indices; negative values are invalid sentinels.

    Returns:
        NDArray: Extracted spectra with shape `(n_reg, n_frame, nx)`.
    """
    n_frame, _, nx = cube.shape
    n_reg = len(ys)
    output = np.zeros((n_reg, n_frame, nx), dtype=np.float32)

    for o in prange(n_reg):
        y_idx = ys[o]
        x_idx = xs[o]
        for f in range(n_frame):
            for i in range(len(y_idx)):
                x = x_idx[i]
                y = y_idx[i]
                if x < 0 or y < 0:
                    continue
                output[o, f, x] += cube[f, y, x]
    return output


@njit(parallel=True, cache=True)
def twopx_interp(spec: NDArray, wave_map: NDArray, wl_grid: NDArray) -> NDArray:
    """
    Resample spectra from detector pixels onto a uniform wavelength grid.

    Args:
        spec (NDArray): Input spectra with shape `(n_reg, n_frame, n_pixels)`.
        wave_map (NDArray): Wavelength at each detector pixel with shape `(n_reg, n_pixels)`.
        wl_grid (NDArray): Target wavelength grid with shape `(n_wave,)`.

    Returns:
        NDArray: Resampled spectra with shape `(n_reg, n_frame, n_wave)`.
    """
    n_reg, n_frame, _ = spec.shape
    n_wave = len(wl_grid)

    wn_map = 1.0 / wave_map
    wn_grid = 1.0 / wl_grid

    out = np.zeros((n_reg, n_frame, n_wave))

    for r in prange(n_reg):
        # We reverse (::-1) because np.interp expects increasing X values
        # and wavenumber usually decreases as pixel index increases.
        x_in = wn_map[r, ::-1]

        for f in range(n_frame):
            y_in = spec[r, f, ::-1]
            out[r, f, :] = np.interp(wn_grid, x_in, y_in)

    return out
