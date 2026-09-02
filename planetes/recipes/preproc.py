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

    spec_tel: NDArray
    spec_bsl: NDArray
    spec_wavesc: NDArray
    bsl_to_reg: Mapping[int, list[int]]
    bsl_to_tel: Mapping[int, tuple[int, int]]
    wl_grid: NDArray


@dataclass(frozen=True, slots=True)
class TargetSpectra:
    """One target cube extracted onto the calibrated wavelength grid."""

    spec: NDArray


@dataclass(frozen=True, slots=True)
class _ExtractionModel:
    flat: FlatProduct
    wave_map: NDArray
    detector_flat: NDArray
    aligned_flat: NDArray
    wavelength_grid: NDArray


def preprocess_calibration(
    config: PipelineConfig,
    flat: FlatProduct,
    wave: WaveProduct,
) -> CalibrationSpectra:
    """Extract and resample the P2VM calibration cubes."""
    log.info("--- Step: PREPROC P2VM ---")
    model = _extraction_model(flat, wave)
    calibration = config.calibration

    log.info("Preprocessing P2VM and FLAT files...")
    spec_bsl = np.asarray(
        [
            _extract_cube(
                calibration.p2vm[name],
                model.flat.dark_map,
                model,
            )
            for name in BASELINES
        ]
    )
    spec_tel = np.asarray(
        [
            _extract_cube(
                calibration.flats[name],
                model.flat.dark_map,
                model,
            )
            for name in TELESCOPES
        ]
    )

    spec_wavesc = np.empty((N_OUTPUTS, 0, len(model.wavelength_grid)), dtype=np.float64)
    if phase_reference := calibration.phase_reference:
        phase_dark = model.flat.dark_map
        if phase_reference.dark is not None:
            phase_dark = _mean_cube(phase_reference.dark)
        spec_wavesc = _extract_cube(
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
        spec_tel=spec_tel,
        spec_bsl=spec_bsl,
        spec_wavesc=spec_wavesc,
        bsl_to_reg=bsl_to_reg,
        bsl_to_tel=bsl_to_tel,
        wl_grid=model.wavelength_grid,
    )
    log.info("--- Step: PREPROC P2VM [DONE] ---")
    log.info("")
    return spectra


def preprocess_target(
    config: PipelineConfig,
    target_name: str,
    flat: FlatProduct,
    wave: WaveProduct,
) -> TargetSpectra:
    """Extract and resample one science target cube."""
    log.info(f"--- Step: PREPROC {target_name} ---")
    model = _extraction_model(flat, wave)
    target = config.target(target_name)
    dark = model.flat.dark_map if target.dark is None else _mean_cube(target.dark)
    spec = _extract_cube(
        target.data,
        dark,
        model,
    )
    log.info(f"Preprocessed target: {target_name}")
    log.info(f"--- Step: PREPROC {target_name} [DONE] ---")
    log.info("")
    return TargetSpectra(spec=spec)


def _extraction_model(flat: FlatProduct, wave: WaveProduct) -> _ExtractionModel:
    wl_grid = np.linspace(
        WAVELENGTH_MIN_UM,
        WAVELENGTH_MAX_UM,
        WAVELENGTH_CHANNELS,
    )
    spec_flat_detector = extract_spec_sparse(
        flat.flat_map[None, :, :], flat.profile_ys, flat.profile_xs
    )
    spec_flat_detector = np.clip(spec_flat_detector, a_min=1e-8, a_max=None)
    spec_flat = 1 / twopx_interp(1 / spec_flat_detector, wave.wave_map, wl_grid)
    return _ExtractionModel(
        flat=flat,
        wave_map=wave.wave_map,
        detector_flat=spec_flat_detector,
        aligned_flat=spec_flat,
        wavelength_grid=wl_grid,
    )


def _mean_cube(path: Path) -> NDArray:
    log.info(f"    Building dark map from {path}...")
    return frame_mean(iter_cube(path))


def _extract_cube(
    path: Path,
    dark: NDArray,
    model: _ExtractionModel,
) -> NDArray:
    log.info(f"    Extracting spectra from {path}...")
    aligned_chunks = []
    for data in iter_cube(path):
        spec = extract_spec_sparse(
            data - dark,
            model.flat.profile_ys,
            model.flat.profile_xs,
        )
        spec_aligned = twopx_interp(
            spec / model.detector_flat,
            model.wave_map,
            model.wavelength_grid,
        )
        spec_aligned *= model.aligned_flat
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
