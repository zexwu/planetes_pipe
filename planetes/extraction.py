"""Shared spectral extraction, resampling and calibration-sequence reading."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numba import njit, prange
from numpy.typing import NDArray

from .io.raw import iter_raw_cube
from .products import (
    require_parent,
    validate,
    validate_wavelength,
)


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
def resample_spectra(spec: NDArray, wave_map: NDArray, wl_grid: NDArray) -> NDArray:
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


def _snapshot(values):
    result = np.array(values, copy=True)
    result.flags.writeable = False
    return result


@dataclass(frozen=True, slots=True)
class SpectralExtractor:
    """Prepared, read-only extraction geometry and response on one explicit target grid."""

    profile_x_pixels: NDArray
    profile_y_pixels: NDArray
    detector_shape: tuple[int, int]
    setup: object
    wavelength_map_m: NDArray
    wavelength_m: NDArray
    detector_flat_spectra: NDArray
    resampled_flat_spectra: NDArray

    def extract(self, cube: NDArray, dark) -> NDArray:
        """Subtract one explicitly supplied dark and extract every frame of this chunk."""
        if (
            cube.ndim != 3
            or cube.shape[1:] != self.detector_shape
            or dark.mean_adu.shape != self.detector_shape
        ):
            raise ValueError("Raw cube, dark and extraction detector geometry differ")
        spec = extract_spec_sparse(
            cube - dark.mean_adu, self.profile_y_pixels, self.profile_x_pixels
        )
        aligned = resample_spectra(
            spec / self.detector_flat_spectra,
            self.wavelength_map_m,
            self.wavelength_m,
        )
        aligned *= self.resampled_flat_spectra
        return aligned


def prepare_extraction(flat, wave, wavelength_m: NDArray) -> SpectralExtractor:
    """Validate scientific parents and coverage once, then snapshot the required arrays."""
    validate(flat)
    validate(wave)
    validate_wavelength(wavelength_m)
    require_parent(wave, "flat", flat)
    if wave.wavelength_map_m.shape[1] != flat.flat_map_adu.shape[1]:
        raise ValueError("Flat and wavelength detector dimensions differ")
    for output, row in enumerate(wave.wavelength_map_m):
        if wavelength_m[0] < row[0] or wavelength_m[-1] > row[-1]:
            raise ValueError(
                f"Output {output}: target {wavelength_m[0]:.9g}..{wavelength_m[-1]:.9g} m outside calibration {row[0]:.9g}..{row[-1]:.9g} m"
            )
    xs, ys = _snapshot(flat.profile_x_pixels), _snapshot(flat.profile_y_pixels)
    wave_map, grid = _snapshot(wave.wavelength_map_m), _snapshot(wavelength_m)
    spec_flat_detector = extract_spec_sparse(flat.flat_map_adu[None, :, :], ys, xs)
    spec_flat_detector = np.clip(spec_flat_detector, a_min=1e-8, a_max=None)
    spec_flat = 1 / resample_spectra(1 / spec_flat_detector, wave_map, grid)
    return SpectralExtractor(
        xs,
        ys,
        flat.flat_map_adu.shape,
        flat.setup,
        wave_map,
        grid,
        _snapshot(spec_flat_detector),
        _snapshot(spec_flat),
    )


def extract_file(path: Path, extractor, dark) -> NDArray:
    """Read and extract a whole file, retaining all frames and the 16-frame chunking."""

    return np.concatenate(
        [extractor.extract(chunk, dark) for chunk in iter_raw_cube(path)], axis=1
    )
