"""Shared spectral extraction, resampling and calibration-sequence reading."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numba import njit, prange
from numpy.typing import NDArray

from .instrument import DetectorSetup
from .io.keywords import RAW_IMAGE
from .io.raw import iter_raw_cube
from .io.hdu import STORAGE_RTOL
from .products import Dark, Flat, Wave


@njit(parallel=True, fastmath=True, cache=True)
def extract_sparse_spectra(cube: NDArray, ys: NDArray, xs: NDArray) -> NDArray:
    """Sum selected detector pixels into one spectrum per output and frame.

    Parameters
    ----------
    cube : ndarray, shape (frame, y, x)
        Detector samples in ADU, normally already dark-subtracted.
    ys, xs : ndarray of int, shape (output, sample)
        Zero-based spatial and dispersion pixel indices. Paired (-1, -1)
        entries are skipped; all other indices must lie within the detector.

    Returns
    -------
    spectra : ndarray, shape (output, frame, x)
        Pixel sums in ADU. Columns without selected pixels contain zero.

    Notes
    -----
    This Numba kernel does not check bounds. prepare_extraction checks geometry
    for science/P2VM extraction; direct callers must provide valid pixel indices.
    No weighting, flat correction or wavelength interpolation occurs here.
    """
    n_frame, _, nx = cube.shape
    n_reg = len(ys)
    output = np.zeros((n_reg, n_frame, nx))

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
    """Interpolate detector spectra linearly in wavenumber onto a wavelength grid.

    Parameters
    ----------
    spec : ndarray, shape (output, frame, detector x)
        Spectral values to interpolate; the operation preserves their units.
    wave_map : ndarray, shape (output, detector x)
        Increasing wavelengths in metres along each detector spectrum.
    wl_grid : ndarray, shape (wavelength,)
        Increasing target wavelengths in metres, covered by every output.
        Uniform wavelength spacing is usual but is not required by this kernel.

    Returns
    -------
    aligned : ndarray, shape (output, frame, wavelength)
        Values linearly interpolated as a function of 1 / wavelength.

    Notes
    -----
    This is point interpolation, not flux-conserving integration over bins.
    Coverage is checked by prepare_extraction before calling this kernel;
    np.interp itself returns endpoint values outside the input interval.
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


def _readonly_copy(values):
    result = np.array(values, copy=True)
    result.flags.writeable = False
    return result


@dataclass(frozen=True, slots=True)
class SpectralExtractor:
    """Prepared, read-only extraction geometry and response on one explicit target grid."""

    profile_x_pixels: NDArray
    profile_y_pixels: NDArray
    detector_shape: tuple[int, int]
    setup: DetectorSetup | None
    wavelength_map_m: NDArray
    wavelength_m: NDArray
    detector_flat_spectra: NDArray
    resampled_flat_spectra: NDArray

    def extract(self, cube: NDArray, dark: Dark) -> NDArray:
        """Subtract the selected dark and extract one chunk on the prepared grid.

        Parameters
        ----------
        cube : ndarray, shape (frame, y, x)
            Raw detector samples in ADU at calculation precision.
        dark : Dark
            Mean image in ADU with exactly the prepared detector shape.
            Exposure/setup matching is the responsibility of the file entry point.

        Returns
        -------
        spectra : ndarray, shape (output, frame, wavelength)
            Dark-subtracted, resampled signal in instrumental ADU. Input arrays
            and the prepared geometry are not modified.

        Notes
        -----
        Divide the extracted signal by the detector-coordinate flat, interpolate
        in wavenumber, then restore the prepared flat scale on the target grid.
        This flattens the response during interpolation without changing flux units.
        """
        if (
            cube.ndim != 3
            or cube.shape[1:] != self.detector_shape
            or dark.mean_adu.shape != self.detector_shape
        ):
            raise ValueError("Raw cube, dark and extraction detector geometry differ")
        spec = extract_sparse_spectra(
            cube - dark.mean_adu, self.profile_y_pixels, self.profile_x_pixels
        )
        aligned = resample_spectra(
            spec / self.detector_flat_spectra, self.wavelength_map_m, self.wavelength_m
        )
        aligned *= self.resampled_flat_spectra
        return aligned


def prepare_extraction(
    flat: Flat, wave: Wave, wavelength_m: NDArray
) -> SpectralExtractor:
    """Prepare reusable extraction geometry and response for one wavelength grid.

    Parameters
    ----------
    flat : Flat
        Detector ADU response and zero-based sparse extraction pixels.
    wave : Wave
        Increasing wavelength map in metres on the same (output, detector x) grid.
    wavelength_m : ndarray, shape (wavelength,)
        Increasing target wavelengths in metres, covered by every output.
        Science callers pass the grid stored in their P2VM calibration.

    Returns
    -------
    extractor : SpectralExtractor
        Read-only copies of geometry, wavelength grids and flat spectra.
        Reuse its extract method for chunks or individual frames.

    Notes
    -----
    Reject incompatible dimensions, invalid sparse indices and wavelength gaps.
    Only float32 storage roundoff is tolerated at coverage endpoints. Detector
    flat sums are floored at 1e-8 before division. The target flat is the reciprocal
    of the interpolated reciprocal detector flat, preserving the existing response
    normalization; it is not the direct interpolation of the flat.
    """
    height, width = flat.flat_map_adu.shape
    xs, ys = flat.profile_x_pixels, flat.profile_y_pixels
    if wave.wavelength_map_m.shape != (len(xs), width) or xs.shape != ys.shape:
        raise ValueError("Flat and wavelength detector dimensions differ")
    # Numba's sparse gather has no bounds checks: verify indices once here.
    if (
        ((xs == -1) != (ys == -1)).any()
        or (xs < -1).any()
        or (ys < -1).any()
        or (xs >= width).any()
        or (ys >= height).any()
    ):
        raise ValueError("Invalid extraction pixel indices")
    if (
        not (np.diff(wave.wavelength_map_m, axis=1) > 0).all()
        or not (np.diff(wavelength_m) > 0).all()
    ):
        raise ValueError("Wavelengths must increase for spectral resampling")
    for output, row in enumerate(wave.wavelength_map_m):
        # Only tolerate the error of one on-disk rounding at a coverage edge.
        if (
            wavelength_m[0] < row[0] - abs(row[0]) * STORAGE_RTOL
            or wavelength_m[-1] > row[-1] + abs(row[-1]) * STORAGE_RTOL
        ):
            raise ValueError(
                f"Output {output}: target {wavelength_m[0]:.9g}..{wavelength_m[-1]:.9g} m outside calibration {row[0]:.9g}..{row[-1]:.9g} m"
            )
    xs, ys = (
        _readonly_copy(flat.profile_x_pixels),
        _readonly_copy(flat.profile_y_pixels),
    )
    wave_map_m, grid_m = (
        _readonly_copy(wave.wavelength_map_m),
        _readonly_copy(wavelength_m),
    )
    detector_flat = extract_sparse_spectra(flat.flat_map_adu[None, :, :], ys, xs)
    detector_flat = np.clip(detector_flat, a_min=1e-8, a_max=None)
    grid_flat = 1 / resample_spectra(1 / detector_flat, wave_map_m, grid_m)
    return SpectralExtractor(
        xs,
        ys,
        flat.flat_map_adu.shape,
        flat.setup,
        wave_map_m,
        grid_m,
        _readonly_copy(detector_flat),
        _readonly_copy(grid_flat),
    )


def extract_spectra_from_file(
    path: Path, extractor: SpectralExtractor, dark: Dark, *, hdu: int | str = RAW_IMAGE
) -> NDArray:
    """Extract every frame of one raw FITS cube using prepared calibrations.

    Parameters
    ----------
    path : Path
        Raw ADU cube shaped (frame, y, x).
    extractor : SpectralExtractor
        Prepared detector geometry, flat response and target wavelength grid.
    dark : Dark
        Selected mean image in ADU on the same detector grid.
    hdu : int or str, optional
        Image HDU to read; use 0 explicitly for primary-HDU bench cubes.

    Returns
    -------
    spectra : ndarray, shape (output, frame, wavelength)
        Instrumental ADU spectra in original frame order.

    Notes
    -----
    Read and extract 16 frames at a time, then concatenate all extracted chunks.
    The returned spectra therefore still scale in memory with the total frame count.
    This helper does not select calibrations or check exposure/shutter metadata;
    those checks belong to the calling file workflow.
    """

    return np.concatenate(
        [extractor.extract(chunk, dark) for chunk in iter_raw_cube(path, hdu=hdu)],
        axis=1,
    )
