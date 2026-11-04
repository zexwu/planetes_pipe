"""Spectral extraction and group delay shared by the WAVE, P2VM and science steps."""

from dataclasses import dataclass

import numpy as np
from numba import njit, prange
from numpy.typing import NDArray

from planetes_pipe.io.raw import Raw
from planetes_pipe.steps.dark import Dark
from planetes_pipe.steps.flat import Flat, extract_sparse_spectra
from planetes_pipe.steps.wave import Wave


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
            Exposure/setup matching is checked by the calling step.

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
        if wavelength_m[0] < row[0] or wavelength_m[-1] > row[-1]:
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
        wave_map_m,
        grid_m,
        _readonly_copy(detector_flat),
        _readonly_copy(grid_flat),
    )


def extract_spectra(raw: Raw, extractor: SpectralExtractor, dark: Dark) -> NDArray:
    """Extract every frame of a raw cube using prepared calibrations.

    Parameters
    ----------
    raw : Raw
        Raw ADU cube shaped (frame, y, x).
    extractor : SpectralExtractor
        Prepared detector geometry, flat response and target wavelength grid.
    dark : Dark
        Selected mean image in ADU on the same detector grid.

    Returns
    -------
    spectra : ndarray, shape (output, frame, wavelength)
        Instrumental ADU spectra in original frame order.

    Notes
    -----
    Read and extract 16 frames at a time, then concatenate all extracted chunks.
    The returned spectra therefore still scale in memory with the total frame count.
    This helper does not select calibrations or check exposure/shutter metadata;
    those checks belong to the calling step (extract_p2vm_spectra, reduce_science).
    """

    return np.concatenate(
        [extractor.extract(chunk, dark) for chunk in raw.chunks()], axis=1
    )


def estimate_group_delay(
    visdata: NDArray,
    wavelength: NDArray,
    search_range: tuple[float, float] = (-30e-6, 30e-6),
    n_newton: int = 5,
) -> NDArray:
    """Locate the coherent-sum maximum and refine its optical delay with Newton steps.

    Parameters
    ----------
    visdata : complex ndarray, shape (sample, wavelength)
        Coherent spectra or dimensionless phasors. Each row is fitted independently;
        all wavelength channels enter equally, without masking or noise weights.
    wavelength : ndarray, shape (wavelength,)
        Wavelengths in metres, matching the final axis of visdata.
    search_range : tuple of float, optional
        Coarse delay interval [lower, upper) in metres; default +/-30 micrometres.
        The fixed grid spacing is 1e-8 m. This is an initialization interval,
        not a bound on the returned delay.
    n_newton : int, optional
        Number of unconstrained Newton updates of squared coherent-sum amplitude;
        default 5. A zero Hessian leaves that sample's current delay unchanged.

    Returns
    -------
    group_delay_m : ndarray, shape (sample,)
        Delay maximizing abs(sum(C(lambda) * exp(-2j*pi*GD/lambda))) locally,
        in metres, with this explicit phase/sign convention.

    Notes
    -----
    This estimator does not certify a detected fringe. An all-zero spectrum
    returns the lower search edge, and nonfinite inputs can yield nonfinite
    results. Interpret these values together with signal validity diagnostics.
    """
    k = -2j * np.pi / wavelength
    w = -2 * np.pi / wavelength

    gd_grid = np.arange(search_range[0], search_range[1], 1e-8)
    phasors_grid = np.exp(gd_grid[:, None] * k[None, :])
    coherence = np.abs(np.tensordot(visdata, phasors_grid, axes=([-1], [-1])))

    best = np.argmax(coherence, axis=-1)
    gd_current = gd_grid[best]

    for _ in range(n_newton):
        gd_exp = gd_current[:, None]
        rotated = visdata * np.exp(1j * w * gd_exp)
        sum_0 = np.sum(rotated, axis=-1)
        sum_1 = np.sum(rotated * (1j * w), axis=-1)
        sum_2 = np.sum(rotated * (-(w**2)), axis=-1)

        sum_0_conjugate = np.conj(sum_0)
        gradient = 2 * np.real(sum_1 * sum_0_conjugate)
        hessian = 2 * np.real(sum_2 * sum_0_conjugate + sum_1 * np.conj(sum_1))
        difference = np.zeros_like(gradient)
        valid = hessian != 0
        difference[valid] = gradient[valid] / hessian[valid]
        gd_current = gd_current - difference

    return gd_current
