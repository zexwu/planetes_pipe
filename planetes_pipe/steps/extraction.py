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

    def read_variance(self, temporal_rms_adu: NDArray) -> NDArray:
        """Read-noise variance of one extracted spectrum [ADU^2]: (output, wavelength).

        The per-pixel variance summed over each aperture, scaled and resampled as
        extract() scales the signal. Resampling correlations are neglected.
        """
        if temporal_rms_adu.shape != self.detector_shape:
            raise ValueError("Dark RMS and extraction detector geometry differ")
        summed = extract_sparse_spectra(
            np.square(temporal_rms_adu)[None], self.profile_y_pixels, self.profile_x_pixels
        )
        aligned = resample_spectra(
            summed / np.square(self.detector_flat_spectra), self.wavelength_map_m, self.wavelength_m
        )
        return (aligned * np.square(self.resampled_flat_spectra))[:, 0]


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


def stacked_group_delay(
    visdata: NDArray,
    wavelength: NDArray,
    search_range: tuple[float, float] = (-100e-6, 100e-6),
    grid_step_m: float = 0.5e-6,
) -> tuple[float, float]:
    """One group delay from several frames, and how clearly the data show it.

    Sums over frames the power of the coherent sum at each trial delay, so fringe motion
    between frames does not cancel it. Returns the delay of the highest power [m], refined
    by a parabola, and that power over the median power of the search range: a few means
    no fringes in the range, and the delay then means nothing. It grows with the range:
    on 2026-11-05 frames with fringes gave at least 2000 over +-1000 um, and frames with
    fringes outside +-100 um gave at most 8 over that range. Use a wide search range: a
    fringe peak is narrow, so the range costs little and misses nothing.

    visdata is (frame, wavelength), complex; wavelength in metres. Channels are not
    weighted: choose the channels to use beforehand.
    """
    grid = np.arange(search_range[0], search_range[1], grid_step_m)
    phasors = np.exp(-2j * np.pi * grid[:, None] / wavelength[None, :])
    power = np.sum(np.abs(visdata @ phasors.T) ** 2, axis=0)
    best = int(np.argmax(power))
    delay = grid[best]
    if 0 < best < len(grid) - 1:
        low, top, high = power[best - 1 : best + 2]
        curvature = low - 2 * top + high
        if curvature != 0:
            delay = delay + 0.5 * (low - high) / curvature * grid_step_m
    return float(delay), float(power[best] / np.median(power))


def estimate_group_delay(
    visdata: NDArray,
    wavelength: NDArray,
    search_range: tuple[float, float] = (-30e-6, 30e-6),
    n_newton: int = 5,
    grid_step_m: float = 1e-8,
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
        This is an initialization interval, not a bound on the returned delay.
    n_newton : int, optional
        Number of unconstrained Newton updates of squared coherent-sum amplitude;
        default 5. A zero Hessian leaves that sample's current delay unchanged.
    grid_step_m : float, optional
        Coarse grid spacing, default 1e-8 m. Live views use a coarser grid for speed.

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

    gd_grid = np.arange(search_range[0], search_range[1], grid_step_m)
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


GD_BLOCK_FRAMES = 64  # frames sharing one fringe search; the bench OPD drifts ~1 um/s
GD_SPAN_M = 50e-6  # each frame's search about its block's fringe
NO_FRINGE_PEAK = 20.0  # stacked_group_delay peak ratio; fringes give thousands, none < 10


def tracked_group_delay(
    visdata: NDArray,
    wavelength: NDArray,
    search_range: tuple[float, float] = (-1000e-6, 1000e-6),
    grid_step_m: float = 1e-8,
) -> tuple[NDArray, NDArray]:
    """Per-frame GD [m] found wherever the fringes are in search_range: (frame,), and peak.

    Each block of GD_BLOCK_FRAMES frames finds its fringe with stacked_group_delay over
    search_range; each frame is then estimated (estimate_group_delay) within GD_SPAN_M
    of it. A single-frame search over the whole range at this precision is costly, and
    a narrow one misses fringes beyond it: on 2026-11-06T175510 T4 was at -300 um and
    a +-100 um search returned delays scattered over it, with VISAMP 0.8.

    With at least GD_BLOCK_FRAMES frames showing fringes, the frames are then estimated
    again with the dispersion of the data (dispersion_phase) removed, DISPERSION_PASSES
    times. Dispersion bends
    the phase in wavenumber and flattens the top of the delay peak, so the GD of a frame
    slides along it with the noise: on 2026-11-05 baseline 41 (11 rad of bow) the GD
    scattered 1.1 um from frame to frame, and 0.04 um once the bow was removed. The
    dispersion phase has no linear term, so the GD keeps its meaning.

    Also returns each frame's block peak ratio (stacked_group_delay): below
    NO_FRINGE_PEAK the block shows no fringe in the range and its GD means nothing.
    """
    gd_m = np.empty(len(visdata))
    peak = np.empty(len(visdata))
    blocks = []
    for start in range(0, len(visdata), GD_BLOCK_FRAMES):
        rows = slice(start, start + GD_BLOCK_FRAMES)
        centre, peak[rows] = stacked_group_delay(visdata[rows], wavelength, search_range)
        blocks.append((rows, (centre - GD_SPAN_M, centre + GD_SPAN_M)))
        gd_m[rows] = estimate_group_delay(visdata[rows], wavelength, search_range=blocks[-1][1],
                                          grid_step_m=grid_step_m)
    fringes = peak >= NO_FRINGE_PEAK
    if fringes.sum() >= GD_BLOCK_FRAMES:
        for _ in range(DISPERSION_PASSES):
            flat = visdata * np.exp(-1j * dispersion_phase(visdata[fringes], wavelength,
                                                           gd_m[fringes]))[None, :]
            for rows, span in blocks:
                gd_m[rows] = estimate_group_delay(flat[rows], wavelength, search_range=span,
                                                  grid_step_m=grid_step_m)
    return gd_m, peak


# Highest order of the dispersion polynomial in wavenumber. On 2026-11-06 the per-channel
# residual (deg) fell 2 -> 3 on every baseline (S3-S1 1.99 -> 0.62: third-order dispersion is
# real), 3 -> 4 on S3-S2 only (1.30 -> 0.99), and by < 0.1 beyond: what is left is a static
# ~20 nm ripple on the T4 baselines and noise, which a higher order would start to absorb.
DISPERSION_ORDER = 4
# The first dispersion fit uses the blurred GDs, which wash out the band edges: on a
# simulated 11 rad bow it left 0.3 rad and moved the GD 0.8 um; a second pass leaves
# 0.01 rad and 0.03 um, and a third changes nothing.
DISPERSION_PASSES = 2


def dispersion_phase(visdata: NDArray, wavelength: NDArray, gd_m: NDArray) -> NDArray:
    """Phase across the band common to all frames beyond their GD [rad]: (wavelength,).

    Each frame is shifted by its GD and its mean phase, the frames are summed, and the
    unwrapped phase of the sum is fitted with a DISPERSION_ORDER polynomial in wavenumber
    whose constant and linear terms are dropped. visdata is (frame, wavelength).
    """
    sigma = 1 / wavelength
    aligned = visdata * np.exp(-2j * np.pi * gd_m[:, None] * sigma[None, :])
    aligned *= np.exp(-1j * np.angle(aligned.sum(axis=1)))[:, None]
    total = aligned.sum(axis=0)
    x = (sigma - sigma.mean()) / np.ptp(sigma)
    coef = np.polynomial.polynomial.polyfit(x, np.unwrap(np.angle(total)), DISPERSION_ORDER,
                                            w=np.abs(total))
    coef[:2] = 0
    return np.polynomial.polynomial.polyval(x, coef)
