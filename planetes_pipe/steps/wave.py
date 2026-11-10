"""WAVE: detector wavelength solution from a lamp exposure and laboratory lines.

    from planetes_pipe.steps import wave
    out = wave.run(lamp_raw, lamp_dark_path, flat_path, output_dir="calibration")  # {"WAVE": path}
"""

import logging
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits as afits
from numba import njit
from numpy.typing import NDArray
from scipy.optimize import OptimizeWarning, curve_fit
from scipy.signal import find_peaks

from planetes_pipe.instrument import layout
from planetes_pipe.io import products
from planetes_pipe.io.keywords import Keyword
from planetes_pipe.io.raw import Raw, check_calibrations, read_raw, record_saturation
from planetes_pipe.logs import step
from planetes_pipe.steps.dark import Dark, read_dark
from planetes_pipe.steps.flat import Flat, extract_sparse_spectra, read_flat

log = logging.getLogger(__name__)


@dataclass(slots=True)
class Wave:
    """Wavelength solution in metres for each detector output."""

    wavelength_map_m: NDArray  # (output, x)
    # Every fitted lamp line, accepted or rejected (fit_rejected; None: all accepted).
    fit_residual_m: NDArray | None = None  # laboratory minus fitted wavelength
    fit_x_pixels: NDArray | None = None
    fit_y_pixels: NDArray | None = None
    fit_wavelength_m: NDArray | None = None  # laboratory wavelength
    fit_output: NDArray | None = None
    fit_rejected: NDArray | None = None
    # Diagnostics for the report.
    lamp_spectra_adu: NDArray | None = None  # (output, x), extracted
    lamp_image_adu: NDArray | None = None  # (y, x), mean lamp minus its dark


def accepted_residual_m(wave: Wave) -> NDArray | None:
    """Residuals of the lines the fit kept."""
    if wave.fit_residual_m is None or wave.fit_rejected is None:
        return wave.fit_residual_m
    return wave.fit_residual_m[~wave.fit_rejected]


# Laboratory wavelengths [m], retaining the original float64 reference values.
REFERENCE_WAVELENGTH_M = np.array([
    9.6578e-07,  # pipeline_reference
    9.784500000000001e-07,  # NIST_addition
    1.04701e-06,  # pipeline_reference
    1.0673569999999998e-06,  # NIST_addition
    1.14881e-06,  # pipeline_reference
    1.166871e-06,  # NIST_addition
    1.2112300000000001e-06,  # pipeline_reference
    1.2139739999999999e-06,  # NIST_addition
    1.2343389999999998e-06,  # NIST_addition
    1.2402829999999998e-06,  # NIST_addition
    1.24393e-06,  # pipeline_reference
    1.245612e-06,  # NIST_addition
    1.2487660000000001e-06,  # NIST_addition
    1.27023e-06,  # pipeline_reference
    1.28027e-06,  # pipeline_reference
    1.2956700000000002e-06,  # pipeline_reference
    1.3008299999999999e-06,  # pipeline_reference
    1.3214000000000002e-06,  # pipeline_reference
    1.32726e-06,  # pipeline_reference
    1.3313199999999998e-06,  # pipeline_reference
    1.3367100000000001e-06,  # pipeline_reference
    1.34065e-06,  # pipeline_reference
    1.3504200000000002e-06,  # pipeline_reference
    1.35993e-06,  # pipeline_reference
    1.36227e-06,  # pipeline_reference
    1.3678599999999998e-06,  # pipeline_reference
    1.37186e-06,  # pipeline_reference
    1.38257e-06,  # pipeline_reference
    1.39075e-06,  # pipeline_reference
    1.4093699999999998e-06,  # pipeline_reference
], dtype=np.float64)


# fmt: off
DETECTION_SIGMA   = 3.0


# Total degree of lambda(x, y). Degree 4 adds the x^2 y^2 change of the dispersion's
# curvature across the outputs; held-out-line errors drop ~25% from degree 3 on the
# 2026-11-05..10 lamps, degree 5 gains < 3 pm and degree 6 overfits.
SURFACE_DEGREE    = 4


RANSAC_ITERATIONS = 1_000_000


# Line centres: Gaussian plus linear background fitted to +-CENTROID_HALF_WIDTH pixels.
CENTROID_HALF_WIDTH = 4
CENTROID_MAX_SHIFT  = 1.5           # pixels from the detected peak
CENTROID_SIGMA      = (0.3, 4.0)    # accepted Gaussian sigma range, pixels
# fmt: on


def make_wave(raw: Raw, dark: Dark, flat: Flat) -> Wave:
    """Calibrate detector wavelengths from a lamp exposure and a laboratory line list.

    Parameters
    ----------
    raw : Raw
        Raw lamp ADU cube, shaped (frame, y, x).
    dark : Dark
        Mean dark on the lamp detector grid, selected for the lamp exposure.
        The lamp may require a different dark from the other calibrations.
    flat : Flat
        Extraction pixel indices and trace centers on the same detector grid.

    Returns
    -------
    wave : Wave
        Wavelength map in metres, shaped (24 outputs, detector x), with accepted
        fit residuals and their zero-based detector coordinates. Not written to disk.

    Notes
    -----
    Average all lamp frames, subtract dark once, then fit the extracted lines.
    This step uses FLAT geometry but does not divide the lamp by the flat response.
    """

    check_calibrations(raw, dark=dark, flat=flat)
    image = raw.mean()
    if image.shape != dark.mean_adu.shape or image.shape != flat.flat_map_adu.shape:
        raise ValueError("Lamp/dark/flat detector shapes differ")
    image = image - dark.mean_adu
    wave = fit_wavelength_solution(image, flat, REFERENCE_WAVELENGTH_M)
    wave.lamp_image_adu = image.astype(np.float32)
    return wave


def _peak_threshold(spectrum: NDArray) -> np.floating[Any]:
    """Gaussian-equivalent MAD threshold above the median."""
    median = np.median(spectrum)
    mad = np.median(abs(spectrum - median))
    return median + DETECTION_SIGMA * mad * 1.4826


def _gaussian_line(t, amplitude, centre, sigma, offset, slope):
    return amplitude * np.exp(-0.5 * ((t - centre) / sigma) ** 2) + offset + slope * t


def line_centres(spectrum: NDArray, peak_pixels: NDArray) -> NDArray:
    """Sub-pixel centres of lamp lines detected at integer peak_pixels.

    Fit a Gaussian plus a linear background to +-CENTROID_HALF_WIDTH pixels around
    each peak. Fall back to the three-sample logarithmic parabola when the fit fails
    or leaves CENTROID_MAX_SHIFT / CENTROID_SIGMA, and to the integer peak when the
    three samples are not positive and concave in log.
    """
    spectrum = np.asarray(spectrum, dtype=np.float64)
    centres = np.asarray(peak_pixels, dtype=np.float64).copy()
    for k, peak in enumerate(np.asarray(peak_pixels, dtype=int)):
        lo = max(peak - CENTROID_HALF_WIDTH, 0)
        hi = min(peak + CENTROID_HALF_WIDTH + 1, spectrum.size)
        t = np.arange(lo, hi) - peak
        s = spectrum[lo:hi]
        if t.size >= 6:
            guess = [s[t == 0][0] - np.median(s), 0.0, 1.0, np.median(s), 0.0]
            try:
                with warnings.catch_warnings():  # only the centre is used, not its covariance
                    warnings.simplefilter("ignore", OptimizeWarning)
                    fit, _ = curve_fit(_gaussian_line, t, s, p0=guess, maxfev=2000)
            except (RuntimeError, ValueError):
                fit = None
            if (fit is not None and fit[0] > 0 and abs(fit[1]) < CENTROID_MAX_SHIFT
                    and CENTROID_SIGMA[0] < abs(fit[2]) < CENTROID_SIGMA[1]):
                centres[k] = peak + fit[1]
                continue
        if 0 < peak < spectrum.size - 1:
            low, mid, high = spectrum[peak - 1:peak + 2]
            if min(low, mid, high) > 0:
                low, mid, high = np.log([low, mid, high])
                curvature = 2 * mid - low - high
                if curvature > 0:
                    centres[k] = peak + 0.5 * (high - low) / curvature
    return centres


def _surface_terms(x, y, degree: int, shape: tuple[int, int]) -> list[NDArray]:
    """Monomials u^(d-i) v^i of total degree d <= degree, in fixed order.

    u, v are detector x, y scaled to [-1, 1] over shape (ny, nx); in raw pixels the
    degree-4 terms reach 1e12 and the least-squares fit loses the wavelength map.
    """
    ny, nx = shape
    u = 2 * np.asarray(x, dtype=np.float64) / (nx - 1) - 1
    v = 2 * np.asarray(y, dtype=np.float64) / (ny - 1) - 1
    return [u ** (d - i) * v**i for d in range(degree + 1) for i in range(d + 1)]


def fit_wavelength_solution(
    wave_image_adu: NDArray, flat: Flat, reference_wavelength_m: NDArray
) -> Wave:
    """Fit a global polynomial wavelength surface to matched lamp-line positions.

    Parameters
    ----------
    wave_image_adu : ndarray, shape (y, x)
        Mean lamp image with its selected dark already subtracted, in ADU.
    flat : Flat
        Sparse extraction pixels and trace centers on the lamp detector grid.
        The 24 outputs must be ordered by increasing detector y.
    reference_wavelength_m : ndarray, shape (line,)
        Strictly increasing laboratory wavelengths in metres. Matching indices
        refer to sorted lines, so the supplied list must already be ordered.

    Returns
    -------
    wave : Wave
        wavelength_map_m is shaped (24, x). fit_residual_m stores observed
        laboratory wavelength minus fitted wavelength for accepted samples;
        fit_x_pixels and fit_y_pixels give their zero-based detector positions.

    Notes
    -----
    Use the mean of the last four outputs as the reference lamp spectrum.
    Match it to the line list with quadratic RANSAC, align each output to that
    reference, then fit lambda(x, y) with all monomials of total degree <= SURFACE_DEGREE.
    Three rounds reject residuals beyond 3 population standard deviations.
    No per-line uncertainty weighting or absolute-accuracy certification is used.
    The last-four-output choice is a fixed bench convention, not a measured S/N
    selection. Its suitability must be checked for a different detector layout.
    """

    xs = flat.profile_x_pixels
    ys = flat.profile_y_pixels
    _, nx = flat.flat_map_adu.shape
    detector_x = np.arange(nx)
    y_trace = flat.trace_y_pixels

    lamp_spectra = extract_sparse_spectra(wave_image_adu[None, :, :], ys, xs)[:, 0, :]

    # Match lamp peaks to the known wavelength reference.
    lines_m = reference_wavelength_m

    line_x = []
    line_y = []
    line_wave_m = []
    line_output = []
    # Fixed bench reference quartet; no automatic S/N-based output selection.
    mean_spectrum = lamp_spectra[-4:].mean(axis=0)

    mean_peaks, _ = find_peaks(
        mean_spectrum, height=_peak_threshold(mean_spectrum), distance=5
    )
    mean_matches, _ = match_lines_ransac(
        lines_m, mean_peaks, tol=3, n_iter=RANSAC_ITERATIONS
    )
    log.info("reference spectrum (outputs %d-%d): %d peaks, %d matched to %d known lines",
             layout.n_outputs - 4, layout.n_outputs - 1, len(mean_peaks), len(mean_matches),
             len(lines_m))
    matched_per_output = []

    for output in range(layout.n_outputs):
        spec = lamp_spectra[output]

        # Locate peaks independently, then align them to the reference quartet.
        peak_pixels, _ = find_peaks(spec, height=_peak_threshold(spec), distance=5)

        peak_pixels = line_centres(spec, peak_pixels)

        # Constrain inter-output shifts with a near-unity quadratic pixel map.
        matches, _ = match_peaks_grid(
            mean_peaks,
            peak_pixels,
            tol=1,
            offset_limit=15,
            curve_limit=1e-3,
            slope_bounds=(0.98, 1.02),
            steps_per_unit=5,
        )

        both = list(set(matches[:, 0]) & set(mean_matches[:, 1]))
        matched_per_output.append(len(both))
        log.debug("output %2d: %d peaks, %d matched lines", output, len(peak_pixels), len(both))

        # line_indices: indices of the matched lines in the known line list
        # peak_indices: indices of the matched peaks in the observed spectrum
        line_indices = np.where(np.isin(mean_matches[:, 1], both))[0]
        peak_indices = np.where(np.isin(matches[:, 0], both))[0]
        line_indices = mean_matches[line_indices, 0]
        peak_indices = matches[peak_indices, 1]

        refined_x = peak_pixels[peak_indices]
        line_x.append(refined_x)
        line_y.append(np.interp(refined_x, detector_x, y_trace[output]))
        line_wave_m.append(lines_m[line_indices])
        line_output.append(np.full(len(refined_x), output))

    log.info("lines matched per output: %d-%d (mean %.1f)", min(matched_per_output),
             max(matched_per_output), np.mean(matched_per_output))
    # Fit lambda(x, y) to the matched detector coordinates.
    fit_x = np.concatenate(line_x)
    fit_y = np.concatenate(line_y)
    fit_wave_m = np.concatenate(line_wave_m)

    # Fit wl = f(x, y) with all monomials of total degree <= SURFACE_DEGREE.
    deg = SURFACE_DEGREE
    design = np.column_stack(_surface_terms(fit_x, fit_y, deg, flat.flat_map_adu.shape))
    terms = [f"x{d - i}y{i}" for d in range(deg + 1) for i in range(d + 1)]
    log.debug("surface terms: %s", ", ".join(terms))
    coeffs, _, _, _ = np.linalg.lstsq(design, fit_wave_m, rcond=None)
    residuals = fit_wave_m - design @ coeffs
    std = np.std(residuals)
    first_std = std
    log.debug("residual std %.3f nm before rejection", std * 1e9)
    for iteration in range(3):
        reject = abs(residuals) > 3 * std
        coeffs = np.linalg.lstsq(design[~reject], fit_wave_m[~reject], rcond=None)[0]
        residuals = fit_wave_m - design @ coeffs
        std = np.std(residuals[~reject])
        log.debug("rejection pass %d: %d outliers, residual std %.3f nm", iteration + 1,
                  reject.sum(), std * 1e9)
    log.info("wavelength fit (%d lines, degree %d): residual std %.2f -> %.2f nm after "
             "rejecting %d", len(fit_wave_m), deg, first_std * 1e9, std * 1e9, reject.sum())

    # Evaluate the fitted 2-D polynomial on every detector trace.
    trace_x = np.repeat(np.arange(nx)[None, :], layout.n_outputs, axis=0)
    trace_y = y_trace
    basis = np.array(_surface_terms(trace_x, trace_y, deg, flat.flat_map_adu.shape))
    wave_map_m = np.einsum("i,iyx->yx", coeffs, basis)

    wave = Wave(
        wavelength_map_m=wave_map_m,
        fit_residual_m=residuals,
        fit_x_pixels=fit_x,
        fit_y_pixels=fit_y,
        fit_wavelength_m=fit_wave_m,
        fit_output=np.concatenate(line_output),
        fit_rejected=reject,
        lamp_spectra_adu=lamp_spectra,
    )
    return wave


def match_lines_ransac(
    theoretical_lines: Sequence[float],
    obs_peaks: Sequence[float],
    tol: float = 2.0,
    min_inliers: int = 5,
    n_iter: int = 10000,
) -> tuple[NDArray, tuple[float, ...]]:
    """Match laboratory wavelengths to peak pixels with a monotonic quadratic model.

    Parameters
    ----------
    theoretical_lines : sequence of float
        Laboratory wavelengths in metres; sorted internally.
    obs_peaks : sequence of float
        Detected zero-based peak coordinates in pixels; sorted internally.
    tol : float, optional
        Maximum absolute pixel mismatch for a pair; the comparison is strict.
    min_inliers : int, optional
        Minimum accepted pairs; fewer pairs raise RuntimeError.
    n_iter : int, optional
        Number of random three-pair proposals. The Numba random seed is fixed at 0.

    Returns
    -------
    matches : ndarray of int, shape (match, 2)
        (line index, peak index) pairs into the sorted inputs, with no repeated
        peak assignment. These are not indices into an unsorted original input.
    coefficients : tuple of float
        Least-squares (a, b, c) for pixel = a * wavelength**2 + b * wavelength + c,
        refitted to all accepted pairs. Units are pixel/m**2, pixel/m and pixel.
    """
    # Match in wavelength/pixel order; returned indices refer to these sorted arrays.
    obs = np.sort(obs_peaks)
    mod = np.sort(theoretical_lines)

    n_obs, n_mod = len(obs), len(mod)
    if n_obs < 3 or n_mod < 3:
        raise RuntimeError("Not enough points to match.")

    # Score monotonic quadratic models using a fixed random sequence.
    best_coeffs = _ransac_kernel(obs, mod, n_iter, tol, n_mod, n_obs)

    # Advance the peak pointer after each match so a peak cannot be reused.
    a, b, c = best_coeffs
    pred_pixels = a * mod**2 + b * mod + c

    final_inliers = []
    obs_ptr = 0
    for m_idx in range(n_mod):
        target = pred_pixels[m_idx]
        while obs_ptr < n_obs and obs[obs_ptr] < target - tol:
            obs_ptr += 1
        if obs_ptr < n_obs and abs(obs[obs_ptr] - target) < tol:
            final_inliers.append((m_idx, obs_ptr))
            obs_ptr += 1

    if len(final_inliers) < min_inliers:
        msg = f"line matching failed: only {len(final_inliers)} matches found."
        raise RuntimeError(msg)

    # Refit the selected pairs without constraining the proposal coefficients.
    final_mi = [x[0] for x in final_inliers]
    final_oi = [x[1] for x in final_inliers]

    coeffs = np.polyfit(mod[final_mi], obs[final_oi], 2)

    return np.array(final_inliers), tuple(coeffs)


def match_peaks_grid(
    reference_pixels: Sequence[float],
    observed_pixels: Sequence[float],
    tol: float = 1,
    min_inliers: int = 6,
    offset_limit: float = 20,
    slope_bounds: tuple[float, float] = (0.98, 1.02),
    curve_limit: float = 1e-3,
    steps_per_unit: float = 2.0,
) -> tuple[NDArray, tuple[float, ...]]:
    """Align two detected peak lists using a quadratic pixel-coordinate transform.

    Parameters
    ----------
    reference_pixels, observed_pixels : sequence of float
        Zero-based peak positions in pixels; each input is sorted internally.
    tol : float, optional
        Pixel matching tolerance. Final accepted pairs satisfy abs(residual) < tol.
    min_inliers : int, optional
        Minimum matched pairs required to return a fit; otherwise raise RuntimeError.
    offset_limit : float, optional
        Symmetric search bound for constant offset c, in pixels.
    slope_bounds : tuple of float, optional
        Lower and upper search bounds for dimensionless scale b.
    curve_limit : float, optional
        Positive symmetric bound for curvature a, in inverse pixels.
    steps_per_unit : float, optional
        Offset-grid density: c step = tol / steps_per_unit. Slope and curvature
        sampling have their own fixed rules and do not use this parameter.

    Returns
    -------
    matches : ndarray of int, shape (match, 2)
        Indices into the sorted (reference, observed) lists, assigned one-to-one.
    coefficients : tuple of float
        Refitted (a, b, c) for observed_pixel = a * reference_pixel**2
        + b * reference_pixel + c. The final least-squares fit is unconstrained
        and may leave the initialization grid bounds.
    """
    obs = np.sort(observed_pixels)
    mod = np.sort(reference_pixels)

    # Sample constant offsets at the requested fraction of the matching tolerance.
    c_step = tol / steps_per_unit
    c_grid = np.arange(-offset_limit, offset_limit + c_step, c_step)

    # Adjacent slopes shift the furthest reference peak by half the tolerance.
    pixel_span = np.max(np.abs(mod)) if len(mod) > 0 else 1000.0
    b_step = (tol / 2.0) / pixel_span
    b_grid = np.arange(slope_bounds[0], slope_bounds[1] + b_step, b_step)

    # Five curvature proposals span [-curve_limit, +curve_limit].
    a_step = curve_limit / 2.0  # 5 steps total
    a_grid = np.arange(
        -curve_limit, curve_limit + a_step / 10.0, a_step
    )  # small epsilon

    # Report expensive searches without changing the requested grid.
    total_iters = len(a_grid) * len(b_grid) * len(c_grid)
    if total_iters > 2e6:
        log.warning("grid search is large (%d iterations)", total_iters)

    # Choose a one-to-one matching proposal on the fixed grid.
    a_best, b_best, c_best = _grid_search_kernel(obs, mod, tol, a_grid, b_grid, c_grid)

    # Reconstruct pairs using the final strict tolerance before least-squares fitting.
    pred = a_best * mod**2 + b_best * mod + c_best
    matches = []
    ptr = 0
    for i, target in enumerate(pred):
        while ptr < len(obs) and obs[ptr] < target - tol:
            ptr += 1
        if ptr < len(obs) and abs(obs[ptr] - target) < tol:
            matches.append((i, ptr))
            ptr += 1

    if len(matches) < min_inliers:
        raise RuntimeError(f"Grid search failed: found {len(matches)} matches.")

    final_idx = np.array(matches)

    # The refit uses the accepted pairs and may leave the initialization bounds.
    final_coeffs = np.polyfit(mod[final_idx[:, 0]], obs[final_idx[:, 1]], 2)

    return final_idx, tuple(final_coeffs)

@njit(fastmath=True, cache=True)
def _ransac_kernel(
    obs: NDArray, mod: NDArray, n_iter: int, tol: float, n_mod: int, n_obs: int
) -> tuple[float, float, float]:
    """Search quadratic wavelength-to-pixel models and retain the most matched lines.

    Parameters
    ----------
    obs, mod : ndarray
        Increasing observed pixels and laboratory wavelengths [m], respectively.
    n_iter : int
        Number of random three-pair proposals, using the fixed random seed 0.
    tol : float
        Strict pixel matching tolerance.
    n_mod, n_obs : int
        Lengths of mod and obs; supplied by match_lines_ransac.

    Returns
    -------
    coefficients : tuple of float
        (a, b, c) for pixel = a * wavelength**2 + b * wavelength + c.
        Only models increasing over the laboratory wavelength range are scored.
        Tied scores retain the first model; the caller checks the final match count.
    """
    best_count = -1
    best_coeffs = (0.0, 0.0, 0.0)

    # A quadratic is monotonic on this interval if its endpoint slopes are positive.
    mod_min = mod[0]
    mod_max = mod[-1]
    np.random.seed(0)  # For reproducibility in Numba (uses global state)

    for _ in range(n_iter):
        # Draw three distinct laboratory lines without replacement.
        i1 = np.random.randint(0, n_mod)
        i2 = np.random.randint(0, n_mod)
        while i1 == i2:
            i2 = np.random.randint(0, n_mod)
        i3 = np.random.randint(0, n_mod)
        while i3 == i1 or i3 == i2:
            i3 = np.random.randint(0, n_mod)

        # Pair samples in increasing wavelength and pixel order.
        if i1 > i2:
            i1, i2 = i2, i1
        if i2 > i3:
            i2, i3 = i3, i2
        if i1 > i2:
            i1, i2 = i2, i1

        idxs_m = np.array([i1, i2, i3])
        m_samp = mod[idxs_m]

        # Draw three distinct observed peaks.
        o1 = np.random.randint(0, n_obs)
        o2 = np.random.randint(0, n_obs)
        while o1 == o2:
            o2 = np.random.randint(0, n_obs)
        o3 = np.random.randint(0, n_obs)
        while o3 == o1 or o3 == o2:
            o3 = np.random.randint(0, n_obs)
        idxs_o = np.array([o1, o2, o3])
        # The increasing-dispersion assumption pairs lowest with lowest.
        idxs_o_sorted = np.sort(idxs_o)
        p_samp = obs[idxs_o_sorted]

        # Analytic quadratic through the three proposed wavelength/pixel pairs.
        x1, x2, x3 = m_samp[0], m_samp[1], m_samp[2]
        y1, y2, y3 = p_samp[0], p_samp[1], p_samp[2]

        # Skip repeated wavelength values even when their indices are distinct.
        denom = (x1 - x2) * (x1 - x3) * (x2 - x3)
        if denom == 0:
            continue

        # Coefficients of pixel = a * wavelength**2 + b * wavelength + c.
        a = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / denom
        b = (x3 * x3 * (y1 - y2) + x2 * x2 * (y3 - y1) + x1 * x1 * (y2 - y3)) / denom
        c = (
            x2 * x3 * (x2 - x3) * y1
            + x3 * x1 * (x3 - x1) * y2
            + x1 * x2 * (x1 - x2) * y3
        ) / denom

        # Positive endpoint derivatives ensure increasing dispersion throughout.
        if (2 * a * mod_min + b <= 0) or (2 * a * mod_max + b <= 0):
            continue

        # Count one-to-one matches; the monotonic model permits a forward-only scan.
        current_inliers = 0
        obs_ptr = 0

        for m_idx in range(n_mod):
            mx = mod[m_idx]
            target = a * mx**2 + b * mx + c

            while obs_ptr < n_obs and obs[obs_ptr] < target - tol:
                obs_ptr += 1

            if obs_ptr < n_obs:
                diff = obs[obs_ptr] - target
                if diff < tol and diff > -tol:
                    current_inliers += 1
                    obs_ptr += 1

        if current_inliers > best_count:
            best_count = current_inliers
            best_coeffs = (a, b, c)

    return best_coeffs


@njit(fastmath=True, cache=True)
def _grid_search_kernel(
    obs: NDArray,
    mod: NDArray,
    tol: int,
    a_grid: NDArray,
    b_grid: NDArray,
    c_grid: NDArray,
) -> tuple[float, float, float]:
    """Select the quadratic pixel transform with the largest one-to-one match count.

    Parameters
    ----------
    obs, mod : ndarray
        Increasing observed and reference peak coordinates in pixels.
    tol : float
        Inclusive pixel tolerance for scoring grid proposals.
    a_grid, b_grid, c_grid : ndarray
        Curvature [1/pixel], scale [dimensionless] and offset [pixel] grids.

    Returns
    -------
    parameters : tuple of float
        Best (a, b, c) for observed = a * reference**2 + b * reference + c.
        Ties retain the first proposal. Return early if all reference peaks match;
        the caller subsequently applies its strict tolerance and refits the pairs.
    """
    n_obs = len(obs)
    n_mod = len(mod)

    best_score = -1
    best_params = (0.0, 1.0, 0.0)

    for a in a_grid:
        for b in b_grid:
            for c in c_grid:
                score = 0
                ptr = 0

                for m_idx in range(n_mod):
                    pred = a * mod[m_idx] ** 2 + b * mod[m_idx] + c
                    while ptr < n_obs and obs[ptr] < pred - tol:
                        ptr += 1
                    if ptr < n_obs and abs(obs[ptr] - pred) <= tol:
                        score += 1
                        ptr += 1

                if score > best_score:
                    best_score = score
                    best_params = (a, b, c)

                    if best_score == n_mod:
                        return best_params
    return best_params



QC_WAVE_MIN = Keyword("QC.MINWAVE.SC", "Shortest wavelength covered by every output", "m")
QC_WAVE_MAX = Keyword("QC.MAXWAVE.SC", "Longest wavelength covered by every output", "m")
QC_WAVE_RMS = Keyword("QC.RMSWAVE.SC", "Sample stdev of accepted lamp fit residuals", "m")
QC_WAVE_LINES = Keyword("QC.NWAVE.SC", "Number of accepted lamp fit residuals")


def qc(product: Wave) -> dict[Keyword, float | None]:
    """Common wavelength coverage and accepted lamp-residual sample stdev (ddof=1), metres."""
    residual = accepted_residual_m(product)
    n = 0 if residual is None else residual.size
    low = float(np.max(np.min(product.wavelength_map_m, axis=1)))
    high = float(np.min(np.max(product.wavelength_map_m, axis=1)))
    std = float(np.std(residual, ddof=1)) if n >= 2 else None
    return {
        QC_WAVE_MIN: low,
        QC_WAVE_MAX: high,
        QC_WAVE_RMS: std,
        QC_WAVE_LINES: n,
    }


_IMAGES = [
    ("lamp_spectra_adu", "PLANETES_LAMP_SPECTRA", "adu"),
    ("lamp_image_adu", "PLANETES_LAMP_IMAGE", "adu"),
]


def write_wave(path: Path, product: Wave, *, header=None) -> None:
    """Save a WAVE solution (metres) with its QC; never overwrite."""
    wavelengths = [products.column(f"DATA{i + 1}", row[None], "m")
                   for i, row in enumerate(product.wavelength_map_m)]
    extensions = [
        products.detector_table(),
        afits.BinTableHDU.from_columns(wavelengths, name="WAVE_DATA_SC"),
    ]
    if product.fit_residual_m is not None:
        fit = [
            products.column("RESIDUAL", product.fit_residual_m, "m"),
            products.column("X", product.fit_x_pixels, "pixel"),
            products.column("Y", product.fit_y_pixels, "pixel"),
        ]
        for name, values, unit in (("WAVELENGTH", product.fit_wavelength_m, "m"),
                                   ("OUTPUT", product.fit_output, ""),
                                   ("REJECTED", product.fit_rejected, "")):
            if values is not None:
                if name != "WAVELENGTH":
                    values = np.asarray(values, dtype=np.int64)
                fit.append(products.column(name, values, unit))
        extensions.append(afits.BinTableHDU.from_columns(fit, name="PLANETES_WAVE_FIT"))
    extensions += [products.image_hdu(ext, getattr(product, name), unit)
                   for name, ext, unit in _IMAGES if getattr(product, name) is not None]
    products.save_product(path, "WAVE", extensions, header=header, qc=qc(product))


def read_wave(path: Path) -> Wave:
    """Read a checksummed WAVE solution at computation precision."""
    with products.open_product(path, "WAVE") as hdus:
        rows = [products.read_column(hdus, "WAVE_DATA_SC", f"DATA{i + 1}")[0]
                for i in range(layout.n_outputs)]
        wave = Wave(wavelength_map_m=np.stack(rows))
        if "PLANETES_WAVE_FIT" in hdus:
            wave.fit_residual_m = products.read_column(hdus, "PLANETES_WAVE_FIT", "RESIDUAL")
            wave.fit_x_pixels = products.read_column(hdus, "PLANETES_WAVE_FIT", "X")
            wave.fit_y_pixels = products.read_column(hdus, "PLANETES_WAVE_FIT", "Y")
            names = hdus["PLANETES_WAVE_FIT"].columns.names
            for field, column in (("fit_wavelength_m", "WAVELENGTH"), ("fit_output", "OUTPUT"),
                                  ("fit_rejected", "REJECTED")):
                if column in names:
                    setattr(wave, field, products.read_column(hdus, "PLANETES_WAVE_FIT", column))
            if wave.fit_rejected is not None:
                wave.fit_rejected = wave.fit_rejected.astype(bool)
        for name, ext, _ in _IMAGES:
            if ext in hdus:
                setattr(wave, name, products.read_image(hdus, ext))
        return wave


@step("WAVE")
def run(lamp_file, dark_file, flat_file, *, output_dir) -> dict[str, Path]:
    """planetes_wave: one WAVELAMP_RAW + its DARK + FLAT -> WAVE."""
    raw, flat = read_raw(lamp_file, "WAVELAMP_RAW"), read_flat(flat_file)
    product = make_wave(raw, read_dark(dark_file), flat)

    inputs = {"WAVELAMP_RAW": [lamp_file], "DARK": [dark_file], "FLAT": [flat_file]}
    header = products.provenance("planetes_wave", lamp_file, inputs)
    record_saturation(header, {"LAMP": raw}, flat.extracted_pixels())
    path = products.product_path(output_dir, lamp_file, "WAVE")
    write_wave(path, product, header=header)
    return {"WAVE": path}
