"""WAVE step: lamp/reference inputs, line matching and wavelength surface fit."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numba import njit
from numpy.typing import NDArray
from scipy.signal import find_peaks

from . import log
from .dark import frame_mean
from .extraction import extract_sparse_spectra
from .instrument import N_OUTPUTS
from .io.fits import read_product
from .io.raw import check_raw_calibrations, iter_raw_cube
from .products import (
    Dark,
    Flat,
    Wave,
    WavelengthReference,
)

DETECTION_SIGMA = 3.0
SURFACE_DEGREE = 3
RANSAC_ITERATIONS = 1_000_000


def make_wave(
    path: Path,
    dark: Dark,
    flat: Flat,
    *,
    reference: WavelengthReference | None = None,
) -> Wave:
    """Read the lamp with its DARK and FLAT geometry; return wavelengths in metres."""

    check_raw_calibrations(path, dark, flat, category="WAVELAMP_RAW")
    reference = load_wavelength_reference() if reference is None else reference
    image = frame_mean(iter_raw_cube(path))
    if image.shape != dark.mean_adu.shape or image.shape != flat.flat_map_adu.shape:
        raise ValueError("Lamp/dark/flat detector shapes differ")
    image = image - dark.mean_adu
    return fit_wavelength_solution(image, flat, reference.wavelength_m)


def fit_wavelength_solution(
    wave_image_adu: NDArray,
    flat: Flat,
    reference_wavelength_m: NDArray,
) -> Wave:
    """Fit the retained wavelength solution from a dark-subtracted lamp image."""

    profile_x_pixels = flat.profile_x_pixels
    profile_y_pixels = flat.profile_y_pixels
    _, nx = flat.flat_map_adu.shape
    detector_x = np.arange(nx)
    y_trace = flat.trace_y_pixels

    lamp_spectra = extract_sparse_spectra(
        wave_image_adu[None, :, :], profile_y_pixels, profile_x_pixels
    )[:, 0, :]

    # Match lamp peaks to the known wavelength reference.
    reference_lines_m = reference_wavelength_m

    line_x_samples = []  # To store x-centroids for all lines
    line_y_samples = []  # To store y-centroids for all lines
    line_wavelength_samples_m = []  # To store wavelengths for all matched lines
    mean_spectrum = lamp_spectra[-4:].mean(axis=0)

    # Gaussian-equivalent MAD threshold above the median.
    def peak_threshold(_spec: NDArray) -> np.floating[Any]:
        med = np.median(_spec)
        mad = np.median(abs(_spec - med))
        return med + DETECTION_SIGMA * mad * 1.4826

    mean_peak_pixels, _ = find_peaks(
        mean_spectrum, height=peak_threshold(mean_spectrum), distance=5
    )
    mean_line_matches, _ = match_lines_ransac(
        reference_lines_m,
        mean_peak_pixels,
        tol=3,
        n_iter=RANSAC_ITERATIONS,
    )
    log.info(
        "Matched mean spectrum peaks to known lines: "
        f"{len(mean_peak_pixels)} peaks -> {len(mean_line_matches)} matches."
    )

    for output_index in range(N_OUTPUTS):
        spec = lamp_spectra[output_index]

        # Find peaks in the spectrum of individual region;
        peak_pixels, _ = find_peaks(spec, height=peak_threshold(spec), distance=5)

        # Add sub-pixel refinement for the detected peaks;
        # assume the log(line) is approximated by a parabola near the peak
        pn1, pct, pp1 = spec[peak_pixels - 1], spec[peak_pixels], spec[peak_pixels + 1]
        _pn1, _pct, _pp1 = (
            np.log(abs(pn1) + 1e-8),
            np.log(abs(pct) + 1e-8),
            np.log(abs(pp1) + 1e-8),
        )

        good = (pn1 > 1) & (pp1 > 1) & (pct > 1) & (abs(2 * _pct - _pp1 - _pn1) > 1)

        p_shift = np.zeros_like(peak_pixels, dtype=float)
        p_shift[good] = 0.5 * (_pp1 - _pn1)[good] / (2 * _pct - _pp1 - _pn1)[good]
        peak_pixels = 1.0 * peak_pixels + p_shift

        # match the observed peaks to the mean spectrum peaks
        # to filter out spurious detections
        output_line_matches, _ = match_peaks_grid(
            mean_peak_pixels,
            peak_pixels,
            tol=1,
            offset_limit=15,
            curve_limit=1e-3,
            slope_bounds=(0.98, 1.02),
            steps_per_unit=5,
        )

        both = list(set(output_line_matches[:, 0]) & set(mean_line_matches[:, 1]))
        log.info(
            f"OUTPUT {output_index:02d}: {len(peak_pixels)} peaks ∩ {len(reference_lines_m)} "
            f"known lines -> {len(both)} matched."
        )

        # reference_indices: indices of the matched lines in the known line list
        # peak_indices: indices of the matched peaks in the observed spectrum
        reference_indices = np.where(np.isin(mean_line_matches[:, 1], both))[0]
        peak_indices = np.where(np.isin(output_line_matches[:, 0], both))[0]
        reference_indices = mean_line_matches[reference_indices, 0]
        peak_indices = output_line_matches[peak_indices, 1]

        refined_x = peak_pixels[peak_indices]
        line_x_samples.append(refined_x)
        line_y_samples.append(np.interp(refined_x, detector_x, y_trace[output_index]))
        line_wavelength_samples_m.append(reference_lines_m[reference_indices])

    # Fit lambda(x, y) to the matched detector coordinates.
    fit_x_pixels = np.concatenate(line_x_samples)
    fit_y_pixels = np.concatenate(line_y_samples)
    fit_wavelength_m = np.concatenate(line_wavelength_samples_m)

    # Monomials use total degree <= SURFACE_DEGREE.
    def surface_basis(x, y, max_deg: int = 3) -> NDArray:
        terms = [x ** (d - i) * y**i for d in range(max_deg + 1) for i in range(d + 1)]
        return np.array(terms)

    def surface_design_matrix(x, y, max_deg: int = 3) -> NDArray:
        terms = [x ** (d - i) * y**i for d in range(max_deg + 1) for i in range(d + 1)]
        return np.column_stack(terms)

    # Fit the wl = f(x, y), using a 3rd order polynomial
    deg = SURFACE_DEGREE
    log.info("Fitting 2D wavelength solution...")
    design_matrix = surface_design_matrix(fit_x_pixels, fit_y_pixels, max_deg=deg)
    terms = [f"x{d-i}y{i}" for d in range(deg + 1) for i in range(d + 1)]
    log.debug("Kernel used for wavelength_map_m fit: ")
    log.debug(", ".join(terms))
    surface_coefficients, _, _, _ = np.linalg.lstsq(
        design_matrix, fit_wavelength_m, rcond=None
    )
    residuals = fit_wavelength_m - design_matrix @ surface_coefficients
    std = np.std(residuals)
    log.info(f"Initial std dev of residuals: {std * 1e9:.2f} nm")
    for iteration in range(3):
        log.info(f"Performing outlier rejection [{iteration + 1}/3]")
        reject = abs(residuals) > 3 * std
        surface_coefficients = np.linalg.lstsq(
            design_matrix[~reject], fit_wavelength_m[~reject], rcond=None
        )[0]
        residuals = fit_wavelength_m - design_matrix @ surface_coefficients
        std = np.std(residuals[~reject])
        log.info(
            f"    Std dev of residuals: {std * 1e9:.2f} nm "
            f"after rejecting {reject.sum()} outliers"
        )

    # Evaluate the fitted 2-D polynomial on every detector trace.
    detector_x_pixels = np.repeat(np.arange(nx)[None, :], N_OUTPUTS, axis=0)
    detector_y_pixels = y_trace
    full_design_matrix = surface_basis(
        detector_x_pixels, detector_y_pixels, max_deg=deg
    )
    wavelength_map_m = np.einsum("i,iyx->yx", surface_coefficients, full_design_matrix)

    wave_product = Wave(
        wavelength_map_m=wavelength_map_m,
        fit_residual_m=residuals[~reject].copy(),
        fit_x_pixels=fit_x_pixels[~reject].copy(),
        fit_y_pixels=fit_y_pixels[~reject].copy(),
    )
    log.info("Computed full wavelength map")
    log.info("--- Step: WAVE [DONE] ---")
    return wave_product


def match_lines_ransac(
    theoretical_lines: Sequence[float],
    obs_peaks: Sequence[float],
    tol: float = 2.0,
    min_inliers: int = 5,
    n_iter: int = 10000,
) -> tuple[NDArray, tuple[float, ...]]:
    """Match reference wavelengths to detected peak pixels with a quadratic RANSAC fit."""
    # 1. Setup Data
    obs = np.sort(np.asarray(obs_peaks, dtype=np.float64))
    mod = np.sort(np.asarray(theoretical_lines, dtype=np.float64))

    n_obs, n_mod = len(obs), len(mod)
    if n_obs < 3 or n_mod < 3:
        raise RuntimeError("Not enough points to match.")

    # 2. Run the ransac Kernel
    best_coeffs = _ransac_kernel(obs, mod, n_iter, tol, n_mod, n_obs)

    # 3. Reconstruct the Matches
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

    # 4. Final Fit
    # Refine the quadratic using ALL inliers (Least Squares)
    final_mi = [x[0] for x in final_inliers]
    final_oi = [x[1] for x in final_inliers]

    final_coeffs_refined = np.polyfit(mod[final_mi], obs[final_oi], 2)

    return np.array(final_inliers), tuple(final_coeffs_refined)


def match_peaks_grid(
    reference_pixels: Sequence[float],
    observed_pixels: Sequence[float],
    tol: float = 1,
    min_inliers: int = 6,
    offset_limit: float = 20,
    slope_bounds: tuple[float, float] = (0.98, 1.02),
    curve_limit: float = 1e-3,
    steps_per_unit: float = 2.0,  # Density of search. Higher = more robust but slower.
) -> tuple[NDArray, tuple[float, ...]]:
    """Match two sets of spectral peak pixels with a quadratic grid search."""
    # 1. Prepare Data
    obs = np.sort(np.ascontiguousarray(observed_pixels, dtype=np.float64))
    mod = np.sort(np.ascontiguousarray(reference_pixels, dtype=np.float64))

    # 2. Auto-Calculate Grid Steps
    # We need step sizes small enough so we don't "step over" a valid match.
    # Rule of thumb: Step size should cause a shift of ~0.5 * tol

    # Offset (c) steps: 0.5 pixels
    c_step = tol / steps_per_unit
    c_grid = np.arange(-offset_limit, offset_limit + c_step, c_step)

    # Slope (b) steps:
    # A change in slope d_b shifts the last pixel by d_b * max_wave.
    # We want d_b * max_wave ~ 0.5 * tol  =>  d_b = 0.5 * tol / max_wave
    max_wave = np.max(np.abs(mod)) if len(mod) > 0 else 1000.0
    b_step = (tol / 2.0) / max_wave
    b_grid = np.arange(slope_bounds[0], slope_bounds[1] + b_step, b_step)

    # Curvature (a) steps:
    # Usually we can search just 3 points: [-limit, 0, +limit] or coarse steps
    # If a is very small, we might skip it in the grid and solve it in the final fit.
    # Let's do a coarse grid for robustness.
    a_step = curve_limit / 2.0  # 5 steps total
    a_grid = np.arange(
        -curve_limit, curve_limit + a_step / 10.0, a_step
    )  # small epsilon

    # Sanity check on grid size
    total_iters = len(a_grid) * len(b_grid) * len(c_grid)
    if total_iters > 2e6:
        log.warning(f"Grid search is large ({total_iters} iterations).")

    # 3. Run Grid Search
    # This finds the "Best Approximate" parameters
    a_best, b_best, c_best = _grid_search_kernel(obs, mod, tol, a_grid, b_grid, c_grid)

    # 4. Final Refinement (The "Polish" Step)
    # Use the approximate grid parameters to identify inliers, then run Least Squares
    pred = a_best * mod**2 + b_best * mod + c_best
    matches = []
    ptr = 0
    for i, target in enumerate(pred):
        while ptr < len(obs) and obs[ptr] < target - tol:
            ptr += 1
        # Check if match
        if ptr < len(obs) and abs(obs[ptr] - target) < tol:
            matches.append((i, ptr))
            ptr += 1

    if len(matches) < min_inliers:
        raise RuntimeError(f"Grid search failed: found {len(matches)} matches.")

    final_idx = np.array(matches)

    # 5. Polyfit on the identified inliers for sub-pixel accuracy
    final_coeffs = np.polyfit(mod[final_idx[:, 0]], obs[final_idx[:, 1]], 2)

    return final_idx, tuple(final_coeffs)


@njit(fastmath=True, cache=True)
def _ransac_kernel(
    obs: NDArray, mod: NDArray, n_iter: int, tol: float, n_mod: int, n_obs: int
) -> tuple[float, float, float]:
    """
    The heavy lifting kernel compiled to machine code.
    Returns the best coefficients found.
    """
    best_count = -1
    best_coeffs = (0.0, 0.0, 0.0)

    # Pre-calculate min/max for monotonicity check optimization
    mod_min = mod[0]
    mod_max = mod[-1]
    np.random.seed(0)  # For reproducibility in Numba (uses global state)

    for _ in range(n_iter):
        # 1. Fast Random Selection of 3 unique indices
        # (Faster than numpy.choice with replace=False inside a loop)
        i1 = np.random.randint(0, n_mod)
        i2 = np.random.randint(0, n_mod)
        while i1 == i2:
            i2 = np.random.randint(0, n_mod)
        i3 = np.random.randint(0, n_mod)
        while i3 == i1 or i3 == i2:
            i3 = np.random.randint(0, n_mod)

        # Sort indices to ensure stability
        # (Though not strictly necessary for solving, good for consistency)
        if i1 > i2:
            i1, i2 = i2, i1
        if i2 > i3:
            i2, i3 = i3, i2
        if i1 > i2:
            i1, i2 = i2, i1

        # Correct sampling based on original logic:
        # Sample 3 unique MOD indices
        idxs_m = np.array([i1, i2, i3])
        m_samp = mod[idxs_m]

        # Sample 3 unique OBS indices
        o1 = np.random.randint(0, n_obs)
        o2 = np.random.randint(0, n_obs)
        while o1 == o2:
            o2 = np.random.randint(0, n_obs)
        o3 = np.random.randint(0, n_obs)
        while o3 == o1 or o3 == o2:
            o3 = np.random.randint(0, n_obs)
        idxs_o = np.array([o1, o2, o3])
        # Sort the OBS samples to match the sorted MOD samples (assuming monotonic mapping)
        # The original code did `sort(rng.choice)`,
        # implying it matches lowest-to-lowest, etc.
        idxs_o_sorted = np.sort(idxs_o)
        p_samp = obs[idxs_o_sorted]

        # 2. Solve Quadratic Equation (Vandermonde)
        # We solve: A * c = y  ->  c = inv(A) * y
        # Hardcoded 3x3 solver is faster, but linalg.solve is acceptable in Numba
        x1, x2, x3 = m_samp[0], m_samp[1], m_samp[2]
        y1, y2, y3 = p_samp[0], p_samp[1], p_samp[2]

        # Denominator for Cramer's rule / Lagrange (avoids matrix overhead)
        denom = (x1 - x2) * (x1 - x3) * (x2 - x3)
        if denom == 0:
            continue  # Should not happen with unique floats

        # Lagrange Interpolation / Analytic quadratic fit
        # A bit verbose but very fast (no matrix alloc)
        a = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / denom
        b = (x3 * x3 * (y1 - y2) + x2 * x2 * (y3 - y1) + x1 * x1 * (y2 - y3)) / denom
        c = (
            x2 * x3 * (x2 - x3) * y1
            + x3 * x1 * (x3 - x1) * y2
            + x1 * x2 * (x1 - x2) * y3
        ) / denom

        # 3. Monotonicity Check Optimization
        # Derivative d(Pix)/d(Wave) = 2*a*x + b
        # Since this is linear, we only check the ENDPOINTS of the range.
        # If it's > 0 at min and max, it's > 0 everywhere in between.
        if (2 * a * mod_min + b <= 0) or (2 * a * mod_max + b <= 0):
            continue

        # 4. Inlier Counting (Zero Allocation)
        # Instead of building a list, we just count. Memory allocation is slow.
        current_inliers = 0
        obs_ptr = 0

        for m_idx in range(n_mod):
            # Calculate prediction on the fly (no array allocation)
            mx = mod[m_idx]
            target = a * mx**2 + b * mx + c

            # Fast scan (Two pointers)
            while obs_ptr < n_obs and obs[obs_ptr] < target - tol:
                obs_ptr += 1

            if obs_ptr < n_obs:
                diff = obs[obs_ptr] - target
                # abs(diff) < tol
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
    """
    Scans the parameter grid and returns the best (a,b,c) parameters.
    """
    n_obs = len(obs)
    n_mod = len(mod)

    best_score = -1
    best_params = (0.0, 1.0, 0.0)

    # Iterate over the grid
    for a in a_grid:
        for b in b_grid:
            for c in c_grid:
                # 2. Inlier Counting
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

                    # Early exit:
                    if best_score == n_mod:
                        return best_params
    return best_params


def wavelength_reference_path() -> Path:
    """Return the installed static line-list file."""
    from importlib.resources import files

    return Path(str(files("planetes").joinpath("data/wavelength_lines.fits")))


def load_wavelength_reference(path: Path | None = None) -> WavelengthReference:
    """Load the packaged or explicitly supplied static reference, including origin labels."""

    return read_product(
        wavelength_reference_path() if path is None else path, WavelengthReference
    )
