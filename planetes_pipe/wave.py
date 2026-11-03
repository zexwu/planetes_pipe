"""WAVE: detector wavelength solution from a lamp exposure and laboratory lines.

    from planetes_pipe import wave
    out = wave.run(lamp_raw, lamp_dark_path, flat_path, output_dir="calibration")  # {"WAVE": path}
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.signal import find_peaks

from planetes_pipe import files, log
from planetes_pipe.dark import Dark, read_dark
from planetes_pipe.flat import Flat, read_flat
from planetes_pipe.instrument import Raw, check_setup, layout
from planetes_pipe.kernels import _grid_search_kernel, _ransac_kernel, extract_sparse_spectra

@dataclass(slots=True)
class Wave:
    """Wavelength solution in metres for each detector output."""

    wavelength_map_m: NDArray  # (output, x)
    fit_residual_m: NDArray | None = None
    fit_x_pixels: NDArray | None = None
    fit_y_pixels: NDArray | None = None


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


SURFACE_DEGREE    = 3


RANSAC_ITERATIONS = 1_000_000


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

    check_setup(raw.setup, dark.setup, exposure=True, role="DARK")
    check_setup(raw.setup, flat.setup, exposure=False, role="FLAT")
    image = raw.mean()
    if image.shape != dark.mean_adu.shape or image.shape != flat.flat_map_adu.shape:
        raise ValueError("Lamp/dark/flat detector shapes differ")
    image = image - dark.mean_adu
    return fit_wavelength_solution(image, flat, REFERENCE_WAVELENGTH_M)


def fit_wavelength_solution(
    wave_image_adu: NDArray, flat: Flat, reference_wavelength_m: NDArray
) -> Wave:
    """Fit a global cubic wavelength surface to matched lamp-line positions.

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
    reference, then fit lambda(x, y) with all monomials of total degree <= 3.
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
    # Fixed bench reference quartet; no automatic S/N-based output selection.
    mean_spectrum = lamp_spectra[-4:].mean(axis=0)

    # Gaussian-equivalent MAD threshold above the median.
    def peak_threshold(_spec: NDArray) -> np.floating[Any]:
        med = np.median(_spec)
        mad = np.median(abs(_spec - med))
        return med + DETECTION_SIGMA * mad * 1.4826

    mean_peaks, _ = find_peaks(
        mean_spectrum, height=peak_threshold(mean_spectrum), distance=5
    )
    mean_matches, _ = match_lines_ransac(
        lines_m, mean_peaks, tol=3, n_iter=RANSAC_ITERATIONS
    )
    log.info(
        "Matched mean spectrum peaks to known lines: "
        f"{len(mean_peaks)} peaks -> {len(mean_matches)} matches."
    )

    for output in range(layout.n_outputs):
        spec = lamp_spectra[output]

        # Locate peaks independently, then align them to the reference quartet.
        peak_pixels, _ = find_peaks(spec, height=peak_threshold(spec), distance=5)

        # A Gaussian line has a quadratic log profile near its peak. Refine
        # only positive samples with sufficient log curvature (fixed bench cuts).
        pn1, pct, pp1 = spec[peak_pixels - 1], spec[peak_pixels], spec[peak_pixels + 1]
        _pn1, _pct, _pp1 = (
            np.log(abs(pn1) + 1e-8),
            np.log(abs(pct) + 1e-8),
            np.log(abs(pp1) + 1e-8),
        )

        good = (pn1 > 1) & (pp1 > 1) & (pct > 1) & (abs(2 * _pct - _pp1 - _pn1) > 1)

        p_shift = np.zeros(peak_pixels.shape)
        p_shift[good] = 0.5 * (_pp1 - _pn1)[good] / (2 * _pct - _pp1 - _pn1)[good]
        peak_pixels = 1.0 * peak_pixels + p_shift

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
        log.info(
            f"OUTPUT {output:02d}: {len(peak_pixels)} peaks ∩ {len(lines_m)} "
            f"known lines -> {len(both)} matched."
        )

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

    # Fit lambda(x, y) to the matched detector coordinates.
    fit_x = np.concatenate(line_x)
    fit_y = np.concatenate(line_y)
    fit_wave_m = np.concatenate(line_wave_m)

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
    design = surface_design_matrix(fit_x, fit_y, max_deg=deg)
    terms = [f"x{d - i}y{i}" for d in range(deg + 1) for i in range(d + 1)]
    log.debug("Kernel used for wavelength_map_m fit: ")
    log.debug(", ".join(terms))
    coeffs, _, _, _ = np.linalg.lstsq(design, fit_wave_m, rcond=None)
    residuals = fit_wave_m - design @ coeffs
    std = np.std(residuals)
    log.info(f"Initial std dev of residuals: {std * 1e9:.2f} nm")
    for iteration in range(3):
        log.info(f"Performing outlier rejection [{iteration + 1}/3]")
        reject = abs(residuals) > 3 * std
        coeffs = np.linalg.lstsq(design[~reject], fit_wave_m[~reject], rcond=None)[0]
        residuals = fit_wave_m - design @ coeffs
        std = np.std(residuals[~reject])
        log.info(
            f"    Std dev of residuals: {std * 1e9:.2f} nm "
            f"after rejecting {reject.sum()} outliers"
        )

    # Evaluate the fitted 2-D polynomial on every detector trace.
    trace_x = np.repeat(np.arange(nx)[None, :], layout.n_outputs, axis=0)
    trace_y = y_trace
    basis = surface_basis(trace_x, trace_y, max_deg=deg)
    wave_map_m = np.einsum("i,iyx->yx", coeffs, basis)

    wave = Wave(
        wavelength_map_m=wave_map_m,
        fit_residual_m=residuals[~reject].copy(),
        fit_x_pixels=fit_x[~reject].copy(),
        fit_y_pixels=fit_y[~reject].copy(),
    )
    log.info("Computed full wavelength map")
    log.info("--- Step: WAVE [DONE] ---")
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
        log.warning(f"Grid search is large ({total_iters} iterations).")

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


def qc(product: Wave) -> dict:
    """Common wavelength coverage and lamp-residual sample stdev (ddof=1), metres."""
    residual = product.fit_residual_m
    n = 0 if residual is None else residual.size
    low = float(np.max(np.min(product.wavelength_map_m, axis=1)))
    high = float(np.min(np.max(product.wavelength_map_m, axis=1)))
    std = float(np.std(residual, ddof=1)) if n >= 2 else None
    return {
        "ESO QC MINWAVE SC": low,
        "ESO QC MAXWAVE SC": high,
        "ESO QC RMSWAVE SC": std,
        "ESO QC NWAVE SC": n,
    }


def write_wave(path: Path, product: Wave, *, header=None) -> None:
    """Save a WAVE solution (metres) with its QC; never overwrite."""
    extensions = [
        files.detector_table(),
        files.table("WAVE_DATA_SC", [files.column(f"DATA{i + 1}", row[None], "m")
                                    for i, row in enumerate(product.wavelength_map_m)]),
    ]
    if product.fit_residual_m is not None:
        extensions.append(files.table("PLANETES_WAVE_FIT", [
            files.column("RESIDUAL", product.fit_residual_m, "m"),
            files.column("X", product.fit_x_pixels, "pixel"),
            files.column("Y", product.fit_y_pixels, "pixel"),
        ]))
    files.save_product(path, "WAVE", extensions, header=header, qc=qc(product))


def read_wave(path: Path) -> Wave:
    """Read a checksummed WAVE solution at computation precision."""
    with files.open_product(path, "WAVE") as hdus:
        fitted = "PLANETES_WAVE_FIT" in hdus
        return Wave(
            wavelength_map_m=np.stack([
                files.read_column(hdus, "WAVE_DATA_SC", f"DATA{i + 1}")[0]
                for i in range(layout.n_outputs)
            ]),
            fit_residual_m=files.read_column(hdus, "PLANETES_WAVE_FIT", "RESIDUAL")
            if fitted else None,
            fit_x_pixels=files.read_column(hdus, "PLANETES_WAVE_FIT", "X")
            if fitted else None,
            fit_y_pixels=files.read_column(hdus, "PLANETES_WAVE_FIT", "Y")
            if fitted else None,
        )


def run(wavelamp_raw, dark, flat, *, output_dir) -> dict[str, Path]:
    """Recipe planetes_wave: WAVELAMP_RAW + its DARK + FLAT -> WAVE."""
    product = make_wave(files.read_raw(wavelamp_raw, "WAVELAMP_RAW"), read_dark(dark),
                        read_flat(flat))
    header = files.provenance("planetes_wave", wavelamp_raw,
                             {"WAVELAMP_RAW": [wavelamp_raw], "DARK": [dark], "FLAT": [flat]})
    out = files.output_paths(output_dir, wavelamp_raw, "WAVE")
    write_wave(out["WAVE"], product, header=header)
    return out
