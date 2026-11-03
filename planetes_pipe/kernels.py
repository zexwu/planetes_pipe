"""Numba kernels: every compiled function of the pipeline is in this file.

Kept apart so a pure-NumPy fallback can replace this one module if numba is
not accepted by ESO. Callers pass plain float64/int arrays.
"""

import numpy as np
from numba import njit, prange
from numpy.typing import NDArray

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


@njit(fastmath=True, cache=True)
def gaussian(x, a, x0, sigma, c):
    """Evaluate a Gaussian spatial profile plus a constant background.

    Parameters
    ----------
    x : ndarray
        Pixel coordinates at which to evaluate the profile.
    a : float
        Peak height above the background, in ADU.
    x0, sigma : float
        Center and Gaussian standard deviation, in pixels.
    c : float
        Constant background in ADU.

    Returns
    -------
    profile : ndarray
        ADU values at x, with the same shape as x.
    """
    return a * np.exp(-0.5 * ((x - x0) / sigma) ** 2) + c


@njit(fastmath=True, cache=True)
def gaussian_jacobian(x, a, x0, sigma):
    """Evaluate profile derivatives in the fit's parameter order.

    Parameters
    ----------
    x : ndarray
        Spatial pixel coordinates.
    a : float
        Gaussian peak height above background, in ADU.
    x0, sigma : float
        Center and standard deviation in pixels.

    Returns
    -------
    da, dx0, dsigma, dc : ndarray
        Derivatives of the ADU model with respect to (a, x0, sigma, c).
        Each array has the same shape as x.
    """
    dx = x - x0
    inv_s2 = 1.0 / (sigma * sigma)
    e = np.exp(-0.5 * dx * dx * inv_s2)

    da = e
    dx0 = a * e * dx * inv_s2
    ds = a * e * dx * dx / sigma**3
    dc = np.ones_like(x)

    return da, dx0, ds, dc


@njit(fastmath=True, cache=True)
def fit_gaussian(x, y, p0, max_iter=6):
    """Fit an unweighted Gaussian plus background to one spatial cross-section.

    Parameters
    ----------
    x, y : ndarray, shape (sample,)
        Pixel coordinates and corresponding ADU values, after bad-pixel removal.
    p0 : ndarray, shape (4,)
        Initial (amplitude_adu, center_pixel, sigma_pixel, background_adu).
    max_iter : int, optional
        Number of damped Gauss-Newton updates; default 6, with no convergence exit.

    Returns
    -------
    parameters : ndarray, shape (4,)
        Fitted values in the same order as p0; sigma is bounded below by 0.3 pixel.
    center_error : float
        Formal center error in pixels from residual-scaled normal equations.
        A singular/nonfinite system returns p0 and an error sentinel of 1e8.

    Notes
    -----
    The diagonal damping is fixed at 1e-3. The error estimate is a local fit
    heuristic, not a propagated detector-noise uncertainty.
    """

    a, x0, sigma, c = p0
    for _ in range(max_iter):
        model = gaussian(x, a, x0, sigma, c)
        r = y - model

        da, dx0, ds, dc = gaussian_jacobian(x, a, x0, sigma)

        # Normal equations J^T J dp = J^T r
        JtJ = np.zeros((4, 4))
        Jtr = np.zeros(4)

        for i in range(x.size):
            Ji = np.array([da[i], dx0[i], ds[i], dc[i]])
            JtJ += np.outer(Ji, Ji)
            Jtr += Ji * r[i]

        # Damping for stability
        for i in range(4):
            JtJ[i, i] += 1e-3

        det = np.linalg.det(JtJ)
        if det == 0.0 or not np.isfinite(det):
            return p0, 1e8
        dp = np.linalg.solve(JtJ, Jtr)

        a += dp[0]
        x0 += dp[1]
        sigma += dp[2]
        c += dp[3]

        # Hard constraints
        if sigma < 0.3:
            sigma = 0.3
        # ---- Error estimate ----
        dof = max(1, x.size - 4)
        resid_var = np.sum(r * r) / dof
        cov = np.linalg.inv(JtJ) * resid_var
        x0_err = np.sqrt(cov[1, 1])

    return np.array([a, x0, sigma, c]), x0_err


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
