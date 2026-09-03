"""Detector flat-field and spectral-trace recipe."""

import numpy as np
from numba import njit
from numpy.typing import NDArray
from scipy.signal import find_peaks, medfilt

from .. import log
from ..config import PipelineConfig
from ..cube import iter_cube
from ..instrument import N_OUTPUTS, TELESCOPES
from ..numerics import frame_mean, frame_mean_std
from ..product import FlatProduct, write_product

PROFILE_WIDTH = 10
BAD_THRESHOLD = 5.0


def run_flat(config: PipelineConfig) -> FlatProduct:
    """Build detector maps and sparse spectral traces."""
    log.info("--- Step: FLAT ---")
    log.info("Building DARK, BAD and FLAT maps...")
    dark_map, dark_std = frame_mean_std(iter_cube(config.calibration.dark))
    flat_maps = [
        frame_mean(iter_cube(config.calibration.flats[name])) - dark_map
        for name in TELESCOPES
    ]
    product = build_flat_product(
        dark_map,
        dark_std,
        flat_maps,
    )
    write_product(config.output_dir / "flat.npz", product)
    log.info("Saved flat, dark, bad-pixel, and trace data.")

    log.info("--- Step: FLAT [DONE] ---")
    return product


def build_flat_product(
    dark_map: NDArray,
    dark_std: NDArray,
    flat_maps: list[NDArray],
    *,
    n_outputs: int = N_OUTPUTS,
    profile_width: int = PROFILE_WIDTH,
    bad_threshold: float = BAD_THRESHOLD,
) -> FlatProduct:
    """Compute flat maps and traces from pre-reduced detector images."""
    bad_map = compute_bad_map_from_stats(dark_map, dark_std, threshold=bad_threshold)

    flat_map = np.mean(flat_maps, axis=0)

    h, w = flat_map.shape
    h_half = h // n_outputs // 2

    spatial_profile = flat_map.sum(axis=1)
    thresh = np.percentile(spatial_profile, 75)
    ref_ys, props = find_peaks(spatial_profile, distance=h_half, height=thresh)
    log.debug(f"Found {len(ref_ys)} Peaks with THRESH {thresh:.0f}...")
    if len(ref_ys) < n_outputs:
        raise ValueError(
            f"Expected {n_outputs} spectral traces, found only {len(ref_ys)}"
        )

    top_indices = np.argsort(props["peak_heights"])[-n_outputs:]
    ref_ys = np.sort(ref_ys[top_indices])

    ys_list = []
    profile_xs = []
    profile_ys = []

    log.info("Building sparse extraction profiles...")
    for i, ref_y in enumerate(ref_ys):
        log.debug(f"    Tracing spectrum {i+1}/{n_outputs} at Y={ref_y}...")
        xs = np.arange(w)
        ys, ys_err = trace_spectrum(
            flat_map, bad_map, ref_x=w // 2, ref_y=ref_y, range_y=h_half
        )

        ys_med = medfilt(ys, kernel_size=3)
        good = (ys_err < 0.5) & (np.arange(w) < w - 50) & (np.arange(w) > 50)
        poly_coeffs = np.polyfit(xs[good], ys_med[good], deg=3)
        ys_smoothed = np.polyval(poly_coeffs, xs)
        Y, X = np.indices((h, w))

        spatial_mask = np.abs(Y - ys_smoothed[X]) < profile_width / 2

        # Keep rectangular arrays for Numba; extractors skip bad-pixel sentinels.
        sparse_x = X[spatial_mask].copy()
        sparse_y = Y[spatial_mask].copy()
        invalid = bad_map[sparse_y, sparse_x] > 0
        sparse_x[invalid] = -1
        sparse_y[invalid] = -1
        profile_xs.append(sparse_x)
        profile_ys.append(sparse_y)

        ys_list.append(ys_smoothed)

    product = FlatProduct(
        profile_xs=np.array(profile_xs),
        profile_ys=np.array(profile_ys),
        flat_map=flat_map,
        dark_map=dark_map,
        bad_map=bad_map,
        ys=np.array(ys_list),
    )
    return product


@njit(fastmath=True, cache=True)
def gaussian(x, a, x0, sigma, c):
    return a * np.exp(-0.5 * ((x - x0) / sigma) ** 2) + c


@njit(fastmath=True, cache=True)
def gaussian_jacobian(x, a, x0, sigma):
    """
    Returns partial derivatives wrt (a, x0, sigma, c)
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
    """
    Fast nonlinear Gaussian fit using Gauss–Newton.
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


def trace_spectrum(image, bad_mask, ref_x, ref_y, range_y):
    ny, nx = image.shape
    y_centers = np.full(nx, np.nan)
    y_errs = np.full(nx, np.nan)

    y_full = np.arange(ny)

    def _scan(start_x, end_x, step, p0):
        popt = p0.copy()
        y_curr = popt[1]

        for x in range(start_x, end_x, step):
            y0 = int(max(0, y_curr - range_y // 2))
            y1 = int(min(ny, y0 + range_y))

            valid = bad_mask[y0:y1, x] == 0
            if valid.sum() < 5:
                y_centers[x] = y_curr
                y_errs[x] = popt[2]
                continue

            y = y_full[y0:y1][valid].astype(np.float64)
            z = image[y0:y1, x][valid].astype(np.float64)

            if not (y[0] <= popt[1] <= y[-1]):
                popt[1] = y_curr = y.mean()

            new_popt, y_err = fit_gaussian(y, z, popt)

            if abs(new_popt[1] - y_curr) < 2.0 and y_err < 0.5:
                popt = new_popt
                y_curr = popt[1]

            y_centers[x] = y_curr
            y_errs[x] = y_err

        return popt

    col = image[:, ref_x]
    p0 = np.array(
        [col.max() - np.median(col), ref_y, 3, np.median(col)],
        dtype=np.float64,
    )

    _scan(ref_x, nx, 1, p0)
    _scan(ref_x - 1, -1, -1, p0)

    return y_centers, y_errs


def compute_bad_map_from_stats(
    mean_img: NDArray, std_img: NDArray, threshold: float = 5.0
) -> NDArray:
    """Build a bad-pixel map from precomputed temporal mean and RMS images."""
    BAD_HOT, BAD_DEAD, BAD_RON = 1, 2, 4

    med_mean = np.median(mean_img)
    med_rms = np.median(std_img)

    range_max = med_mean + threshold * med_rms
    range_min = med_mean - threshold * med_rms
    std_max = med_rms * threshold

    bpm = np.zeros(mean_img.shape, dtype=np.int16)

    bpm[mean_img > range_max] |= BAD_HOT
    bpm[mean_img < range_min] |= BAD_DEAD
    bpm[std_img > std_max] |= BAD_RON

    return bpm
