"""FLAT step: telescope inputs, extraction aperture, detector response and traces."""

from collections.abc import Iterable, Mapping
from dataclasses import replace
from pathlib import Path

import numpy as np
from numba import njit
from numpy.typing import NDArray
from scipy.signal import find_peaks, medfilt

from . import log
from .dark import frame_mean
from .instrument import N_OUTPUTS, TELESCOPES, check_setup
from .io.raw import (
    check_raw_calibrations,
    identify_raw,
    iter_raw_cube,
    read_raw_metadata,
)
from .products import (
    BadPixelMap,
    Dark,
    Flat,
)

PROFILE_WIDTH = 10.0


def calibrate_flat_file(
    telescope_flats: Mapping[str, Path] | Iterable[Path],
    dark: Dark,
    bad: BadPixelMap,
    *,
    profile_width: float = PROFILE_WIDTH,
) -> Flat:
    """Read four telescope flats with their DARK/BAD; return response and geometry.

    ``profile_width`` is the full extraction aperture in detector pixels.
    For incomplete bench headers, supply a {telescope: path} mapping.
    """

    telescope_flats = identify_raw(telescope_flats, "FLAT_RAW")
    maps = []
    setup = read_raw_metadata(telescope_flats[TELESCOPES[0]], category="FLAT_RAW").setup
    for name in TELESCOPES:
        check_raw_calibrations(telescope_flats[name], dark, category="FLAT_RAW")

        check_setup(
            read_raw_metadata(telescope_flats[name], category="FLAT_RAW").setup,
            setup,
            exposure=False,
            role="flat sequence",
        )
        maps.append(frame_mean(iter_raw_cube(telescope_flats[name])) - dark.mean_adu)
    result = compute_flat(dark, bad, maps, profile_width=profile_width)
    return replace(result, setup=setup)


def compute_flat(
    dark: Dark,
    bad: BadPixelMap,
    flat_maps: list[NDArray],
    *,
    n_outputs: int = N_OUTPUTS,
    profile_width: float = PROFILE_WIDTH,
) -> Flat:
    """Compute flat maps and traces from pre-reduced detector images."""
    if bad.bad_pixel_flags.shape != dark.mean_adu.shape:
        raise ValueError("BAD and DARK detector shapes differ")
    bad_map = bad.bad_pixel_flags

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
    sigmas = []
    profile_xs = []
    profile_ys = []

    log.info("Building sparse extraction profiles...")
    for i, ref_y in enumerate(ref_ys):
        log.debug(f"    Tracing spectrum {i+1}/{n_outputs} at Y={ref_y}...")
        xs = np.arange(w)
        ys, ys_err, fitted_sigma = trace_spectrum(
            flat_map,
            bad_map,
            ref_x=w // 2,
            ref_y=ref_y,
            range_y=h_half,
            diagnostics=True,
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
        sigmas.append(fitted_sigma)

    width = max(map(len, profile_xs))
    padded_x = np.full((n_outputs, width), -1, dtype=np.int64)
    padded_y = np.full_like(padded_x, -1)
    for i, (sx, sy) in enumerate(zip(profile_xs, profile_ys)):
        padded_x[i, : len(sx)] = sx
        padded_y[i, : len(sy)] = sy
    product = Flat(
        profile_x_pixels=padded_x,
        profile_y_pixels=padded_y,
        flat_map_adu=flat_map,
        trace_y_pixels=np.array(ys_list),
        profile_sigma_pixels=np.asarray(sigmas),
        telescope_flux_adu=np.asarray(
            [np.sum(image, dtype=np.float64) for image in flat_maps]
        ),
        setup=dark.setup,
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


def trace_spectrum(image, bad_mask, ref_x, ref_y, range_y, *, diagnostics=False):
    ny, nx = image.shape
    y_centers = np.full(nx, np.nan)
    y_errs = np.full(nx, np.nan)
    widths = np.full(nx, np.nan)

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
                widths[x] = abs(new_popt[2])
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

    return (y_centers, y_errs, widths) if diagnostics else (y_centers, y_errs)
