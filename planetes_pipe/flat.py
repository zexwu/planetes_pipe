"""FLAT step: telescope inputs, extraction aperture, detector response and traces."""

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import numpy as np
from numba import njit
from numpy.typing import NDArray
from scipy.signal import find_peaks, medfilt

from . import log
from .dark import frame_mean
from .instrument import N_OUTPUTS, TELESCOPES, check_setup
from .io.keywords import RAW_IMAGE
from .io.raw import iter_raw_cube, read_raw_metadata
from .products import Bad, Dark, Flat

APERTURE_WIDTH_PIXELS = 10.0


def make_flat(
    telescope_flats: Mapping[str, Path],
    dark: Dark,
    bad: Bad,
    *,
    hdu: int | str = RAW_IMAGE,
    aperture_width_pixels: float = APERTURE_WIDTH_PIXELS,
) -> Flat:
    """Subtract the selected dark and calibrate the detector extraction geometry.

    Parameters
    ----------
    telescope_flats : mapping of str to Path
        Four raw ADU cubes keyed by physical telescope: '4', '3', '2', '1'.
        Header shutter states, when present, must agree with these identities.
    dark : Dark
        Selected mean dark in ADU on exactly the raw detector grid. Known DIT,
        ROI and detector settings must match; dimensions are checked even when
        setup is unknown. No dark scaling or broadcasting is allowed.
    bad : Bad
        Detector bit mask shaped (y, x); every nonzero pixel is excluded from
        the sparse extraction aperture.
    hdu : int or str, optional
        Raw image HDU shared by the four files; use 0 for primary-HDU cubes.
    aperture_width_pixels : float, optional
        Full extraction aperture about the fitted trace, in pixels; default 10.
        This width is independent of the fitted Gaussian sigma.

    Returns
    -------
    flat : Flat
        Mean response in ADU, zero-based traces and sparse pixel indices,
        Gaussian sigmas, per-telescope total fluxes and detector setup.
        Saving is left to the caller.

    Notes
    -----
    Each file is averaged over all its frames, then the four dark-subtracted
    images receive equal weight. Physical outputs are ordered by increasing y.
    """

    if set(telescope_flats) != set(TELESCOPES):
        raise ValueError("Telescope flats require a path for each of 4,3,2,1")
    if len({Path(path) for path in telescope_flats.values()}) != len(TELESCOPES):
        raise ValueError("Each telescope requires a different flat file")
    maps = []
    setup = None
    for name in TELESCOPES:
        path = telescope_flats[name]
        raw = read_raw_metadata(path, category="FLAT_RAW", hdu=hdu)
        for telescope, state in enumerate(raw.shutters, 1):
            if state is not None and state != (str(telescope) in name):
                raise ValueError(
                    f"Flat for T{name} conflicts with recorded shutters: {path}"
                )
        check_setup(raw.setup, dark.setup, exposure=True, role="DARK")
        if setup is None:
            setup = raw.setup
        check_setup(raw.setup, setup, exposure=False, role="flat sequence")
        image = frame_mean(iter_raw_cube(path, hdu=hdu))
        if image.shape != dark.mean_adu.shape:
            raise ValueError(
                f"Flat and DARK detector shapes differ: "
                f"{image.shape} != {dark.mean_adu.shape} ({path})"
            )
        maps.append(image - dark.mean_adu)
    result = build_flat(maps, bad, aperture_width_pixels=aperture_width_pixels)
    return replace(result, setup=setup)


def build_flat(
    flat_images: list[NDArray],
    bad: Bad,
    *,
    n_outputs: int = N_OUTPUTS,
    aperture_width_pixels: float = APERTURE_WIDTH_PIXELS,
) -> Flat:
    """Locate spectral traces and build their bad-pixel-masked extraction apertures.

    Parameters
    ----------
    flat_images : list of ndarray
        Four already dark-subtracted mean images in ADU, each shaped (y, x),
        ordered T4, T3, T2, T1. No dark subtraction occurs here.
    bad : Bad
        Detector bit mask on the same grid; nonzero pixels are excluded.
    n_outputs : int, optional
        Number of traces to retain from the strongest spatial peaks; normally 24.
    aperture_width_pixels : float, optional
        Full aperture in pixels. Select abs(y - trace_y) < width / 2.

    Returns
    -------
    flat : Flat
        Averaged response, cubic-smoothed trace centers, fitted Gaussian sigmas
        and sparse apertures. Pixel coordinates are zero-based. Paired (-1, -1)
        entries mark bad pixels or padding. Detector setup is left unset.

    Notes
    -----
    Trace centers are median-filtered over three columns, then fitted by a cubic
    using position errors < 0.5 pixel and columns 50 < x < nx - 50. These fixed
    bench criteria are retained; they are not an adaptive trace-quality model.
    """
    flat_map = np.mean(flat_images, axis=0)
    if bad.bad_pixel_flags.shape != flat_map.shape:
        raise ValueError("BAD and FLAT detector shapes differ")
    bad_map = bad.bad_pixel_flags

    h, w = flat_map.shape
    h_half = h // n_outputs // 2

    profile = flat_map.sum(axis=1)
    thresh = np.percentile(profile, 75)
    ref_ys, props = find_peaks(profile, distance=h_half, height=thresh)
    log.debug(f"Found {len(ref_ys)} Peaks with THRESH {thresh:.0f}...")
    if len(ref_ys) < n_outputs:
        raise ValueError(
            f"Expected {n_outputs} spectral traces, found only {len(ref_ys)}"
        )

    top_indices = np.argsort(props["peak_heights"])[-n_outputs:]
    ref_ys = np.sort(ref_ys[top_indices])

    traces = []
    sigmas = []
    sparse_xs = []
    sparse_ys = []

    log.info("Building sparse extraction profiles...")
    for i, ref_y in enumerate(ref_ys):
        log.debug(f"    Tracing spectrum {i + 1}/{n_outputs} at Y={ref_y}...")
        xs = np.arange(w)
        ys, ys_err, sigma = trace_spectrum(
            flat_map, bad_map, ref_x=w // 2, ref_y=ref_y, range_y=h_half
        )

        # Smooth isolated column fluctuations before the global cubic trace fit.
        ys_med = medfilt(ys, kernel_size=3)
        # Fixed bench cuts: formal center error < 0.5 pixel and an interior x range.
        # The 50-pixel edge exclusion is retained, not inferred from this exposure.
        good = (ys_err < 0.5) & (np.arange(w) < w - 50) & (np.arange(w) > 50)
        coeffs = np.polyfit(xs[good], ys_med[good], deg=3)
        trace_y = np.polyval(coeffs, xs)
        Y, X = np.indices((h, w))

        aperture = np.abs(Y - trace_y[X]) < aperture_width_pixels / 2

        # Keep rectangular arrays for Numba; extractors skip bad-pixel sentinels.
        sparse_x = X[aperture].copy()
        sparse_y = Y[aperture].copy()
        invalid = bad_map[sparse_y, sparse_x] > 0
        sparse_x[invalid] = -1
        sparse_y[invalid] = -1
        sparse_xs.append(sparse_x)
        sparse_ys.append(sparse_y)

        traces.append(trace_y)
        sigmas.append(sigma)

    width = max(map(len, sparse_xs))
    padded_x = np.full((n_outputs, width), -1, dtype=np.int64)
    padded_y = np.full_like(padded_x, -1)
    for i, (sx, sy) in enumerate(zip(sparse_xs, sparse_ys)):
        padded_x[i, : len(sx)] = sx
        padded_y[i, : len(sy)] = sy
    product = Flat(
        profile_x_pixels=padded_x,
        profile_y_pixels=padded_y,
        flat_map_adu=flat_map,
        trace_y_pixels=np.array(traces),
        profile_sigma_pixels=np.asarray(sigmas),
        telescope_flux_adu=np.asarray([np.sum(image) for image in flat_images]),
    )
    return product


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


def trace_spectrum(image, bad_mask, ref_x, ref_y, range_y):
    """Follow one Gaussian trace from a reference column in both x directions.

    Parameters
    ----------
    image : ndarray, shape (y, x)
        Dark-subtracted detector image in ADU.
    bad_mask : ndarray, shape (y, x)
        Zero denotes a usable pixel; all other values are excluded from fitting.
    ref_x, ref_y : int
        Zero-based starting column and approximate trace center in pixels.
    range_y : int
        Height of the spatial fit window, in pixels, recentered on the last fit.

    Returns
    -------
    centers : ndarray, shape (x,)
        Trace center in pixels; retain the last center when a new fit is rejected.
    position_errors : ndarray, shape (x,)
        Center-error estimates used by the later trace-selection rule. With fewer
        than five usable pixels, the previous sigma is used as the legacy fallback.
    gaussian_sigma : ndarray, shape (x,)
        Accepted fit widths in pixels; NaN where no width was accepted.

    Notes
    -----
    Accept an update only if its center moves < 2 pixels and its error is < 0.5
    pixel. These criteria and the low-sample fallback are fixed bench choices.
    """
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

            y = y_full[y0:y1][valid]
            z = image[y0:y1, x][valid]

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
    p0 = np.array([col.max() - np.median(col), ref_y, 3, np.median(col)])

    _scan(ref_x, nx, 1, p0)
    _scan(ref_x - 1, -1, -1, p0)

    return y_centers, y_errs, widths
