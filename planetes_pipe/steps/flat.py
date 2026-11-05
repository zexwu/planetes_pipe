"""FLAT: spectral trace positions, extraction apertures and detector response.

    from planetes_pipe.steps import flat
    out = flat.run(dark_path, bad_path, flat_raws, output_dir="calibration")  # {"FLAT": path}
"""

import logging
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from astropy.io import fits as afits
from astropy.io.fits import Header
from numba import njit, prange
from numpy.typing import NDArray
from scipy.ndimage import median_filter
from scipy.signal import find_peaks, medfilt

from planetes_pipe.instrument import layout
from planetes_pipe.io import products
from planetes_pipe.io.keywords import WINDOW, Keyword
from planetes_pipe.io.raw import (
    Raw,
    check_calibrations,
    check_match,
    detector_cards,
    read_raws,
    record_saturation,
)
from planetes_pipe.logs import step
from planetes_pipe.steps.dark import Bad, Dark, read_bad, read_dark

log = logging.getLogger(__name__)


@dataclass(slots=True)
class Flat:
    """Detector flat and sparse extraction geometry; indices are zero-based."""

    profile_x_pixels: NDArray  # (output, sample), -1 sentinel
    profile_y_pixels: NDArray  # (output, sample), -1 sentinel
    flat_map_adu: NDArray  # (y, x)
    trace_y_pixels: NDArray  # (output, x)
    profile_sigma_pixels: NDArray | None = None  # (output, x)
    telescope_flux_adu: NDArray | None = None  # (telescope,)
    detector: Header | None = None  # DET cards of the raw flats
    trace_x_range: NDArray | None = None  # (output, 2), inclusive fitted limits, zero-based

    def extracted_pixels(self) -> NDArray:
        """Boolean (y, x) image: True where a pixel belongs to an extracted output."""
        mask = np.zeros(self.flat_map_adu.shape, dtype=bool)
        valid = (self.profile_x_pixels >= 0) & (self.profile_y_pixels >= 0)
        mask[self.profile_y_pixels[valid], self.profile_x_pixels[valid]] = True
        return mask


APERTURE_WIDTH_PIXELS = 10.0


TRACE_MIN_SIGNAL_FRACTION = 0.05


TRACE_BACKGROUND_SIGMA = 5.0


TRACE_SIGNAL_WINDOW = 11


def _trace_illumination(
    image: NDArray, bad_mask: NDArray, ref_y: int, range_y: int
) -> tuple[NDArray, int]:
    """Select bright columns from a masked spatial strip; return mask and seed x.

    Signal is the spatial 90th percentile minus the median of the outer three
    rows on each side. An 11-column median smooths only the selection signal.
    Require both 5% of its peak and five background-pixel MAD sigmas. These are
    trace-selection thresholds, not a propagated photometric S/N estimate.
    """
    y0 = max(0, int(ref_y) - range_y // 2)
    strip = image[y0 : y0 + range_y].copy()
    strip[bad_mask[y0 : y0 + range_y] != 0] = np.nan
    edges = np.concatenate((strip[:3], strip[-3:]))
    background = np.nanmedian(edges, axis=0)
    noise = 1.4826 * np.nanmedian(abs(edges - background))
    signal = np.nan_to_num(np.nanpercentile(strip, 90, axis=0) - background)
    signal = median_filter(signal, size=TRACE_SIGNAL_WINDOW, mode="nearest")
    threshold = max(
        TRACE_BACKGROUND_SIGMA * noise, TRACE_MIN_SIGNAL_FRACTION * signal.max()
    )
    return signal > threshold, int(signal.argmax())


def make_flat(
    flats: dict[str, Raw],
    dark: Dark,
    bad: Bad,
    *,
    aperture_width_pixels: float = APERTURE_WIDTH_PIXELS,
    metrology_trace: bool = False,
) -> Flat:
    """Subtract the selected dark and calibrate the detector extraction geometry.

    Parameters
    ----------
    flats : dict of str to Raw
        The four flats keyed by open telescope ('4', '3', '2', '1'), as returned
        by products.read_raws.
    dark : Dark
        Selected mean dark in ADU on exactly the raw detector grid. Known DIT,
        ROI and detector settings must match; dimensions are checked even when
        DET cards are unknown. No dark scaling or broadcasting is allowed.
    bad : Bad
        Detector bit mask shaped (y, x); every nonzero pixel is excluded from
        the sparse extraction aperture.
    aperture_width_pixels : float, optional
        Full extraction aperture about the fitted trace, in pixels; default 10.
        This width is independent of the fitted Gaussian sigma.

    metrology_trace : bool, optional
        Detect 25 traces and exclude the middle metrology trace, retaining 24
        science outputs in detector-y order. No absolute pixel coordinate is used.

    Returns
    -------
    flat : Flat
        Mean response in ADU, zero-based traces and sparse pixel indices,
        Gaussian sigmas, per-telescope total fluxes and the flats' DET cards.
        Not written to disk; run() does that.

    Notes
    -----
    Each file is averaged over all its frames, then the four dark-subtracted
    images receive equal weight. Physical outputs are ordered by increasing y.
    """

    maps = []
    first = flats[layout.telescopes[0]]
    for telescope in layout.telescopes:
        raw = flats[telescope]
        check_calibrations(raw, dark=dark)
        check_match(raw.header, first.detector, WINDOW, "flat sequence")
        image = raw.mean()
        if image.shape != dark.mean_adu.shape:
            raise ValueError(
                f"Flat and DARK detector shapes differ: "
                f"{image.shape} != {dark.mean_adu.shape}"
            )
        maps.append(image - dark.mean_adu)
    result = build_flat(maps, bad, n_outputs=25 if metrology_trace else layout.n_outputs,
                        aperture_width_pixels=aperture_width_pixels)
    if metrology_trace:
        keep = np.arange(25) != 12  # Middle of the 25 y-ordered traces.
        log.info("dropped the metrology trace (12 of 25, y=%.0f); %d outputs kept",
                 np.median(result.trace_y_pixels[12]), keep.sum())
        result = replace(result, profile_x_pixels=result.profile_x_pixels[keep],
                         profile_y_pixels=result.profile_y_pixels[keep],
                         trace_y_pixels=result.trace_y_pixels[keep],
                         profile_sigma_pixels=result.profile_sigma_pixels[keep],
                         trace_x_range=result.trace_x_range[keep])
    return replace(result, detector=first.detector)


def build_flat(
    flat_images: list[NDArray],
    bad: Bad,
    *,
    n_outputs: int = layout.n_outputs,
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
        entries mark bad pixels or padding. DET cards are left unset.

    Notes
    -----
    Each trace starts at its brightest smoothed flat column. Fit a cubic only
    to illuminated columns with accepted Gaussian fits and errors < 0.5 pixel;
    see _trace_illumination for the explicit signal thresholds. Centers are
    median-filtered over three columns. The returned trace retains the full
    detector grid for compatibility; outside the selected region it is an
    extrapolation, not a measured trace.
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
    log.debug("%d profile peaks above %.0f ADU (75th percentile)", len(ref_ys), thresh)
    if len(ref_ys) < n_outputs:
        raise ValueError(
            f"Expected {n_outputs} spectral traces, found only {len(ref_ys)}"
        )

    top_indices = np.argsort(props["peak_heights"])[-n_outputs:]
    ref_ys = np.sort(ref_ys[top_indices])

    traces = []
    sigmas = []
    trace_ranges = []
    sparse_xs = []
    sparse_ys = []

    for i, ref_y in enumerate(ref_ys):
        xs = np.arange(w)
        illuminated, ref_x = _trace_illumination(flat_map, bad_map, ref_y, h_half)
        ys, ys_err, sigma = trace_spectrum(
            flat_map, bad_map, ref_x=ref_x, ref_y=ref_y, range_y=h_half
        )

        # Smooth isolated column fluctuations before the global cubic trace fit.
        ys_med = medfilt(ys, kernel_size=3)
        # A rejected tracking update can retain a small error but has no sigma.
        good = illuminated & (ys_err < 0.5) & np.isfinite(sigma)
        if good.sum() < 4:
            raise ValueError(f"Trace {i}: fewer than four illuminated fitted columns")
        fit_x = xs[good]
        trace_ranges.append((fit_x[0], fit_x[-1]))
        log.debug("trace %2d: y=%d, x=%d..%d, %d fitted columns, seed x=%d", i, ref_y,
                  fit_x[0], fit_x[-1], len(fit_x), ref_x)
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

    columns = [last - first + 1 for first, last in trace_ranges]
    log.info("%d traces at y=%d..%d, fitted over x=%d..%d (%d-%d columns each)", n_outputs,
             ref_ys[0], ref_ys[-1], min(r[0] for r in trace_ranges),
             max(r[1] for r in trace_ranges), min(columns), max(columns))
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
        trace_x_range=np.asarray(trace_ranges),
        telescope_flux_adu=np.asarray([np.sum(image) for image in flat_images]),
    )
    return product


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
    ref_y need only be near the trace: the start is the background-subtracted
    centroid at ref_x, recentred three times, since a lopsided profile's peak (where
    ref_y usually comes from) can lie 2-3 pixels from its Gaussian center.
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
    p0 = np.array([col.max() - np.median(col), _seed_center(image, bad_mask, ref_x, ref_y,
                                                            range_y), 3, np.median(col)])

    _scan(ref_x, nx, 1, p0)
    _scan(ref_x - 1, -1, -1, p0)

    return y_centers, y_errs, widths

def _seed_center(image, bad_mask, x, y_start, range_y, iterations=3) -> float:
    """Centroid of column x above its window minimum, recentring the window each time."""
    center = float(y_start)
    for _ in range(iterations):
        y0 = int(max(0, round(center) - range_y // 2))
        y1 = int(min(image.shape[0], y0 + range_y))
        valid = bad_mask[y0:y1, x] == 0
        if valid.sum() < 5:
            break
        z = image[y0:y1, x][valid]
        z = z - z.min()
        if not z.sum() > 0:
            break
        center = float(np.sum(np.arange(y0, y1)[valid] * z) / z.sum())
    return center


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



QC_FLUX = Keyword("QC.FLATFLUX.SC{0}", "Total flux per DIT, T{1}", "adu")


def qc(product: Flat) -> dict[Keyword, float]:
    """Total flux of each dark-subtracted mean flat per DIT, in T4,T3,T2,T1 order."""
    if product.telescope_flux_adu is None:
        return {}
    fluxes = zip(layout.telescopes, product.telescope_flux_adu)
    return {QC_FLUX.at(i, telescope): float(flux) for i, (telescope, flux) in enumerate(fluxes, 1)}


def to_fits_pixels(values):
    """Zero-based pixel coordinates -> FITS one-based; the -1 sentinel is kept."""
    return np.where(values < 0, values, values + 1)


def from_fits_pixels(values):
    """FITS one-based pixel coordinates -> zero-based; the -1 sentinel is kept."""
    return np.where(values < 0, values, values - 1)


def write_flat(path: Path, product: Flat, *, header=None) -> None:
    """Save a FLAT with its QC; pixel indices are one-based on disk. Never overwrite."""
    detector = products.detector_table()
    columns = [
        products.column("REGNAME", detector.data["REGNAME"]),
        products.column("CENTERY", to_fits_pixels(product.trace_y_pixels), "pixel"),
    ]
    if product.trace_x_range is not None:
        columns.append(products.column("XRANGE", to_fits_pixels(product.trace_x_range), "pixel"))
    if product.profile_sigma_pixels is not None:
        columns.append(products.column("WIDTH", product.profile_sigma_pixels, "pixel"))
    profile = [
        products.column("PIXEL_X", to_fits_pixels(product.profile_x_pixels), "pixel"),
        products.column("PIXEL_Y", to_fits_pixels(product.profile_y_pixels), "pixel"),
    ]
    extensions = [
        detector,
        products.image_hdu("PLANETES_FLAT_IMAGE", product.flat_map_adu, "adu"),
        afits.BinTableHDU.from_columns(columns, name="PROFILE_PARAMS"),
        afits.BinTableHDU.from_columns(profile, name="PLANETES_PROFILE"),
    ]
    if product.telescope_flux_adu is not None:
        extensions.append(products.image_hdu("PLANETES_FLAT_FLUX", product.telescope_flux_adu, "adu"))
    products.save_product(path, "FLAT", extensions, header=header, qc=qc(product),
                      detector=product.detector)


def read_flat(path: Path) -> Flat:
    """Read a checksummed FLAT at computation precision, zero-based pixel indices."""
    with products.open_product(path, "FLAT") as hdus:
        flat = Flat(
            flat_map_adu=products.read_image(hdus, "PLANETES_FLAT_IMAGE"),
            trace_y_pixels=from_fits_pixels(
                products.read_column(hdus, "PROFILE_PARAMS", "CENTERY")),
            profile_x_pixels=from_fits_pixels(
                products.read_column(hdus, "PLANETES_PROFILE", "PIXEL_X")),
            profile_y_pixels=from_fits_pixels(
                products.read_column(hdus, "PLANETES_PROFILE", "PIXEL_Y")),
            detector=detector_cards(hdus[0].header),
        )
        profile = hdus["PROFILE_PARAMS"].columns.names
        if "XRANGE" in profile:
            flat.trace_x_range = from_fits_pixels(
                products.read_column(hdus, "PROFILE_PARAMS", "XRANGE")).astype(int)
        if "WIDTH" in profile:
            flat.profile_sigma_pixels = products.read_column(hdus, "PROFILE_PARAMS", "WIDTH")
        if "PLANETES_FLAT_FLUX" in hdus:
            flat.telescope_flux_adu = products.read_image(hdus, "PLANETES_FLAT_FLUX")
        return flat


@step("FLAT")
def run(dark_file, bad_file, flat_files, *, output_dir,
        aperture_width_pixels=APERTURE_WIDTH_PIXELS, metrology_trace=False) -> dict[str, Path]:
    """planetes_flat: four FLAT_RAW (one telescope open each) + DARK + BAD -> FLAT."""
    dark, bad = read_dark(dark_file), read_bad(bad_file)
    flats = read_raws(flat_files, "FLAT_RAW", layout.telescopes)
    options = dict(aperture_width_pixels=aperture_width_pixels, metrology_trace=metrology_trace)
    product = make_flat(flats, dark, bad, **options)

    reference = flats["4"].path  # Names the product and supplies its observation header.
    inputs = {"FLAT_RAW": flat_files, "DARK": [dark_file], "BAD": [bad_file]}
    header = products.provenance("planetes_flat", reference, inputs, **options)
    record_saturation(header, {f"T{telescope}": raw for telescope, raw in flats.items()})
    path = products.product_path(output_dir, reference, "FLAT")
    write_flat(path, product, header=header)
    return {"FLAT": path}
