"""FLAT: spectral trace positions, extraction apertures and detector response.

    from planetes_pipe import flat
    out = flat.run(dark_path, bad_path, flat_raws, output_dir="calibration")  # {"FLAT": path}
"""

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import median_filter
from scipy.signal import find_peaks, medfilt

from planetes_pipe import files, log
from planetes_pipe.dark import Bad, Dark, read_bad, read_dark
from planetes_pipe.instrument import (DetectorSetup, Raw, check_setup, layout, order_raws,
                                      shutter_identity)
from planetes_pipe.kernels import fit_gaussian

@dataclass(slots=True)
class Flat:
    """Detector flat and sparse extraction geometry; indices are zero-based."""

    profile_x_pixels: NDArray  # (output, sample), -1 sentinel
    profile_y_pixels: NDArray  # (output, sample), -1 sentinel
    flat_map_adu: NDArray  # (y, x)
    trace_y_pixels: NDArray  # (output, x)
    profile_sigma_pixels: NDArray | None = None  # (output, x)
    telescope_flux_adu: NDArray | None = None  # (telescope,)
    setup: DetectorSetup | None = None
    trace_x_range: NDArray | None = None  # (output, 2), inclusive fitted limits, zero-based


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
    telescope_flats: list[Raw],
    dark: Dark,
    bad: Bad,
    *,
    aperture_width_pixels: float = APERTURE_WIDTH_PIXELS,
    metrology_trace: bool = False,
) -> Flat:
    """Subtract the selected dark and calibrate the detector extraction geometry.

    Parameters
    ----------
    telescope_flats : list of Raw
        Four raw ADU cubes in any order; telescope identity comes from shutters.
    dark : Dark
        Selected mean dark in ADU on exactly the raw detector grid. Known DIT,
        ROI and detector settings must match; dimensions are checked even when
        setup is unknown. No dark scaling or broadcasting is allowed.
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
        Gaussian sigmas, per-telescope total fluxes and detector setup.
        Not written to disk; run() does that.

    Notes
    -----
    Each file is averaged over all its frames, then the four dark-subtracted
    images receive equal weight. Physical outputs are ordered by increasing y.
    """

    maps = []
    setup = None
    for raw in order_raws(telescope_flats, layout.telescopes):
        check_setup(raw.setup, dark.setup, exposure=True, role="DARK")
        if setup is None:
            setup = raw.setup
        check_setup(raw.setup, setup, exposure=False, role="flat sequence")
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
        result = replace(result, profile_x_pixels=result.profile_x_pixels[keep],
                         profile_y_pixels=result.profile_y_pixels[keep],
                         trace_y_pixels=result.trace_y_pixels[keep],
                         profile_sigma_pixels=result.profile_sigma_pixels[keep],
                         trace_x_range=result.trace_x_range[keep])
    return replace(result, setup=setup)


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
        entries mark bad pixels or padding. Detector setup is left unset.

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
    log.debug(f"Found {len(ref_ys)} Peaks with THRESH {thresh:.0f}...")
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

    log.info("Building sparse extraction profiles...")
    for i, ref_y in enumerate(ref_ys):
        log.debug(f"    Tracing spectrum {i + 1}/{n_outputs} at Y={ref_y}...")
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
        log.info(
            f"    Trace {i}: x={fit_x[0]}..{fit_x[-1]}, "
            f"{len(fit_x)} fitted columns; seed x={ref_x}"
        )
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


def qc(product: Flat) -> dict:
    """Mean-frame total flux per telescope in ADU/DIT, T4,T3,T2,T1."""
    if product.telescope_flux_adu is None:
        return {}
    return {f"ESO QC FLATFLUX SC{i}": float(flux)
            for i, flux in enumerate(product.telescope_flux_adu, 1)}


def write_flat(path: Path, product: Flat, *, header=None) -> None:
    """Save a FLAT with its QC; pixel indices are one-based on disk. Never overwrite."""
    detector = files.detector_table()
    columns = [
        files.column("REGNAME", detector.data["REGNAME"]),
        files.column("CENTERY", files.one_based_pixels(product.trace_y_pixels), "pixel"),
    ]
    if product.trace_x_range is not None:
        columns.append(files.column("XRANGE", files.one_based_pixels(product.trace_x_range), "pixel"))
    if product.profile_sigma_pixels is not None:
        columns.append(files.column("WIDTH", product.profile_sigma_pixels, "pixel"))
    extensions = [
        detector,
        files.image_hdu("PLANETES_FLAT_IMAGE", product.flat_map_adu, "adu"),
        files.table("PROFILE_PARAMS", columns),
        files.table("PLANETES_PROFILE", [
            files.column("PIXEL_X", files.one_based_pixels(product.profile_x_pixels), "pixel"),
            files.column("PIXEL_Y", files.one_based_pixels(product.profile_y_pixels), "pixel"),
        ]),
    ]
    if product.telescope_flux_adu is not None:
        extensions.append(files.image_hdu("PLANETES_FLAT_FLUX", product.telescope_flux_adu, "adu"))
    files.save_product(path, "FLAT", extensions, header=header, qc=qc(product),
                      setup=product.setup)


def read_flat(path: Path) -> Flat:
    """Read a checksummed FLAT at computation precision, zero-based pixel indices."""
    with files.open_product(path, "FLAT") as hdus:
        profile = hdus["PROFILE_PARAMS"].columns.names
        return Flat(
            flat_map_adu=files.read_image(hdus, "PLANETES_FLAT_IMAGE"),
            trace_y_pixels=files.zero_based_pixels(
                files.read_column(hdus, "PROFILE_PARAMS", "CENTERY")),
            profile_x_pixels=files.zero_based_pixels(
                files.read_column(hdus, "PLANETES_PROFILE", "PIXEL_X")),
            profile_y_pixels=files.zero_based_pixels(
                files.read_column(hdus, "PLANETES_PROFILE", "PIXEL_Y")),
            trace_x_range=files.zero_based_pixels(
                files.read_column(hdus, "PROFILE_PARAMS", "XRANGE")).astype(int)
            if "XRANGE" in profile else None,
            profile_sigma_pixels=files.read_column(hdus, "PROFILE_PARAMS", "WIDTH")
            if "WIDTH" in profile else None,
            telescope_flux_adu=files.read_image(hdus, "PLANETES_FLAT_FLUX")
            if "PLANETES_FLAT_FLUX" in hdus else None,
            setup=files.read_setup(hdus[0].header),
        )


def run(dark, bad, flat_raws, *, output_dir,
        aperture_width_pixels=APERTURE_WIDTH_PIXELS, metrology_trace=False) -> dict[str, Path]:
    """Recipe planetes_flat: four FLAT_RAW (one per open shutter) + DARK + BAD -> FLAT."""
    raws = [files.read_raw(path, "FLAT_RAW") for path in flat_raws]
    product = make_flat(raws, read_dark(dark), read_bad(bad),
                        aperture_width_pixels=aperture_width_pixels,
                        metrology_trace=metrology_trace)
    # Name and inherit the header from the T4 flat, independent of input order.
    reference = next(p for p, raw in zip(flat_raws, raws) if shutter_identity(raw.shutters) == "4")
    header = files.provenance("planetes_flat", reference,
                             {"FLAT_RAW": flat_raws, "DARK": [dark], "BAD": [bad]},
                             aperture_width_pixels=aperture_width_pixels,
                             metrology_trace=metrology_trace)
    out = files.output_paths(output_dir, reference, "FLAT")
    write_flat(out["FLAT"], product, header=header)
    return out
