"""DARK/BAD step: raw-stack statistics, bad-pixel parameters and live dark loading."""

from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path

import numpy as np
from astropy.io import fits as afits
from numpy.typing import NDArray

from .io import keywords as kw
from .io.fits import read_product
from .io.hdu import as_compute_array
from .io.raw import iter_raw_cube, read_raw_metadata
from .products import (
    Bad,
    Dark,
)

BAD_PIXEL_THRESHOLD = 5.0


def make_dark_products(
    path: Path, *, bad_pixel_threshold: float = BAD_PIXEL_THRESHOLD
) -> tuple[Dark, Bad]:
    """Measure DARK mean/RMS and BAD flags from a full raw stack; saving is explicit."""

    metadata = read_raw_metadata(path, category="DARK_RAW")
    dark = measure_dark(iter_raw_cube(path))
    dark = replace(dark, setup=metadata.setup)
    return dark, detect_bad_pixels(dark, threshold=bad_pixel_threshold)


def frame_mean(chunks: Iterable[NDArray]) -> NDArray:
    """Compute a temporal mean from frame chunks."""
    total = None
    count = 0
    for chunk in chunks:
        if total is None:
            total = np.zeros(chunk.shape[1:])
        total += chunk.sum(axis=0)
        count += len(chunk)
    if total is None or count == 0:
        raise ValueError("Cube contains no frames")
    return total / count


def frame_statistics(chunks: Iterable[NDArray]) -> tuple[NDArray, NDArray, int]:
    """Merge chunk means and centered sums of squares; return population RMS."""
    mean = None
    m2 = None
    count = 0
    for chunk in chunks:
        n = len(chunk)
        if n == 0:
            continue
        chunk_mean = chunk.mean(axis=0)
        chunk_m2 = np.sum((chunk - chunk_mean) ** 2, axis=0)
        if mean is None:
            mean, m2 = chunk_mean, chunk_m2
        else:
            delta = chunk_mean - mean
            m2 += chunk_m2 + delta**2 * (count * n / (count + n))
            mean += delta * (n / (count + n))
        count += n
    if mean is None:
        raise ValueError("Cube contains no frames")
    return mean, np.sqrt(m2 / count), count


def detect_bad_pixels(dark: Dark, *, threshold: float = 5.0) -> Bad:
    """Build a bad-pixel map from precomputed temporal mean and RMS images."""
    if dark.temporal_rms_adu is None:
        raise ValueError("Bad-pixel detection requires measured temporal RMS")
    mean_img, std_img = dark.mean_adu, dark.temporal_rms_adu
    BAD_HOT, BAD_DEAD, BAD_NOISY = 1, 2, 4

    med_mean = np.median(mean_img)
    med_rms = np.median(std_img)

    range_max = med_mean + threshold * med_rms
    range_min = med_mean - threshold * med_rms
    std_max = med_rms * threshold

    bpm = np.zeros(mean_img.shape, dtype=np.int16)

    bpm[mean_img > range_max] |= BAD_HOT
    bpm[mean_img < range_min] |= BAD_DEAD
    bpm[std_img > std_max] |= BAD_NOISY

    return Bad(bpm)


def measure_dark(chunks: Iterable[NDArray]) -> Dark:
    """Measure the mean and, with at least two frames, temporal population RMS."""
    mean, rms, count = frame_statistics(chunks)
    return Dark(mean, rms if count >= 2 else None, count)


def load_dark(path: Path) -> Dark:
    """Load a DARK product, raw stack or mean-only image; unknown statistics stay None."""

    path = Path(path)
    with afits.open(path, memmap=True) as hdus:
        head = hdus[0].header
        if kw.PRO_CATEGORY in head:
            return read_product(path, Dark)
        role = str(head.get(kw.DPR_TYPE, "")).upper()
        if role and role != "DARK":
            raise ValueError("Expected dark input, FITS declares another raw category")
        index = next((i for i, h in enumerate(hdus) if h.data is not None), None)
        if index is None:
            raise ValueError("Dark FITS has no data")
        data = hdus[index].data
        unit = str(hdus[index].header.get("BUNIT", "adu")).lower()
        if unit != "adu":
            raise ValueError("Dark subtraction expects ADU")
        if data.ndim == 3:
            metadata = read_raw_metadata(path, category="DARK_RAW")
            result = measure_dark(iter_raw_cube(path))
            return replace(result, setup=metadata.setup)
        if data.ndim != 2:
            raise ValueError("A GUI dark must be a 2-D image or nonempty cube")
        result = Dark(as_compute_array(data).copy())
    return result
