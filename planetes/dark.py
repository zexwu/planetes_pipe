"""DARK/BAD step: raw-stack statistics, bad-pixel parameters and live dark loading."""

from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path

import numpy as np
from astropy.io import fits as afits
from numpy.typing import NDArray

from .io import keywords as kw
from .io.fits import read_product
from .io.raw import iter_raw_cube, read_raw_metadata
from .products import (
    BadPixelMap,
    Dark,
    DarkInput,
    DarkImage,
)

BAD_PIXEL_THRESHOLD = 5.0


def make_dark_products(
    path: Path, *, bad_pixel_threshold: float = BAD_PIXEL_THRESHOLD
) -> tuple[Dark, BadPixelMap]:
    """Measure DARK mean/RMS and BAD flags from a full raw stack; saving is explicit."""

    metadata = read_raw_metadata(path, category="DARK_RAW")
    dark = measure_dark(iter_raw_cube(path))
    dark = replace(dark, setup=metadata.setup)
    return dark, detect_bad_pixels(dark, threshold=bad_pixel_threshold)


def frame_mean(
    chunks: Iterable[NDArray[np.float32]],
) -> NDArray[np.float32]:
    """Compute a temporal mean from frame chunks."""
    total = None
    count = 0
    for chunk in chunks:
        if total is None:
            total = np.zeros(chunk.shape[1:], dtype=np.float64)
        total += chunk.sum(axis=0, dtype=np.float64)
        count += len(chunk)
    if total is None or count == 0:
        raise ValueError("Cube contains no frames")
    return (total / count).astype(np.float32)


def frame_statistics(
    chunks: Iterable[NDArray[np.float32]],
) -> tuple[NDArray[np.float32], NDArray[np.float32], int]:
    """Compute temporal mean and population RMS from frame chunks."""
    total = None
    total_squared = None
    count = 0
    for chunk in chunks:
        if total is None:
            total = np.zeros(chunk.shape[1:], dtype=np.float64)
            total_squared = np.zeros(chunk.shape[1:], dtype=np.float64)
        total += chunk.sum(axis=0, dtype=np.float64)
        total_squared += np.square(chunk, dtype=np.float64).sum(
            axis=0, dtype=np.float64
        )
        count += len(chunk)
    if total is None or total_squared is None or count == 0:
        raise ValueError("Cube contains no frames")

    mean = total / count
    variance = np.maximum(total_squared / count - mean**2, 0.0)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32), count


def detect_bad_pixels(dark: Dark, *, threshold: float = 5.0) -> BadPixelMap:
    """Build a bad-pixel map from precomputed temporal mean and RMS images."""
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

    return BadPixelMap(bpm)


def measure_dark(chunks: Iterable[NDArray[np.float32]]) -> Dark:
    """Measure temporal mean and population RMS from a complete stack."""
    mean, rms, count = frame_statistics(chunks)
    if count < 2:
        raise ValueError(
            "Complete dark calibration needs at least two frames; use DarkImage for subtraction only"
        )
    return Dark(mean, rms, count)


def load_dark(path: Path) -> DarkInput:
    """GUI loader: complete new DARK, or explicitly selected raw/image subtraction data."""

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
        if data.ndim == 3 and len(data) >= 2:
            metadata = read_raw_metadata(path, category="DARK_RAW")
            result = measure_dark(iter_raw_cube(path))
            return replace(result, setup=metadata.setup)
        if data.ndim == 3 and len(data) == 1:
            data = data[0]
        if data.ndim != 2:
            raise ValueError("A GUI dark must be a 2-D image or nonempty cube")
        result = DarkImage(np.array(data, dtype=np.float32, copy=True))
    return result
