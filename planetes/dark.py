"""DARK/BAD step: raw-stack statistics and bad-pixel parameters."""

from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .io.keywords import RAW_IMAGE
from .io.raw import iter_raw_cube, read_raw_metadata
from .products import Bad, Dark

BAD_PIXEL_THRESHOLD = 5.0


def make_dark_products(
    path: Path,
    *,
    hdu: int | str = RAW_IMAGE,
    bad_pixel_threshold: float = BAD_PIXEL_THRESHOLD,
) -> tuple[Dark, Bad]:
    """Measure a raw dark stack and identify hot, dead and noisy pixels.

    Parameters
    ----------
    path : Path
        Raw ADU cube, shaped (frame, y, x), with at least two frames.
    hdu : int or str, optional
        Image HDU to read; use 0 explicitly for primary-HDU bench cubes.
    bad_pixel_threshold : float, optional
        Multiplier of the spatial median temporal RMS for BAD classification.
        Default is 5; this is not an independently measured read-noise threshold.

    Returns
    -------
    dark : Dark
        Temporal mean and population RMS in ADU, frame count and detector setup.
    bad : Bad
        Detector bit mask: hot=1, dead=2, noisy=4. Bits can be combined.

    Notes
    -----
    All frames contribute with equal weight. No clipping, saving or raw-file
    modification is performed. A one-frame stack cannot supply the RMS needed
    for BAD detection and raises ValueError.
    """

    metadata = read_raw_metadata(path, category="DARK_RAW", hdu=hdu)
    dark = measure_dark(iter_raw_cube(path, hdu=hdu))
    dark = replace(dark, setup=metadata.setup)
    return dark, detect_bad_pixels(dark, threshold=bad_pixel_threshold)


def frame_mean(chunks: Iterable[NDArray]) -> NDArray:
    """Compute the equally weighted temporal mean of a sequence of frame chunks.

    Parameters
    ----------
    chunks : iterable of ndarray
        Nonempty sequence of (frame, y, x) arrays in ADU, all on the same detector
        grid. Supply float64 calculation arrays; raw readers do this conversion.

    Returns
    -------
    mean_adu : ndarray, shape (y, x)
        Mean over all frames, including a shorter final chunk.
    """
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
    """Measure temporal mean and population RMS without large-moment subtraction.

    Parameters
    ----------
    chunks : iterable of ndarray
        Float64 ADU arrays shaped (frame, y, x) on one detector grid.
        Empty chunks are skipped; at least one frame is required overall.

    Returns
    -------
    mean_adu : ndarray, shape (y, x)
        Equally weighted mean over all frames.
    rms_adu : ndarray, shape (y, x)
        sqrt(sum of squared deviations / n_frames), without Bessel correction.
    n_frames : int
        Total number of frames contributing to both statistics.

    Notes
    -----
    Chunk means and centered sums of squares are merged with Chan's formula.
    This avoids loss of precision when a large detector offset has small noise.
    The numerical RMS of one frame is zero; measure_dark marks it as unmeasured.
    """
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
    """Classify detector pixels using a measured dark mean and temporal RMS.

    Parameters
    ----------
    dark : Dark
        Mean and measured temporal population RMS images in ADU.
        A mean-only dark is insufficient and raises ValueError.
    threshold : float, optional
        Multiplier of median RMS, applied to mean-image deviations and RMS.

    Returns
    -------
    bad : Bad
        Integer (y, x) flags. Bits 1/2 mark means above/below median mean plus/
        minus threshold * median RMS. Bit 4 marks RMS above threshold * median RMS.

    Notes
    -----
    The temporal RMS includes all variability in the stack. It is not a separate
    read-noise estimate. Flags describe pixels; this function does not alter data.
    """
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
    """Measure a dark stack, keeping unmeasured temporal noise explicitly unknown.

    Parameters
    ----------
    chunks : iterable of ndarray
        Float64 ADU arrays shaped (frame, y, x) on one detector grid.

    Returns
    -------
    dark : Dark
        Temporal mean in ADU and measured frame count. temporal_rms_adu is the
        population RMS for two or more frames, and None for a single frame.
        Detector setup is left unset; the file entry point supplies it.
    """
    mean, rms, count = frame_statistics(chunks)
    return Dark(mean, rms if count >= 2 else None, count)
