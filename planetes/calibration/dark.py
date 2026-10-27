"""Temporal detector statistics and explicit full versus subtraction-only darks."""

from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray

from ..identity import fingerprint
from ..products import BadPixelMap, DarkCalibration, Provenance, positive, validate

BAD_PIXEL_THRESHOLD = 5.0


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


def compute_badpix(dark: DarkCalibration, *, threshold: float = 5.0) -> BadPixelMap:
    """Build a bad-pixel map from precomputed temporal mean and RMS images."""
    validate(dark)
    threshold = positive(threshold, "bad pixel threshold")
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

    return BadPixelMap(
        bpm,
        provenance=Provenance(
            producer="planetes.dark.compute_badpix",
            parameters={"bad_threshold": threshold},
            dependencies={"dark": fingerprint(dark)},
        ),
    )


def compute_dark(chunks) -> DarkCalibration:
    """Measure temporal mean and population RMS from a complete stack."""
    mean, rms, count = frame_statistics(chunks)
    if count < 2:
        raise ValueError(
            "Complete dark calibration needs at least two frames; use DarkImage for subtraction only"
        )
    result = DarkCalibration(
        mean,
        rms,
        count,
        provenance=Provenance(producer="planetes.dark"),
    )
    validate(result)
    return result
