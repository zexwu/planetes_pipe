"""Temporal detector statistics and explicit full versus subtraction-only darks."""

from collections.abc import Iterable
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from astropy.io import fits as afits

from .fits import iter_raw_cube, read_product
from .models import DarkCalibration, DarkImage, Provenance, positive, validate

BAD_THRESHOLD = 5.0


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


def compute_dark(chunks, *, bad_threshold: float = BAD_THRESHOLD) -> DarkCalibration:
    """Measure the retained population RMS and bad-pixel flags from a complete stack."""
    bad_threshold = positive(bad_threshold, "bad pixel threshold")
    mean, rms, count = frame_statistics(chunks)
    if count < 2:
        raise ValueError(
            "Complete dark calibration needs at least two frames; use DarkImage for subtraction only"
        )
    result = DarkCalibration(
        mean,
        rms,
        compute_bad_map_from_stats(mean, rms, bad_threshold),
        count,
        provenance=Provenance(
            producer="planetes.dark", parameters={"bad_threshold": bad_threshold}
        ),
    )
    validate(result)
    return result


def calibrate_dark_file(
    path: Path, *, bad_threshold: float = BAD_THRESHOLD, hdu: int = 1
) -> DarkCalibration:
    """Read a raw stack and return a complete calibration, without writing files."""
    from dataclasses import replace

    result = compute_dark(iter_raw_cube(path, hdu=hdu), bad_threshold=bad_threshold)
    return replace(
        result,
        provenance=replace(
            result.provenance, inputs={"dark_raw": str(Path(path).absolute())}
        ),
    )


def load_dark(path: Path, *, bad_threshold: float = BAD_THRESHOLD):
    """Load a typed dark, calibrate a raw stack, or explicitly read a subtraction-only image."""
    path = Path(path)
    with afits.open(path, memmap=True) as hdus:
        head = hdus[0].header
        if "PLTSCHEM" in head or "ESO PRO CATG" in head:
            result = read_product(path)
            if not isinstance(result, (DarkCalibration, DarkImage)):
                raise ValueError(f"Expected dark product, got {type(result).__name__}")
            return result
        role = str(head.get("ESO DPR TYPE", "")).upper()
        category = str(head.get("ESO DPR CATG", "")).upper()
        if (
            role and "DARK" not in role and "BIAS" not in role
        ) or category == "SCIENCE":
            raise ValueError(f"FITS declares a non-dark scientific role: {path}")
        index = next((i for i, h in enumerate(hdus) if h.data is not None), None)
        if index is None:
            raise ValueError(f"Dark FITS has no data: {path}")
        data = hdus[index].data
        unit = str(hdus[index].header.get("BUNIT", head.get("BUNIT", "adu"))).lower()
        if unit != "adu":
            raise ValueError(f"Dark subtraction expects ADU, got {unit}")
        if data.ndim == 3 and len(data) >= 2:
            return calibrate_dark_file(path, bad_threshold=bad_threshold, hdu=index)
        if data.ndim == 3 and len(data) == 1:
            data = data[0]
        if data.ndim != 2:
            raise ValueError(
                "A subtraction dark must be a 2-D image or nonempty 3-D cube"
            )
        result = DarkImage(
            np.array(data, dtype=np.float32, copy=True),
            provenance=Provenance(
                producer="planetes.dark.load_image",
                inputs={"dark_image": str(path.absolute())},
            ),
        )
    validate(result)
    return result
