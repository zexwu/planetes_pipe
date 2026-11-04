"""DARK and BAD: detector dark statistics and bad pixels from one DARK_RAW stack.

    from planetes_pipe.steps import dark
    out = dark.run("dark.fits", output_dir="calibration")  # {"DARK": path, "BAD": path}
    master = dark.read_dark(out["DARK"])
"""

from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from astropy.io.fits import Header
from numpy.typing import NDArray

from planetes_pipe.instrument import DetectorSetup
from planetes_pipe.io import products
from planetes_pipe.io.keywords import NFRAMES, Keyword
from planetes_pipe.io.raw import Raw, read_raw, read_setup


@dataclass(slots=True)
class Dark:
    """Dark image in ADU; temporal statistics are None when not measured."""

    mean_adu: NDArray  # (y, x)
    temporal_rms_adu: NDArray | None = None  # (y, x)
    n_frames: int | None = None
    setup: DetectorSetup | None = None


@dataclass(slots=True)
class Bad:
    """Bad-pixel flags: hot=1, dead=2, noisy=4."""

    bad_pixel_flags: NDArray  # (y, x)


BAD_PIXEL_THRESHOLD = 5.0


def make_dark_products(
    raw: Raw,
    *,
    bad_pixel_threshold: float = BAD_PIXEL_THRESHOLD,
) -> tuple[Dark, Bad]:
    """Measure a raw dark stack and identify hot, dead and noisy pixels.

    Parameters
    ----------
    raw : Raw
        Raw dark cube and its recorded setup, from products.read_raw.
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

    dark = measure_dark(raw.chunks())
    dark = replace(dark, setup=raw.setup)
    return dark, detect_bad_pixels(dark, threshold=bad_pixel_threshold)


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
        make_dark_products adds the detector setup of the raw cube.
    """
    mean, rms, count = frame_statistics(chunks)
    return Dark(mean, rms if count >= 2 else None, count)


QC_DARK_MEDIAN = Keyword("QC.MEDIANDARK.SC", "Spatial median of the temporal mean", "adu")
QC_DARK_RMS = Keyword("QC.DARKRMS.SC", "Spatial median of the temporal RMS", "adu")
QC_BAD = Keyword("QC.BADPIX.SC", "Number of flagged pixels")
QC_BAD_DARK = Keyword("QC.BADPIX_DARK.SC", "Number of hot or dead pixels")
QC_BAD_NOISY = Keyword("QC.BADPIX_RMS.SC", "Number of noisy pixels")


def qc(product: Dark | Bad) -> dict[Keyword, float | None]:
    """DARK: medians of the temporal mean and RMS. BAD: flagged pixel counts."""
    if isinstance(product, Dark):
        rms = product.temporal_rms_adu
        return {
            QC_DARK_MEDIAN: float(np.median(product.mean_adu)),
            QC_DARK_RMS: float(np.median(rms)) if rms is not None else None,
        }
    flags = product.bad_pixel_flags
    return {
        QC_BAD: int(np.count_nonzero(flags)),
        QC_BAD_DARK: int(np.count_nonzero(flags & 3)),
        QC_BAD_NOISY: int(np.count_nonzero(flags & 4)),
    }


def write_dark(path: Path, product: Dark, *, header=None) -> None:
    """Save a DARK (mean and temporal RMS, ADU) with its QC; never overwrite."""
    if (
        product.temporal_rms_adu is None
        or product.n_frames is None
        or product.n_frames < 2
        or product.mean_adu.shape != product.temporal_rms_adu.shape
    ):
        raise ValueError("Saved DARK requires measured RMS and at least two frames")
    header = Header() if header is None else header.copy()
    NFRAMES.set(header, product.n_frames)
    products.save_product(path, "DARK", [
        products.image_hdu("IMAGING_DATA_SC", product.mean_adu, "adu"),
        products.image_hdu("IMAGING_ERR_SC", product.temporal_rms_adu, "adu"),
    ], header=header, qc=qc(product), setup=product.setup)


def read_dark(path: Path) -> Dark:
    """Read a checksummed DARK at computation precision."""
    with products.open_product(path, "DARK") as hdus:
        return Dark(
            mean_adu=products.read_image(hdus, "IMAGING_DATA_SC"),
            temporal_rms_adu=products.read_image(hdus, "IMAGING_ERR_SC"),
            n_frames=NFRAMES.get(hdus[0].header),
            setup=read_setup(hdus[0].header),
        )


def write_bad(path: Path, product: Bad, *, header=None) -> None:
    """Save a BAD pixel map (hot=1, dead=2, noisy=4) with its QC; never overwrite."""
    products.save_product(path, "BAD", [products.image_hdu("IMAGING_DATA_SC", product.bad_pixel_flags)],
                      header=header, qc=qc(product))


def read_bad(path: Path) -> Bad:
    """Read a checksummed BAD pixel map."""
    with products.open_product(path, "BAD") as hdus:
        return Bad(products.read_image(hdus, "IMAGING_DATA_SC"))


def run(dark_file, *, output_dir, bad_pixel_threshold=BAD_PIXEL_THRESHOLD) -> dict[str, Path]:
    """planetes_dark: one DARK_RAW -> DARK and BAD."""
    raw = read_raw(dark_file, "DARK_RAW")
    dark, bad = make_dark_products(raw, bad_pixel_threshold=bad_pixel_threshold)

    header = products.provenance("planetes_dark", dark_file, {"DARK_RAW": [dark_file]},
                              bad_pixel_threshold=bad_pixel_threshold)
    dark_path = products.product_path(output_dir, dark_file, "DARK")
    bad_path = products.product_path(output_dir, dark_file, "BAD")
    write_dark(dark_path, dark, header=header)
    write_bad(bad_path, bad, header=header)
    return {"DARK": dark_path, "BAD": bad_path}
