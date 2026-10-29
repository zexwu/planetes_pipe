"""Read raw FITS headers and stream detector frames from an explicit image HDU."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits as afits

from ..instrument import DetectorSetup
from . import keywords as kw
from .hdu import as_compute_array

CHUNK_FRAMES = 16


@dataclass(frozen=True, slots=True)
class RawMetadata:
    """Raw category, physical shutter states and detector settings."""

    category: str
    shutters: tuple[bool | None, bool | None, bool | None, bool | None]
    setup: DetectorSetup


def _detector_hdu(hdus, hdu: int | str):
    """Read the selected cube and reject known unit or dimension mismatches."""
    image = hdus[hdu]
    if len(image.shape) != 3 or min(image.shape) < 1:
        raise ValueError("Expected a nonempty three-dimensional raw detector cube")
    for header in (hdus[0].header, image.header):
        if "BUNIT" in header and str(header["BUNIT"]).strip().lower() != "adu":
            raise ValueError("Raw detector values must be ADU")
    return image


def read_raw_metadata(
    path: Path, *, category: str, hdu: int | str = kw.RAW_IMAGE
) -> RawMetadata:
    """Read physical detector and shutter metadata for an explicitly classified raw file.

    Parameters
    ----------
    path : Path
        Raw FITS file containing a nonempty (frame, y, x) detector cube.
    category : str
        Expected role: DARK_RAW, FLAT_RAW, WAVELAMP_RAW, P2VM_RAW or SINGLE_SCI_RAW.
        A declared DPR TYPE and known shutter states must agree with this role.
    hdu : int or str, optional
        Selected image HDU; default IMAGING_DATA_SC. No automatic HDU search.

    Returns
    -------
    metadata : RawMetadata
        Category, shutter states in physical T1..T4 order, and DetectorSetup.
        Missing physical fields remain None; width/height come from cube dimensions.
        FITS one-based ROI origins are converted to zero-based Python coordinates.

    Notes
    -----
    Known non-ADU units are rejected. Missing BUNIT follows the bench ADU
    convention. Missing exposure/shutter metadata is not evidence of a match;
    callers choose calibrations and explicit file identities. Raw data are read only.
    """
    definition = kw.RAW_TYPES[category]
    with afits.open(path, memmap=True) as hdus:
        header = hdus[0].header
        if kw.PRO_CATEGORY in header:
            raise ValueError(f"Expected raw FITS, found pipeline product: {path}")
        image = _detector_hdu(hdus, hdu)
        declared = str(header.get(kw.DPR_TYPE, "")).strip().upper()
        if declared and declared != definition["dpr_type"]:
            raise ValueError(f"Expected {category}, header declares {declared}")

        shutters = tuple(header.get(key) for key in kw.SHUTTER_KEYS)
        if any(state is not None and type(state) is not bool for state in shutters):
            raise ValueError("Recorded shutter states must be boolean")
        known_open = sum(state is True for state in shutters)
        if (
            not known_open
            <= definition["open_shutters"]
            <= known_open + shutters.count(None)
        ):
            raise ValueError("Raw category and recorded shutter states conflict")

        roi = {name: header.get(key) for name, key in kw.ROI_KEYS.items()}
        for name in ("origin_x", "origin_y"):
            if roi[name] is not None:
                roi[name] -= 1  # FITS origins are one-based.
        for name, size in (("height", image.shape[1]), ("width", image.shape[2])):
            if roi[name] is not None and roi[name] != size:
                raise ValueError(f"Raw {name} disagrees with cube dimensions")
            roi[name] = size
        if any(
            roi[name] is not None and roi[name] < 0 for name in ("origin_x", "origin_y")
        ) or any(
            roi[name] is not None and roi[name] < 1
            for name in ("binning_x", "binning_y")
        ):
            raise ValueError("Invalid detector ROI origin or binning")
        exposure = header.get(kw.DIT, header.get(kw.UIT))
        if exposure is not None and (not np.isfinite(exposure) or exposure <= 0):
            raise ValueError("Exposure must be positive finite seconds")
        setup = DetectorSetup(
            **roi, exposure_s=exposure, detector=header.get(kw.DETECTOR_ID)
        )
        return RawMetadata(category, shutters, setup)


def iter_raw_cube(path: Path, *, hdu: int | str = kw.RAW_IMAGE):
    """Yield detector frames from one explicitly selected raw FITS image HDU.

    Parameters
    ----------
    path : Path
        Raw FITS cube with nonempty (frame, y, x) dimensions, in ADU.
    hdu : int or str, optional
        Image HDU name/index; default IMAGING_DATA_SC, or 0 for primary-HDU data.

    Yields
    ------
    chunk : ndarray, shape (up to 16 frames, y, x)
        Native float64 detector samples in original frame order. All frames,
        including the final partial chunk, are yielded without dark subtraction.

    Notes
    -----
    The FITS handle stays open during iteration. This checks image dimensions
    and declared units; physical role/setup checking uses read_raw_metadata.
    """
    with afits.open(path, memmap=True) as hdus:
        data = _detector_hdu(hdus, hdu).data
        for start in range(0, len(data), CHUNK_FRAMES):
            yield as_compute_array(data[start : start + CHUNK_FRAMES])
