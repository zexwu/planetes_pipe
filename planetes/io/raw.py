"""Raw FITS streaming, known bench metadata and calibration compatibility."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits as afits

from ..instrument import DetectorSetup, check_setup
from . import keywords as kw
from .hdu import as_compute_array

CHUNK_FRAMES = 16


@dataclass(frozen=True, slots=True)
class RawMetadata:
    """Only the raw identity and physical setup needed by the pipeline."""

    category: str
    shutters: tuple[bool | None, bool | None, bool | None, bool | None]
    setup: DetectorSetup


def _detector_hdu(hdus, hdu: int | str | None = None):
    """Select the named/unique image cube; reject ambiguous data and known unit errors."""
    if hdu is not None:
        image = hdus[hdu]
    elif kw.RAW_IMAGE in hdus:
        image = hdus[kw.RAW_IMAGE]
    else:
        cubes = [
            item
            for item in hdus
            if isinstance(item, (afits.PrimaryHDU, afits.ImageHDU, afits.CompImageHDU))
            and len(item.shape) == 3
        ]
        if len(cubes) > 1:
            raise ValueError("Multiple raw detector cubes; select an explicit hdu")
        if not cubes:
            raise ValueError("Expected a nonempty three-dimensional raw detector cube")
        image = cubes[0]
    if not isinstance(
        image, (afits.PrimaryHDU, afits.ImageHDU, afits.CompImageHDU)
    ) or (len(image.shape) != 3 or min(image.shape) < 1):
        raise ValueError("Expected a nonempty three-dimensional raw detector cube")
    for header in (hdus[0].header, image.header):
        if "BUNIT" in header and str(header["BUNIT"]).strip().lower() != "adu":
            raise ValueError("Raw detector values must be ADU")
    return image


def read_raw_header(path: Path) -> afits.Header:
    """Return the unchanged primary header for inspecting provisional keyword mappings."""
    with afits.open(path, memmap=True) as hdus:
        return hdus[0].header.copy()


def parse_raw_header(header: afits.Header, shape, *, category=None) -> RawMetadata:
    """Read recognized bench fields; missing metadata stays unknown, not inferred."""
    dpr_type = str(header.get(kw.DPR_TYPE, "")).strip().upper()
    declared_category = None
    for raw_category, definition in kw.RAW_TYPES.items():
        if definition["dpr_type"] == dpr_type:
            declared_category = raw_category
            break
    category = declared_category if category is None else category
    if category not in kw.RAW_TYPES:
        raise ValueError("Raw category is unknown; supply an explicit category")
    if declared_category is not None and declared_category != category:
        raise ValueError(
            f"Expected raw category {category}, header declares {dpr_type}"
        )
    if len(shape) != 3 or min(shape) < 1:
        raise ValueError("Expected a nonempty three-dimensional raw detector cube")

    shutter_values = tuple(header.get(key) for key in kw.SHUTTER_KEYS)
    if any(value is not None and type(value) is not bool for value in shutter_values):
        raise ValueError("Recorded shutter states must be boolean")
    known_open = sum(value is True for value in shutter_values)
    unknown = shutter_values.count(None)
    expected_open = kw.RAW_TYPES[category]["open_shutters"]
    if not known_open <= expected_open <= known_open + unknown:
        raise ValueError("Raw category and recorded shutter states conflict")

    roi = {name: header.get(key) for name, key in kw.ROI_KEYS.items()}
    for name in ("origin_x", "origin_y"):
        if roi[name] is not None:
            roi[name] -= 1  # FITS origins are one-based.
    for name, size in (("height", shape[1]), ("width", shape[2])):
        if roi[name] is not None and roi[name] != size:
            raise ValueError(f"Raw {name} disagrees with cube dimensions")
        roi[name] = size
    exposure = header.get(kw.DIT, header.get(kw.UIT))
    if exposure is not None and (not np.isfinite(exposure) or exposure <= 0):
        raise ValueError("Exposure must be positive finite seconds")
    if any(
        roi[name] is not None and roi[name] < 0 for name in ("origin_x", "origin_y")
    ) or any(
        roi[name] is not None and roi[name] < 1 for name in ("binning_x", "binning_y")
    ):
        raise ValueError("Invalid detector ROI origin or binning")
    raw = RawMetadata(
        category=category,
        shutters=shutter_values,
        setup=DetectorSetup(
            **roi, exposure_s=exposure, detector=header.get(kw.DETECTOR_ID)
        ),
    )
    return raw


def read_raw_metadata(
    path: Path, *, category=None, hdu: int | str | None = None
) -> RawMetadata:
    """Read physical fields from the FITS header and image dimensions."""
    with afits.open(path, memmap=True) as hdus:
        header = hdus[0].header
        if kw.PRO_CATEGORY in header:
            raise ValueError(f"Expected raw FITS, found pipeline product: {path}")
        image = _detector_hdu(hdus, hdu)
        return parse_raw_header(header, image.shape, category=category)


def iter_raw_cube(path: Path, *, hdu: int | str | None = None):
    """Read at most 16 frames and adapt them to computation precision once."""
    with afits.open(path, memmap=True) as hdus:
        data = _detector_hdu(hdus, hdu).data
        for start in range(0, len(data), CHUNK_FRAMES):
            yield as_compute_array(data[start : start + CHUNK_FRAMES])


def check_raw_calibrations(raw: RawMetadata, dark, flat=None):
    """Check known DIT/ROI conflicts against the selected DARK and FLAT."""
    check_setup(raw.setup, dark.setup, exposure=True, role="DARK")
    if flat is not None:
        check_setup(raw.setup, flat.setup, exposure=False, role="FLAT")
