"""Raw FITS streaming, known bench metadata and calibration compatibility."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits as afits

from ..instrument import DetectorSetup, check_setup
from . import keywords as kw
from .hdu import _read_json, as_compute_array

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
            **roi,
            exposure_s=exposure,
            exposure_source="readback"
            if kw.DIT in header
            else ("command" if kw.UIT in header else "unknown"),
            detector=header.get("ESO DET ID"),
            legacy=True,
        ),
    )
    return raw


def read_raw_metadata(
    path: Path, *, category=None, hdu: int | str | None = None
) -> RawMetadata:
    """Use known physical header values, filling gaps from saved format-3 JSON.

    Saved JSON also retains exposure-source and legacy flags. Contradictions in
    known physical fields fail; acquisition timestamps are not read here.
    """
    with afits.open(path, memmap=True) as hdus:
        header = hdus[0].header
        if kw.PRO_CATEGORY in header:
            raise ValueError(f"Expected raw FITS, found pipeline product: {path}")
        if kw.FORMAT_KEY in header and (
            header[kw.FORMAT_KEY] != kw.RAW_FORMAT
            or header.get("INSTRUME") != "PLANETES"
        ):
            raise ValueError(f"Unsupported PLANETES raw format: {path}")
        image = _detector_hdu(hdus, hdu)
        saved = _read_json(hdus, kw.META_EXT) if kw.META_EXT in hdus else {}
        header_meta = parse_raw_header(
            header,
            image.shape,
            category=category if category is not None else saved.get("category"),
        )
        if saved.get("category", header_meta.category) != header_meta.category:
            raise ValueError("Raw header and metadata disagree: category")
        if not saved:
            return header_meta

        # Format-3 files, including current acquisitions, also store setup in JSON.
        header_setup = header_meta.setup
        saved_fields = saved.get("setup", {})
        saved_setup = DetectorSetup(**saved_fields)
        check_setup(header_setup, saved_setup, exposure=True, role="raw metadata")
        if header_setup.exposure_s is not None and "exposure_source" in saved_fields:
            recorded_dit = saved_setup.exposure_source in (
                "readback",
                "legacy-property",
            )
            if recorded_dit != (kw.DIT in header):
                raise ValueError("Raw header and metadata disagree: exposure source")

        physical = {}
        for name in (
            "exposure_s",
            "origin_x",
            "origin_y",
            "width",
            "height",
            "binning_x",
            "binning_y",
            "detector",
            "orientation",
        ):
            value = getattr(header_setup, name)
            physical[name] = value if value is not None else getattr(saved_setup, name)

        saved_shutters = tuple(saved.get("shutters", header_meta.shutters))
        if len(saved_shutters) != 4 or any(
            state is not None and type(state) is not bool for state in saved_shutters
        ):
            raise ValueError("Recorded shutter states must be four booleans or unknown")
        shutters = []
        for header_state, saved_state in zip(
            header_meta.shutters, saved_shutters, strict=True
        ):
            if (
                header_state is not None
                and saved_state is not None
                and header_state != saved_state
            ):
                raise ValueError("Raw header and metadata disagree: shutters")
            shutters.append(header_state if header_state is not None else saved_state)
        active = sum(state is True for state in shutters)
        expected_open = kw.RAW_TYPES[header_meta.category]["open_shutters"]
        if not active <= expected_open <= active + shutters.count(None):
            raise ValueError("Raw category and recorded shutter states conflict")
        setup = DetectorSetup(
            **physical,
            exposure_source=saved_fields.get(
                "exposure_source", header_setup.exposure_source
            ),
            legacy=saved_fields.get("legacy", header_setup.legacy),
        )
        return RawMetadata(header_meta.category, tuple(shutters), setup)


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
