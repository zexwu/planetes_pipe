"""Classified raw FITS, physical camera metadata and calibration compatibility."""

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from astropy.io import fits as afits
from astropy.time import Time

from ..instrument import BASELINES, TELESCOPES, DetectorSetup
from ..products import DarkCalibration, validate
from . import keywords as kw
from .hdu import _checksums, _json_hdu, _read_json
from .publication import destinations

CHUNK_FRAMES = 16


@dataclass(frozen=True, slots=True)
class RawMetadata:
    """Raw category, physical shutters 1..4 and explicitly sourced capture metadata."""

    category: str
    shutters: tuple[bool, bool, bool, bool]
    setup: DetectorSetup
    configuration_source: str
    host_start_utc: str | None = None
    host_end_utc: str | None = None
    exposure_start_utc: str | None = None
    original: dict = field(default_factory=dict)


def validate_raw_metadata(metadata: RawMetadata, shape):
    """Validate real category/configuration, retaining explicitly unknown legacy fields."""
    if metadata.category not in kw.RAW_TYPES or len(shape) != 3 or min(shape) < 1:
        raise ValueError("Expected a supported raw category and nonempty 3-D cube")
    if len(metadata.shutters) != 4 or any(
        type(x) is not bool for x in metadata.shutters
    ):
        raise ValueError("Four explicit boolean physical shutter states are required")
    if sum(metadata.shutters) != kw.RAW_TYPES[metadata.category][3]:
        raise ValueError("Raw category and shutter configuration conflict")
    if metadata.configuration_source not in (
        "readback",
        "command",
        "operator",
        "legacy-role-map",
    ):
        raise ValueError("Unknown shutter configuration source")
    if metadata.setup.exposure_source not in (
        "readback",
        "command",
        "operator",
        "legacy-property",
        "unknown",
    ):
        raise ValueError("Unknown exposure source")
    setup = metadata.setup
    for key in (
        "origin_x",
        "origin_y",
        "width",
        "height",
        "binning_x",
        "binning_y",
        "exposure_s",
        "detector",
        "orientation",
    ):
        value = getattr(setup, key)
        if value is None:
            if not setup.legacy:
                raise ValueError(f"New raw data require {key}")
            continue
        if key in ("origin_x", "origin_y"):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"Invalid raw {key}")
        elif key in ("width", "height", "binning_x", "binning_y"):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"Invalid raw {key}")
        elif key == "exposure_s" and (not np.isfinite(value) or value <= 0):
            raise ValueError("Exposure must be positive finite seconds")
        elif key in ("detector", "orientation") and (
            not isinstance(value, str) or not value.strip()
        ):
            raise ValueError(f"Invalid raw {key}")
    if not setup.legacy and setup.exposure_source not in (
        "readback",
        "command",
        "operator",
    ):
        raise ValueError("New raw data require a known exposure source")
    if (setup.height, setup.width) != shape[1:]:
        raise ValueError("Raw detector size differs from the recorded ROI")
    if setup.legacy and metadata.configuration_source != "legacy-role-map":
        raise ValueError("Legacy raw requires an explicit adopted role map")
    for value in (
        metadata.host_start_utc,
        metadata.host_end_utc,
        metadata.exposure_start_utc,
    ):
        if value is not None:
            Time(value, format="isot", scale="utc")


def raw_header(metadata: RawMetadata, shape):
    """ESO ICD exposure/window fields; private shutter provenance stays explicit."""
    validate_raw_metadata(metadata, shape)
    h = afits.Header()
    h["INSTRUME"] = "PLANETES"
    category, kind, technique, _ = kw.RAW_TYPES[metadata.category]
    for key, value in (
        (kw.DPR_CATEGORY, category),
        (kw.DPR_TYPE, kind),
        (kw.DPR_TECHNIQUE, technique),
    ):
        h["HIERARCH " + key] = value
    h["HIERARCH " + kw.FORMAT_KEY] = kw.SCHEMA_VERSION
    h["HIERARCH " + kw.NDIT] = shape[0]
    for key, value in zip(kw.SHUTTER_KEYS, metadata.shutters, strict=True):
        h["HIERARCH " + key] = value
    for name, key in kw.ROI_KEYS.items():
        value = getattr(metadata.setup, name)
        if value is not None:
            h["HIERARCH " + key] = value + (1 if name.startswith("origin_") else 0)
    if metadata.setup.exposure_s is not None:
        key = (
            kw.DIT
            if metadata.setup.exposure_source in ("readback", "legacy-property")
            else kw.UIT
        )
        h["HIERARCH " + key] = (
            metadata.setup.exposure_s,
            "[s] Per-integration exposure; source in PLANETES_META",
        )
        if key == kw.DIT:
            h["EXPTIME"] = (
                metadata.setup.exposure_s * shape[0],
                "[s] Sum of unaveraged integration times",
            )
    if metadata.exposure_start_utc is not None:
        h["DATE-OBS"] = metadata.exposure_start_utc
        h["MJD-OBS"] = float(Time(metadata.exposure_start_utc, scale="utc").mjd)
    return h


def read_raw_metadata(path: Path, *, category=None) -> RawMetadata:
    """Read only the new raw contract, not a filename/old-tag compatibility heuristic."""
    with afits.open(path, memmap=True) as hdus:
        h = hdus[0].header
        if (
            h.get(kw.FORMAT_KEY) != kw.SCHEMA_VERSION
            or h.get("INSTRUME") != "PLANETES"
            or kw.PRO_CATEGORY in h
        ):
            raise ValueError(
                f"Expected new PLANETES raw FITS; standardize old raw first: {path}"
            )
        info = _read_json(hdus, kw.META_EXT)
        info["setup"] = DetectorSetup(**info["setup"])
        info["shutters"] = tuple(info["shutters"])
        raw = RawMetadata(**info)
        image = hdus[kw.RAW_IMAGE]
        shape = tuple(image.shape)
        validate_raw_metadata(raw, shape)
        if category is not None and raw.category != category:
            raise ValueError(f"Expected raw category {category}, got {raw.category}")
        expected = raw_header(raw, shape)
        for key in expected:
            if h.get(key) != expected[key]:
                raise ValueError(f"Raw header and metadata disagree: {key}")
        # A fast header read does not stream/checksum the full 10 GB dataset.
        # Publication and the migration verifier check every pixel/checksum once.
        if image.header.get("BUNIT") != "adu":
            raise ValueError("Raw detector values must be ADU")
        return raw


def identify_raw(paths, category):
    """Resolve repeated raw categories by physical shutters, never order or filename."""
    result = {}
    for path in paths:
        raw = read_raw_metadata(Path(path), category=category)
        active = [str(i + 1) for i, value in enumerate(raw.shutters) if value]
        name = "".join(sorted(active, reverse=True))
        if name in result:
            raise ValueError(f"Duplicate shutter configuration {name} for {category}")
        result[name] = Path(path)
    expected = set(TELESCOPES if category == "FLAT_RAW" else BASELINES)
    if set(result) != expected:
        raise ValueError(
            f"Incomplete {category} shutter set: expected {sorted(expected)}"
        )
    return result


def iter_raw_cube(path: Path):
    """Read every named detector frame as float32, retaining 16-frame arithmetic."""
    read_raw_metadata(path)
    with afits.open(path, memmap=True) as hdus:
        data = hdus[kw.RAW_IMAGE].data
        for start in range(0, len(data), CHUNK_FRAMES):
            yield np.asarray(data[start : start + CHUNK_FRAMES], dtype=np.float32)


def write_raw(path: Path, cube, metadata: RawMetadata):
    """Write original raw values/dtype without overwriting or operating hardware."""
    cube = np.asarray(cube)
    header = raw_header(metadata, cube.shape)
    if cube.dtype.kind not in "uif" or any(
        not np.isfinite(cube[start : start + CHUNK_FRAMES]).all()
        for start in range(0, len(cube), CHUNK_FRAMES)
    ):
        raise ValueError("Expected finite real raw detector values")
    image = afits.ImageHDU(cube, name=kw.RAW_IMAGE)
    image.header["BUNIT"] = "adu"
    with destinations([path], verify=verify_raw) as temporary:
        afits.HDUList(
            [
                afits.PrimaryHDU(header=header),
                image,
                _json_hdu(kw.META_EXT, asdict(metadata)),
            ]
        ).writeto(temporary[0], checksum=True)


def check_raw_calibrations(path, dark, flat=None, *, flat_setup=None, category=None):
    """Shared formal task boundary: known DIT/ROI conflicts are scientific errors."""
    from ..instrument import check_setup

    if not isinstance(dark, DarkCalibration):
        raise ValueError("Formal file tasks require complete DARK calibrations")
    validate(dark)
    raw = read_raw_metadata(path, category=category)
    unknown = list(check_setup(raw.setup, dark.setup, exposure=True, role="DARK"))
    geometry = flat.setup if flat is not None else flat_setup
    if flat is not None or flat_setup is not None:
        unknown.extend(check_setup(raw.setup, geometry, exposure=False, role="FLAT"))
    return tuple(sorted(set(unknown)))


def verify_raw(path):
    """Check classified metadata and every HDU checksum at publication."""
    read_raw_metadata(path)
    with afits.open(path, checksum=True) as hdus:
        _checksums(hdus)
