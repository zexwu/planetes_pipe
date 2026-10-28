"""Raw FITS streaming, known bench metadata and calibration compatibility."""

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits as afits

from ..instrument import BASELINES, TELESCOPES, DetectorSetup, check_setup
from . import keywords as kw
from .hdu import _verify_checksums, _read_json, as_compute_array

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
    expected_open_shutters = kw.RAW_TYPES[category]["open_shutters"]
    if not known_open <= expected_open_shutters <= known_open + unknown:
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
    """Read known physical fields; acquisition timing/provenance is not parsed."""
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
        info = _read_json(hdus, kw.META_EXT) if kw.META_EXT in hdus else {}
        raw = parse_raw_header(
            header,
            image.shape,
            category=category if category is not None else info.get("category"),
        )
        if info.get("category", raw.category) != raw.category:
            raise ValueError("Raw header and metadata disagree: category")
        if not info:
            return raw
        # Old format-3 files retain physical fields in PLANETES_META. Ignore
        # capture timestamps, configuration sources and original-header records.
        setup = DetectorSetup(**{**asdict(raw.setup), **info.get("setup", {})})
        check_setup(raw.setup, setup, exposure=True, role="raw metadata")
        if raw.setup.exposure_s is not None and "exposure_source" in info.get(
            "setup", {}
        ):
            recorded_dit = setup.exposure_source in ("readback", "legacy-property")
            if recorded_dit != (kw.DIT in header):
                raise ValueError("Raw header and metadata disagree: exposure source")
        known = {
            key: value
            for key, value in asdict(raw.setup).items()
            if value is not None and key not in ("legacy", "exposure_source")
        }
        values = {**asdict(setup), **known}
        shutters = tuple(info.get("shutters", raw.shutters))
        if len(shutters) != 4 or any(
            state is not None and type(state) is not bool for state in shutters
        ):
            raise ValueError("Recorded shutter states must be four booleans or unknown")
        for first, second in zip(raw.shutters, shutters, strict=True):
            if first is not None and second is not None and first != second:
                raise ValueError("Raw header and metadata disagree: shutters")
        shutters = tuple(
            first if first is not None else second
            for first, second in zip(raw.shutters, shutters, strict=True)
        )
        active = sum(state is True for state in shutters)
        expected_open_shutters = kw.RAW_TYPES[raw.category]["open_shutters"]
        if not active <= expected_open_shutters <= active + shutters.count(None):
            raise ValueError("Raw category and recorded shutter states conflict")
        return RawMetadata(raw.category, shutters, DetectorSetup(**values))


def identify_raw_exposures(
    paths: Mapping[str, Path] | Iterable[Path], category: str
) -> dict[str, tuple[Path, RawMetadata]]:
    """Read once and retain physical identity with each path and its metadata."""
    expected = set(TELESCOPES if category == "FLAT_RAW" else BASELINES)
    explicit = isinstance(paths, Mapping)
    entries = paths.items() if explicit else ((None, path) for path in paths)
    result = {}
    for identity, path in entries:
        path = Path(path)
        raw = read_raw_metadata(path, category=category)
        if explicit:
            identity = str(identity)
            for telescope, state in enumerate(raw.shutters, 1):
                if state is not None and state != (str(telescope) in identity):
                    raise ValueError(
                        f"Explicit identity {identity} conflicts with recorded shutters: {path}"
                    )
        else:
            if None in raw.shutters:
                raise ValueError(
                    f"Unknown shutters for {path}; supply an explicit identity mapping"
                )
            identity = "".join(str(n) for n in range(4, 0, -1) if raw.shutters[n - 1])
        if identity in result:
            raise ValueError(
                f"Duplicate shutter configuration {identity} for {category}"
            )
        result[identity] = (path, raw)
    if set(result) != expected:
        raise ValueError(f"Expected {category} identities {sorted(expected)}")
    if len({path for path, _ in result.values()}) != len(result):
        raise ValueError("Each physical identity requires a different raw file")
    return result


def iter_raw_cube(path: Path, *, hdu: int | str | None = None):
    """Read at most 16 frames and adapt them to computation precision once."""
    with afits.open(path, memmap=True) as hdus:
        data = _detector_hdu(hdus, hdu).data
        for start in range(0, len(data), CHUNK_FRAMES):
            yield as_compute_array(data[start : start + CHUNK_FRAMES])


def check_raw_calibrations(raw: RawMetadata, dark, flat=None, *, flat_setup=None):
    """Check known DIT/ROI conflicts using metadata already read by the caller."""
    check_setup(raw.setup, dark.setup, exposure=True, role="DARK")
    geometry = flat.setup if flat is not None else flat_setup
    if flat is not None or flat_setup is not None:
        check_setup(raw.setup, geometry, exposure=False, role="FLAT")


def verify_raw(path):
    """Check classified metadata and every HDU checksum at publication."""
    read_raw_metadata(path)
    with afits.open(path, checksum=True) as hdus:
        _verify_checksums(hdus)
