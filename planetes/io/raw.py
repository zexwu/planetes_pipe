"""Raw FITS streaming, known bench metadata and calibration compatibility."""

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from astropy.io import fits as afits
from astropy.time import Time

from ..instrument import BASELINES, TELESCOPES, DetectorSetup
from ..products import Dark
from . import keywords as kw
from .hdu import _checksums, _json_hdu, _read_json
from .publication import destinations

CHUNK_FRAMES = 16


@dataclass(frozen=True, slots=True)
class RawMetadata:
    """Known acquisition metadata; absent shutter readback remains unknown."""

    category: str
    shutters: tuple[bool, bool, bool, bool] | None
    setup: DetectorSetup
    configuration_source: str
    host_start_utc: str | None = None
    host_end_utc: str | None = None
    exposure_start_utc: str | None = None
    original: dict = field(default_factory=dict)


def raw_header(metadata: RawMetadata, shape):
    """ESO ICD exposure/window fields; private shutter provenance stays explicit."""
    if metadata.category not in kw.RAW_TYPES or len(shape) != 3 or min(shape) < 1:
        raise ValueError("Expected a supported raw category and nonempty 3-D cube")
    if (
        metadata.shutters is None
        or len(metadata.shutters) != 4
        or any(not isinstance(v, (bool, np.bool_)) for v in metadata.shutters)
        or sum(metadata.shutters) != kw.RAW_TYPES[metadata.category][3]
    ):
        raise ValueError("Raw category requires explicit, consistent shutter states")
    setup = metadata.setup
    if any(
        getattr(setup, name) is not None and getattr(setup, name) < 0
        for name in ("origin_x", "origin_y")
    ) or any(
        getattr(setup, name) is not None and getattr(setup, name) < 1
        for name in ("binning_x", "binning_y")
    ):
        raise ValueError("Invalid detector ROI origin or binning")
    if (setup.height is not None and setup.height != shape[1]) or (
        setup.width is not None and setup.width != shape[2]
    ):
        raise ValueError("Raw detector size differs from the recorded ROI")
    if not setup.legacy and any(
        getattr(setup, name) is None
        for name in (
            "origin_x",
            "origin_y",
            "width",
            "height",
            "binning_x",
            "binning_y",
            "exposure_s",
            "detector",
            "orientation",
        )
    ):
        raise ValueError("New raw data require known detector/ROI/exposure metadata")
    if setup.legacy and metadata.configuration_source != "legacy-role-map":
        raise ValueError("Legacy raw requires an explicit adopted shutter role map")
    if setup.exposure_s is not None and (
        not np.isfinite(setup.exposure_s) or setup.exposure_s <= 0
    ):
        raise ValueError("Exposure must be positive finite seconds")
    h = afits.Header()
    h["INSTRUME"] = "PLANETES"
    category, kind, technique, _ = kw.RAW_TYPES[metadata.category]
    for key, value in (
        (kw.DPR_CATEGORY, category),
        (kw.DPR_TYPE, kind),
        (kw.DPR_TECHNIQUE, technique),
    ):
        h["HIERARCH " + key] = value
    h["HIERARCH " + kw.FORMAT_KEY] = kw.RAW_FORMAT
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


def _detector_cube(hdus, hdu: int | str | None = None):
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
    kind = str(header.get(kw.DPR_TYPE, "")).strip().upper()
    declared = next((name for name, v in kw.RAW_TYPES.items() if v[1] == kind), None)
    category = declared if category is None else category
    if category not in kw.RAW_TYPES:
        raise ValueError("Raw category is unknown; supply an explicit category")
    if declared is not None and declared != category:
        raise ValueError(f"Expected raw category {category}, header declares {kind}")
    if len(shape) != 3 or min(shape) < 1:
        raise ValueError("Expected a nonempty three-dimensional raw detector cube")

    shutter_values = tuple(header.get(key) for key in kw.SHUTTER_KEYS)
    if any(value is not None and type(value) is not bool for value in shutter_values):
        raise ValueError("Recorded shutter states must be boolean")
    known_open = sum(value is True for value in shutter_values)
    unknown = shutter_values.count(None)
    if not known_open <= kw.RAW_TYPES[category][3] <= known_open + unknown:
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
        shutters=shutter_values if unknown == 0 else None,
        setup=DetectorSetup(
            **roi,
            exposure_s=exposure,
            exposure_source="readback"
            if kw.DIT in header
            else ("command" if kw.UIT in header else "unknown"),
            detector=header.get("ESO DET ID"),
            legacy=True,
        ),
        configuration_source="readback" if unknown == 0 else "pending",
        exposure_start_utc=header.get("DATE-OBS"),
        original={"header_status": "provisional", "shutters": shutter_values},
    )
    return raw


def read_raw_metadata(
    path: Path, *, category=None, hdu: int | str | None = None
) -> RawMetadata:
    """Read strict version-3 raw metadata or known fields from a provisional bench FITS."""
    with afits.open(path, memmap=True) as hdus:
        h = hdus[0].header
        if kw.PRO_CATEGORY in h:
            raise ValueError(f"Expected raw FITS, found pipeline product: {path}")
        image = _detector_cube(hdus, hdu)
        shape = tuple(image.shape)
        if kw.FORMAT_KEY not in h and kw.META_EXT not in hdus:
            raw = parse_raw_header(h, shape, category=category)
            raw.original["unverified"] = (
                ["RAW.BUNIT"]
                if "BUNIT" not in h and "BUNIT" not in image.header
                else []
            )
            return raw
        if (
            h.get(kw.FORMAT_KEY) != kw.RAW_FORMAT
            or h.get("INSTRUME") != "PLANETES"
            or kw.META_EXT not in hdus
            or kw.RAW_IMAGE not in hdus
            or image is not hdus[kw.RAW_IMAGE]
        ):
            raise ValueError(
                f"Expected complete PLANETES raw format {kw.RAW_FORMAT}: {path}"
            )
        info = _read_json(hdus, kw.META_EXT)
        info["setup"] = DetectorSetup(**info["setup"])
        info["shutters"] = tuple(info["shutters"])
        raw = RawMetadata(**info)
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


def identify_raw(
    paths: Mapping[str, Path] | Iterable[Path], category: str
) -> dict[str, Path]:
    """Use explicit physical identities or shutter readback, never filenames or order."""
    expected = set(TELESCOPES if category == "FLAT_RAW" else BASELINES)
    if isinstance(paths, Mapping):
        result = {str(name): Path(path) for name, path in paths.items()}
        if set(result) != expected or len(result) != len(paths):
            raise ValueError(
                f"Expected explicit {category} identities {sorted(expected)}"
            )
        for identity, path in result.items():
            raw = read_raw_metadata(path, category=category)
            shutters = raw.shutters or raw.original.get("shutters", ())
            for telescope, state in enumerate(shutters, 1):
                if state is not None and state != (str(telescope) in identity):
                    raise ValueError(
                        f"Explicit identity {identity} conflicts with recorded shutters: {path}"
                    )
        if len(set(result.values())) != len(result):
            raise ValueError("Each physical identity requires a different raw file")
        return result
    result = {}
    for path in paths:
        raw = read_raw_metadata(Path(path), category=category)
        if raw.shutters is None:
            raise ValueError(
                f"Unknown shutters for {path}; supply an explicit identity mapping"
            )
        active = [str(i + 1) for i, value in enumerate(raw.shutters) if value]
        name = "".join(sorted(active, reverse=True))
        if name in result:
            raise ValueError(f"Duplicate shutter configuration {name} for {category}")
        result[name] = Path(path)
    if set(result) != expected:
        raise ValueError(
            f"Incomplete {category} shutter set: expected {sorted(expected)}"
        )
    return result


def iter_raw_cube(path: Path, *, hdu: int | str | None = None):
    """Stream a named/unique detector cube as float32, retaining 16-frame arithmetic."""
    with afits.open(path, memmap=True) as hdus:
        data = _detector_cube(hdus, hdu).data
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

    if not isinstance(dark, Dark) or dark.n_frames < 2:
        raise ValueError("Formal file tasks require complete DARK calibrations")
    raw = read_raw_metadata(path, category=category)
    check_setup(raw.setup, dark.setup, exposure=True, role="DARK")
    geometry = flat.setup if flat is not None else flat_setup
    if flat is not None or flat_setup is not None:
        check_setup(raw.setup, geometry, exposure=False, role="FLAT")


def verify_raw(path):
    """Check classified metadata and every HDU checksum at publication."""
    read_raw_metadata(path)
    with afits.open(path, checksum=True) as hdus:
        _checksums(hdus)
