"""Raw acquisition writer; capture provenance is separate from pipeline reading."""

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from astropy.io import fits as afits
from astropy.time import Time

from ..instrument import DetectorSetup
from . import keywords as kw
from .hdu import _json_hdu, _storage_array
from .publication import publish_outputs
from .raw import CHUNK_FRAMES, verify_raw


@dataclass(frozen=True, slots=True)
class CaptureMetadata:
    """Acquisition record, including timing and configuration sources."""

    category: str
    shutters: tuple[bool, bool, bool, bool] | None
    setup: DetectorSetup
    configuration_source: str
    host_start_utc: str | None = None
    host_end_utc: str | None = None
    exposure_start_utc: str | None = None
    original: dict = field(default_factory=dict)


def build_raw_header(metadata: CaptureMetadata, shape):
    """ESO ICD exposure/window fields; private shutter provenance stays explicit."""
    if metadata.category not in kw.RAW_TYPES or len(shape) != 3 or min(shape) < 1:
        raise ValueError("Expected a supported raw category and nonempty 3-D cube")
    raw_type = kw.RAW_TYPES[metadata.category]
    if (
        metadata.shutters is None
        or len(metadata.shutters) != 4
        or any(not isinstance(v, (bool, np.bool_)) for v in metadata.shutters)
        or sum(metadata.shutters) != raw_type["open_shutters"]
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
    hdr = afits.Header()
    hdr["INSTRUME"] = "PLANETES"
    for key, value in (
        (kw.DPR_CATEGORY, raw_type["dpr_category"]),
        (kw.DPR_TYPE, raw_type["dpr_type"]),
        (kw.DPR_TECHNIQUE, raw_type["dpr_technique"]),
    ):
        hdr["HIERARCH " + key] = value
    hdr["HIERARCH " + kw.FORMAT_KEY] = kw.RAW_FORMAT
    hdr["HIERARCH " + kw.NDIT] = shape[0]
    for key, value in zip(kw.SHUTTER_KEYS, metadata.shutters, strict=True):
        hdr["HIERARCH " + key] = value
    for name, key in kw.ROI_KEYS.items():
        value = getattr(metadata.setup, name)
        if value is not None:
            hdr["HIERARCH " + key] = value + (1 if name.startswith("origin_") else 0)
    if metadata.setup.exposure_s is not None:
        key = (
            kw.DIT
            if metadata.setup.exposure_source in ("readback", "legacy-property")
            else kw.UIT
        )
        hdr["HIERARCH " + key] = (
            metadata.setup.exposure_s,
            "[s] Per-integration exposure; source in PLANETES_META",
        )
        if key == kw.DIT:
            hdr["EXPTIME"] = (
                metadata.setup.exposure_s * shape[0],
                "[s] Sum of unaveraged integration times",
            )
    if metadata.exposure_start_utc is not None:
        hdr["DATE-OBS"] = metadata.exposure_start_utc
        hdr["MJD-OBS"] = float(Time(metadata.exposure_start_utc, scale="utc").mjd)
    return hdr


def write_raw(path: Path, cube, metadata: CaptureMetadata):
    """Write floating raw as float32, preserving integer samples and existing files."""
    cube = _storage_array(cube)
    header = build_raw_header(metadata, cube.shape)
    if cube.dtype.kind not in "uif" or any(
        not np.isfinite(cube[start : start + CHUNK_FRAMES]).all()
        for start in range(0, len(cube), CHUNK_FRAMES)
    ):
        raise ValueError("Expected finite real raw detector values")
    image = afits.ImageHDU(cube, name=kw.RAW_IMAGE)
    image.header["BUNIT"] = "adu"
    with publish_outputs([path], verify=verify_raw) as temporary:
        afits.HDUList(
            [
                afits.PrimaryHDU(header=header),
                image,
                _json_hdu(kw.META_EXT, asdict(metadata)),
            ]
        ).writeto(temporary[0], checksum=True)
