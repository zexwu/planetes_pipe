"""Write raw FITS images with physical fields in the header."""

from pathlib import Path

import numpy as np
from astropy.io import fits as afits

from . import keywords as kw
from .hdu import _storage_array
from .publication import publish_outputs
from .raw import CHUNK_FRAMES, RawMetadata


def build_raw_header(metadata: RawMetadata, shape):
    """Encode category, shutters and physical detector settings in FITS cards."""
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
    hdr["HIERARCH " + kw.NDIT] = shape[0]
    for key, value in zip(kw.SHUTTER_KEYS, metadata.shutters, strict=True):
        hdr["HIERARCH " + key] = value
    for name, key in kw.ROI_KEYS.items():
        value = getattr(metadata.setup, name)
        if value is not None:
            hdr["HIERARCH " + key] = value + (1 if name.startswith("origin_") else 0)
    if setup.detector is not None:
        hdr["HIERARCH " + kw.DETECTOR_ID] = setup.detector
    if setup.exposure_s is not None:
        hdr["HIERARCH " + kw.DIT] = (setup.exposure_s, "[s] Integration time")
        hdr["EXPTIME"] = (setup.exposure_s * shape[0], "[s] Sum of integration times")
    return hdr


def write_raw(path: Path, cube, metadata: RawMetadata):
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
    with publish_outputs([path]) as temporary:
        afits.HDUList([afits.PrimaryHDU(header=header), image]).writeto(
            temporary[0], checksum=True
        )
