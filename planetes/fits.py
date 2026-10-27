"""Raw and science FITS with GRAVITY-derived vocabulary; no execution framework."""

from contextlib import contextmanager
from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile

import numpy as np
from astropy.io import fits as afits
from astropy.time import Time

from . import keywords as kw
from .instrument import TELESCOPES, BASELINES, ABCD_ORDER
from .models import (
    DarkCalibration,
    BadPixelMap,
    FlatCalibration,
    WavelengthCalibration,
    P2VMCalibration,
    ReducedObservables,
    WavelengthReference,
    Provenance,
    DetectorSetup,
    RawMetadata,
    canonical_quantities,
    retain_representation,
    fingerprint,
    validate,
)

CATEGORIES = {
    DarkCalibration: "DARK",
    BadPixelMap: "BAD",
    FlatCalibration: "FLAT",
    WavelengthCalibration: "WAVE",
    P2VMCalibration: "P2VM",
    ReducedObservables: "SINGLE_SCI_P2VMRED",
    WavelengthReference: "LINE_LIST",
}
CHUNK_FRAMES = 16


def _native(values):
    return np.array(values, dtype=values.dtype.newbyteorder("="), copy=True)


def _json_hdu(name, values):
    data = json.dumps(
        values, sort_keys=True, allow_nan=False, default=lambda v: v.item()
    ).encode("utf-8")
    return afits.ImageHDU(np.frombuffer(data, dtype=np.uint8).copy(), name=name)


def _read_json(hdus, name):
    return json.loads(hdus[name].data.tobytes().decode("utf-8"))


def _checksums(hdus):
    for h in hdus:
        if h.verify_checksum() != 1 or h.verify_datasum() != 1:
            raise ValueError("Invalid or missing FITS checksum")


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
        ("ESO DPR CATG", category),
        ("ESO DPR TYPE", kind),
        ("ESO DPR TECH", technique),
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
            or "ESO PRO CATG" in h
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
    with destinations([path], raw=True) as temporary:
        afits.HDUList(
            [
                afits.PrimaryHDU(header=header),
                image,
                _json_hdu(kw.META_EXT, asdict(metadata)),
            ]
        ).writeto(temporary[0], checksum=True)


def _image(name, value, unit=""):
    h = afits.ImageHDU(value, name=name)
    h.header["BUNIT"] = unit
    return h


def _column(name, values, unit="", dim=None):
    values = np.asarray(values)
    kind = {
        "f": "D" if values.dtype.itemsize == 8 else "E",
        "c": "M" if values.dtype.itemsize == 16 else "C",
        "b": "L",
        "i": "K",
        "u": "K",
    }.get(values.dtype.kind)
    if kind is None:
        return afits.Column(
            name=name, array=values, format=f"{max(1,max(map(len,values),default=1))}A"
        )
    count = int(np.prod(values.shape[1:])) if values.ndim > 1 else 1
    if dim is None and values.ndim > 1:
        dim = "(" + ",".join(map(str, values.shape[:0:-1])) + ")"
    return afits.Column(
        name=name, array=values, format=f"{count}{kind}", unit=unit or None, dim=dim
    )


def _table(name, columns):
    return afits.BinTableHDU.from_columns(columns, name=name)


def _detector(product):
    labels = [f"REG{i+1:02d}" for i in range(24)]
    if isinstance(product, (P2VMCalibration, ReducedObservables)):
        for baseline, outputs in product.baseline_regions.items():
            for output, phase in zip(outputs, ABCD_ORDER, strict=True):
                labels[output] = BASELINES[baseline] + "-" + phase
    return _table(kw.DETECTOR_DATA, [_column("REGNAME", np.asarray(labels))])


def encode_product(product, header=None):
    """Encode GRAVITY-derived fields and named necessary PLANETES extensions."""
    validate(product)
    if type(product) not in CATEGORIES:
        raise TypeError(
            "Only complete DARK/BAD/calibration/science objects can be saved"
        )
    h = afits.Header() if header is None else header.copy()
    for key in list(h):
        if key.startswith("ESO DPR "):
            del h[key]
    h["INSTRUME"] = "PLANETES"
    h["HIERARCH " + kw.FORMAT_KEY] = kw.SCHEMA_VERSION
    h["HIERARCH " + kw.HASH_KEY] = fingerprint(product)
    h["HIERARCH ESO PRO CATG"] = CATEGORIES[type(product)]
    h["HIERARCH ESO PRO TYPE"] = (
        "STATIC" if isinstance(product, WavelengthReference) else "REDUCED"
    )
    h["HIERARCH ESO PRO SCIENCE"] = isinstance(product, ReducedObservables)
    setup = getattr(product, "setup", None)
    if setup is not None:
        for name, key in kw.ROI_KEYS.items():
            v = getattr(setup, name)
            if v is not None:
                h["HIERARCH " + key] = v + (1 if name.startswith("origin_") else 0)
        if setup.exposure_s is not None:
            key = (
                kw.DIT
                if setup.exposure_source in ("readback", "legacy-property")
                else kw.UIT
            )
            h["HIERARCH " + key] = (setup.exposure_s, "[s] Source in PLANETES_META")
    canonical = canonical_quantities(product)

    def value(name):
        return canonical.get(name, getattr(product, name))

    hdus = [afits.PrimaryHDU(header=h)]
    if isinstance(product, DarkCalibration):
        hdu = hdus[0]
        hdu.header["NFRAMES"] = product.n_frames
        hdus += [
            _image(kw.RAW_IMAGE, product.mean_adu, "adu"),
            _image(kw.DARK_RMS, product.temporal_rms_adu, "adu"),
        ]
    elif isinstance(product, BadPixelMap):
        image = _image(kw.RAW_IMAGE, product.bad_pixel_flags)
        image.header["HIERARCH PLANETES BAD BITS"] = "hot=1,dead=2,noisy=4"
        hdus.append(image)
    elif isinstance(product, FlatCalibration):
        hdus += [
            _detector(product),
            _image("PLANETES_FLAT_IMAGE", product.flat_map_adu, "adu"),
        ]
        columns = [
            _column("REGNAME", hdus[1].data["REGNAME"]),
            _column("CENTERY", value("trace_y_pixels"), "pixel"),
        ]
        if product.profile_sigma_pixels.size:
            columns.append(_column("WIDTH", product.profile_sigma_pixels, "pixel"))
        hdus += [
            _table(kw.PROFILE_PARAMS, columns),
            _table(
                "PLANETES_PROFILE",
                [
                    _column("PIXEL_X", value("profile_x_pixels"), "pixel"),
                    _column("PIXEL_Y", value("profile_y_pixels"), "pixel"),
                ],
            ),
        ]
        if product.telescope_flux_adu.size:
            hdus.append(_image("PLANETES_FLAT_FLUX", product.telescope_flux_adu, "adu"))
    elif isinstance(product, WavelengthCalibration):
        hdus += [
            _detector(product),
            _table(
                kw.WAVE_DATA,
                [
                    _column(f"DATA{i+1}", row[None], "m")
                    for i, row in enumerate(value("wavelength_map_um"))
                ],
            ),
        ]
        if product.fit_residual_um.size:
            hdus.append(
                _table(
                    "PLANETES_WAVE_FIT",
                    [
                        _column("RESIDUAL", value("fit_residual_um"), "m"),
                        _column("X", product.fit_x_pixels, "pixel"),
                        _column("Y", product.fit_y_pixels, "pixel"),
                    ],
                )
            )
    elif isinstance(product, WavelengthReference):
        hdus.append(
            _table(
                "LINE_LIST",
                [
                    _column("WAVELENGTH", value("wavelength_um"), "m"),
                    _column("ORIGIN", np.asarray(product.origin)),
                ],
            )
        )
    else:
        hdus += [
            _detector(product),
            _table(
                "PLANETES_WAVELENGTH",
                [_column("WAVELENGTH", value("wavelength_um"), "m")],
            ),
        ]
        if isinstance(product, P2VMCalibration):
            columns = [
                _column("REGNAME", hdus[1].data["REGNAME"]),
                _column("TRANSMISSION", product.transmission),
                _column("COHERENCE", product.coherence_amplitude),
                _column("PHASE", product.coherence_phase_rad, "rad"),
            ]
            if product.dc_term_adu.size:
                columns.append(_column("C_MATRIX", product.dc_term_adu, "adu"))
            hdus.append(_table(kw.P2VM_DATA, columns))
            for name, ext, unit in [
                ("p2vm", "PLANETES_P2VM_INVERSE", ""),
                ("opd_um", "PLANETES_OPD", "m"),
                ("group_delay_um", "PLANETES_GD", "m"),
                ("ellipse_phase_rad", "PLANETES_ELLIPSE_PHASE", "rad"),
                ("ellipse_amplitude", "PLANETES_ELLIPSE_AMPLITUDE", ""),
                ("fit_rms_adu", "PLANETES_P2VM_FIT_RMS", "adu"),
            ]:
                if value(name).size:
                    hdus.append(_image(ext, value(name), unit))
        else:
            nf, _, nw = product.photometric_flux_adu.shape
            hdus += [
                _table(
                    "PLANETES_FLUX",
                    [
                        _column("FRAME", np.repeat(np.arange(1, nf + 1), 4)),
                        _column(
                            "STA_INDEX", np.tile(np.asarray(TELESCOPES, dtype=int), nf)
                        ),
                        _column(
                            "FLUXDATA",
                            product.photometric_flux_adu.reshape(-1, nw),
                            "adu",
                        ),
                    ],
                )
            ]
            pairs = np.asarray(
                [
                    [int(TELESCOPES[t]) for t in product.baseline_telescopes[b]]
                    for b in range(6)
                ]
            )
            hdus.append(
                _table(
                    "PLANETES_VIS",
                    [
                        _column("FRAME", np.repeat(np.arange(1, nf + 1), 6)),
                        _column("STA_INDEX", np.tile(pairs, (nf, 1))),
                        _column(
                            "VISDATA", product.coherent_flux_adu.reshape(-1, nw), "adu"
                        ),
                        _column("VISAMP", product.visibility_amplitude.reshape(-1, nw)),
                        _column("FLAG", product.flags.reshape(-1, nw)),
                        _column("GDELAY", value("group_delay_um").reshape(-1), "m"),
                        _column(
                            "DIFFPHASE",
                            product.differential_phase_deg.reshape(-1, nw),
                            "deg",
                        ),
                    ],
                )
            )
    info = {
        "producer": product.provenance.producer,
        "parameters": product.provenance.parameters,
        "inputs": product.provenance.inputs,
        "dependencies": product.provenance.dependencies,
        "unverified": product.provenance.unverified,
        "qc_status": product.provenance.qc_status,
    }
    metadata = {
        "software_version": __import__("planetes").__version__,
        "provenance": info,
        "setup": asdict(product.setup)
        if getattr(product, "setup", None) is not None
        else None,
    }
    hdus.append(_json_hdu(kw.META_EXT, metadata))
    # The same QC measurements are written for Python and CPL products.
    from .qc import measure_quality

    quality, missing = measure_quality(product)
    for key, (number, comment) in quality.items():
        hdus[0].header["HIERARCH " + key] = (number, comment)
    metadata["provenance"]["qc_status"] = {**info["qc_status"], **missing}
    hdus[-1] = _json_hdu(kw.META_EXT, metadata)
    return afits.HDUList(hdus)


def read_product(path: Path, cls=None):
    """Verify checksums, physical units/identities and canonical content before use."""
    with afits.open(path, checksum=True) as hdus:
        h = hdus[0].header
        if h.get(kw.FORMAT_KEY) != kw.SCHEMA_VERSION or h.get("INSTRUME") != "PLANETES":
            raise ValueError(
                "Expected new PLANETES format; regenerate old v0.3/schema-2 products"
            )
        actual = {v: k for k, v in CATEGORIES.items()}.get(h.get("ESO PRO CATG"))
        if actual is None or (cls is not None and actual is not cls):
            raise ValueError("Unexpected product category")
        _checksums(hdus)
        meta = _read_json(hdus, kw.META_EXT)
        info = meta["provenance"]
        info["unverified"] = tuple(info.get("unverified", ()))
        values = {"provenance": Provenance(**info)}
        canonical = {}

        def image(ext, unit):
            item = hdus[ext]
            if item.header.get("BUNIT", "") != unit:
                raise ValueError(f"Incompatible units: {ext}")
            return _native(item.data)

        def column(ext, name, unit=""):
            table = hdus[ext]
            if (table.columns[name].unit or "") != unit:
                raise ValueError(f"Incompatible units: {ext}.{name}")
            return _native(table.data[name])

        def quantity(name, v, *, offset=0):
            canonical[name] = v
            if offset:
                return (
                    np.where(v < 0, v, v - offset)
                    if v.dtype.kind in "iu"
                    else v - offset
                )
            return v * 1e6

        if actual is DarkCalibration:
            values.update(
                mean_adu=image(kw.RAW_IMAGE, "adu"),
                temporal_rms_adu=image(kw.DARK_RMS, "adu"),
                n_frames=h["NFRAMES"],
            )
        elif actual is BadPixelMap:
            if (
                hdus[kw.RAW_IMAGE].header.get("PLANETES BAD BITS")
                != "hot=1,dead=2,noisy=4"
            ):
                raise ValueError("Invalid BAD flag definition")
            values["bad_pixel_flags"] = image(kw.RAW_IMAGE, "")
        elif actual is FlatCalibration:
            values.update(
                flat_map_adu=image("PLANETES_FLAT_IMAGE", "adu"),
                trace_y_pixels=quantity(
                    "trace_y_pixels",
                    column(kw.PROFILE_PARAMS, "CENTERY", "pixel"),
                    offset=1,
                ),
                profile_x_pixels=quantity(
                    "profile_x_pixels",
                    column("PLANETES_PROFILE", "PIXEL_X", "pixel"),
                    offset=1,
                ),
                profile_y_pixels=quantity(
                    "profile_y_pixels",
                    column("PLANETES_PROFILE", "PIXEL_Y", "pixel"),
                    offset=1,
                ),
            )
            if "WIDTH" in hdus[kw.PROFILE_PARAMS].columns.names:
                values["profile_sigma_pixels"] = column(
                    kw.PROFILE_PARAMS, "WIDTH", "pixel"
                )
            if "PLANETES_FLAT_FLUX" in hdus:
                values["telescope_flux_adu"] = image("PLANETES_FLAT_FLUX", "adu")
        elif actual is WavelengthCalibration:
            values["wavelength_map_um"] = quantity(
                "wavelength_map_um",
                np.stack(
                    [column(kw.WAVE_DATA, f"DATA{i+1}", "m")[0] for i in range(24)]
                ),
            )
            if "PLANETES_WAVE_FIT" in hdus:
                values.update(
                    fit_residual_um=quantity(
                        "fit_residual_um", column("PLANETES_WAVE_FIT", "RESIDUAL", "m")
                    ),
                    fit_x_pixels=column("PLANETES_WAVE_FIT", "X", "pixel"),
                    fit_y_pixels=column("PLANETES_WAVE_FIT", "Y", "pixel"),
                )
        elif actual is WavelengthReference:
            values.update(
                wavelength_um=quantity(
                    "wavelength_um", column("LINE_LIST", "WAVELENGTH", "m")
                ),
                origin=tuple(str(v).strip() for v in hdus["LINE_LIST"].data["ORIGIN"]),
            )
        else:
            values["wavelength_um"] = quantity(
                "wavelength_um", column("PLANETES_WAVELENGTH", "WAVELENGTH", "m")
            )
            labels = list(hdus[kw.DETECTOR_DATA].data["REGNAME"])
            values["baseline_regions"] = {
                i: [labels.index(name + "-" + a) for a in ABCD_ORDER]
                for i, name in enumerate(BASELINES)
            }
            values["baseline_telescopes"] = {
                i: (TELESCOPES.index(name[0]), TELESCOPES.index(name[1]))
                for i, name in enumerate(BASELINES)
            }
            if actual is P2VMCalibration:
                if list(hdus[kw.P2VM_DATA].data["REGNAME"]) != labels:
                    raise ValueError("P2VM and detector REGNAME mismatch")
                values.update(
                    transmission=column(kw.P2VM_DATA, "TRANSMISSION"),
                    coherence_amplitude=column(kw.P2VM_DATA, "COHERENCE"),
                    coherence_phase_rad=column(kw.P2VM_DATA, "PHASE", "rad"),
                )
                if "C_MATRIX" in hdus[kw.P2VM_DATA].columns.names:
                    values["dc_term_adu"] = column(kw.P2VM_DATA, "C_MATRIX", "adu")
                for name, ext, unit in [
                    ("p2vm", "PLANETES_P2VM_INVERSE", ""),
                    ("opd_um", "PLANETES_OPD", "m"),
                    ("group_delay_um", "PLANETES_GD", "m"),
                    ("ellipse_phase_rad", "PLANETES_ELLIPSE_PHASE", "rad"),
                    ("ellipse_amplitude", "PLANETES_ELLIPSE_AMPLITUDE", ""),
                    ("fit_rms_adu", "PLANETES_P2VM_FIT_RMS", "adu"),
                ]:
                    if ext in hdus:
                        v = image(ext, unit)
                        values[name] = quantity(name, v) if unit == "m" else v
            else:
                nw = len(values["wavelength_um"])
                nf = len(hdus["PLANETES_FLUX"].data) // 4
                for ext, count in [("PLANETES_FLUX", 4), ("PLANETES_VIS", 6)]:
                    if not np.array_equal(
                        column(ext, "FRAME"), np.repeat(np.arange(1, nf + 1), count)
                    ):
                        raise ValueError("Invalid frame row order")
                if not np.array_equal(
                    column("PLANETES_FLUX", "STA_INDEX"),
                    np.tile(np.asarray(TELESCOPES, dtype=int), nf),
                ):
                    raise ValueError("Invalid telescope identity")
                expected = np.tile(
                    np.array([[int(v) for v in name] for name in BASELINES]), (nf, 1)
                )
                if not np.array_equal(column("PLANETES_VIS", "STA_INDEX"), expected):
                    raise ValueError("Invalid baseline identity")
                values.update(
                    photometric_flux_adu=column(
                        "PLANETES_FLUX", "FLUXDATA", "adu"
                    ).reshape(nf, 4, nw),
                    coherent_flux_adu=column("PLANETES_VIS", "VISDATA", "adu").reshape(
                        nf, 6, nw
                    ),
                    visibility_amplitude=column("PLANETES_VIS", "VISAMP").reshape(
                        nf, 6, nw
                    ),
                    flags=column("PLANETES_VIS", "FLAG").reshape(nf, 6, nw),
                    group_delay_um=quantity(
                        "group_delay_um",
                        column("PLANETES_VIS", "GDELAY", "m").reshape(nf, 6),
                    ),
                    differential_phase_deg=column(
                        "PLANETES_VIS", "DIFFPHASE", "deg"
                    ).reshape(nf, 6, nw),
                )
        if actual in (DarkCalibration, FlatCalibration) and meta["setup"] is not None:
            values["setup"] = DetectorSetup(**meta["setup"])
            setup = values["setup"]
            for name, key in kw.ROI_KEYS.items():
                v = getattr(setup, name)
                expected = (
                    None if v is None else v + (1 if name.startswith("origin_") else 0)
                )
                if h.get(key) != expected:
                    raise ValueError(f"Product setup and header disagree: {key}")
            key = (
                kw.DIT
                if setup.exposure_source in ("readback", "legacy-property")
                else kw.UIT
            )
            if h.get(key) != setup.exposure_s:
                raise ValueError("Product exposure and header disagree")
        product = retain_representation(actual(**values), canonical)
        validate(product)
        if fingerprint(product) != h.get(kw.HASH_KEY):
            raise ValueError("Scientific content fingerprint mismatch")
        return product


@contextmanager
def destinations(paths, *, raw=False):
    """Stage/verify all outputs, publish without overwrite, rollback only owned links."""
    paths = [Path(p).absolute() for p in paths]
    if len(set(paths)) != len(paths):
        raise ValueError("Duplicate output destination")
    for p in paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        if os.path.lexists(p):
            raise FileExistsError(f"Refusing to overwrite {p}")
    if len({p.parent for p in paths}) != 1:
        raise ValueError("Related outputs require one output directory")
    published = []
    with tempfile.TemporaryDirectory(
        prefix=".planetes-", dir=paths[0].parent
    ) as directory:
        temporary = [Path(directory) / p.name for p in paths]
        try:
            yield temporary
            for p in temporary:
                if raw:
                    read_raw_metadata(p)
                    with afits.open(p, checksum=True) as hdus:
                        _checksums(hdus)
                else:
                    read_product(p)
            for source, target in zip(temporary, paths, strict=True):
                os.link(source, target)
                published.append((target, source.stat().st_dev, source.stat().st_ino))
        except BaseException:
            for target, device, inode in reversed(published):
                try:
                    stat = target.lstat()
                    if (stat.st_dev, stat.st_ino) == (device, inode):
                        target.unlink()
                except FileNotFoundError:
                    pass
            raise


@contextmanager
def product_destination(path: Path):
    """Single-product form of the same verified publication boundary."""
    with destinations([path]) as paths:
        yield paths[0]


def write_product(path: Path, product):
    """Explicitly save one scientific product, preserving all existing destinations."""
    encoded = encode_product(product)
    with product_destination(path) as temporary:
        encoded.writeto(temporary, checksum=True)


def fill_dfs_product(path: Path, product):
    """Fill only a freshly created DFS stub with the shared payload encoding."""
    with afits.open(path) as hdus:
        if (
            len(hdus) != 1
            or hdus[0].data is not None
            or hdus[0].header.get("ESO PRO CATG") != CATEGORIES[type(product)]
            or "ESO PRO REC1 ID" not in hdus[0].header
        ):
            raise ValueError("Expected a new empty DFS product")
        encoded = encode_product(product, hdus[0].header)
    with afits.open(path, mode="update") as hdus:
        hdus[0].header = encoded[0].header
        for item in encoded[1:]:
            hdus.append(item)
        for item in hdus:
            item.add_checksum()


def check_raw_calibrations(path, dark, flat=None, *, flat_setup=None, category=None):
    """Shared formal task boundary: known DIT/ROI conflicts are scientific errors."""
    from .models import check_setup

    if not isinstance(dark, DarkCalibration):
        raise ValueError("Formal file tasks require complete DARK calibrations")
    validate(dark)
    raw = read_raw_metadata(path, category=category)
    unknown = list(check_setup(raw.setup, dark.setup, exposure=True, role="DARK"))
    geometry = flat.setup if flat is not None else flat_setup
    if flat is not None or flat_setup is not None:
        unknown.extend(check_setup(raw.setup, geometry, exposure=False, role="FLAT"))
    return tuple(sorted(set(unknown)))
