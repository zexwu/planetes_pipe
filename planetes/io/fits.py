"""Raw and science FITS with GRAVITY-derived vocabulary; no execution framework."""

from dataclasses import asdict
from pathlib import Path

import numpy as np
from astropy.io import fits as afits

from .. import __version__
from ..identity import canonical_quantities, fingerprint, retain_representation
from ..instrument import ABCD_ORDER, BASELINES, TELESCOPES, DetectorSetup
from ..products import (
    BadPixelMap,
    DarkCalibration,
    FlatCalibration,
    P2VMCalibration,
    Provenance,
    ReducedObservables,
    WavelengthCalibration,
    WavelengthReference,
    validate,
)
from . import keywords as kw
from .hdu import _checksums, _json_hdu, _native, _read_json
from .keywords import CATEGORIES
from .publication import destinations


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
        image.header["HIERARCH " + kw.BAD_BITS] = "hot=1,dead=2,noisy=4"
        hdus.append(image)
    elif isinstance(product, FlatCalibration):
        hdus += [
            _detector(product),
            _image(kw.FLAT_IMAGE, product.flat_map_adu, "adu"),
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
                kw.SPARSE_PROFILE,
                [
                    _column("PIXEL_X", value("profile_x_pixels"), "pixel"),
                    _column("PIXEL_Y", value("profile_y_pixels"), "pixel"),
                ],
            ),
        ]
        if product.telescope_flux_adu.size:
            hdus.append(_image(kw.FLAT_FLUX, product.telescope_flux_adu, "adu"))
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
                    kw.WAVE_FIT,
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
                kw.LINE_LIST,
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
                kw.WAVELENGTH,
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
                ("p2vm", kw.P2VM_INVERSE, ""),
                ("opd_um", kw.OPD, "m"),
                ("group_delay_um", kw.GROUP_DELAY, "m"),
                ("ellipse_phase_rad", kw.ELLIPSE_PHASE, "rad"),
                ("ellipse_amplitude", kw.ELLIPSE_AMPLITUDE, ""),
                ("fit_rms_adu", kw.P2VM_FIT_RMS, "adu"),
            ]:
                if value(name).size:
                    hdus.append(_image(ext, value(name), unit))
        else:
            nf, _, nw = product.photometric_flux_adu.shape
            hdus += [
                _table(
                    kw.SCIENCE_FLUX,
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
                    kw.SCIENCE_VISIBILITY,
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
        "software_version": __version__,
        "provenance": info,
        "setup": asdict(product.setup)
        if getattr(product, "setup", None) is not None
        else None,
    }
    hdus.append(_json_hdu(kw.META_EXT, metadata))
    # The same QC measurements are written for Python and CPL products.
    from ..qc import measure_quality

    quality, missing = measure_quality(product)
    for metric, (number, comment) in quality.items():
        hdus[0].header["HIERARCH " + kw.QC_KEYWORDS[metric]] = (number, comment)
    metadata["provenance"]["qc_status"] = {
        **info["qc_status"],
        **{
            kw.QC_STATUS_KEYS.get(metric, kw.QC_KEYWORDS.get(metric, metric)): reason
            for metric, reason in missing.items()
        },
    }
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
        actual = {v: k for k, v in CATEGORIES.items()}.get(h.get(kw.PRO_CATEGORY))
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
            if table.columns[name].format.endswith("L"):
                raw = table.data._get_raw_data()[name]
                if not np.isin(raw, [ord("T"), ord("F")]).all():
                    raise ValueError("Invalid FITS logical flag encoding")
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
            if hdus[kw.RAW_IMAGE].header.get(kw.BAD_BITS) != "hot=1,dead=2,noisy=4":
                raise ValueError("Invalid BAD flag definition")
            values["bad_pixel_flags"] = image(kw.RAW_IMAGE, "")
        elif actual is FlatCalibration:
            values.update(
                flat_map_adu=image(kw.FLAT_IMAGE, "adu"),
                trace_y_pixels=quantity(
                    "trace_y_pixels",
                    column(kw.PROFILE_PARAMS, "CENTERY", "pixel"),
                    offset=1,
                ),
                profile_x_pixels=quantity(
                    "profile_x_pixels",
                    column(kw.SPARSE_PROFILE, "PIXEL_X", "pixel"),
                    offset=1,
                ),
                profile_y_pixels=quantity(
                    "profile_y_pixels",
                    column(kw.SPARSE_PROFILE, "PIXEL_Y", "pixel"),
                    offset=1,
                ),
            )
            if "WIDTH" in hdus[kw.PROFILE_PARAMS].columns.names:
                values["profile_sigma_pixels"] = column(
                    kw.PROFILE_PARAMS, "WIDTH", "pixel"
                )
            if kw.FLAT_FLUX in hdus:
                values["telescope_flux_adu"] = image(kw.FLAT_FLUX, "adu")
        elif actual is WavelengthCalibration:
            values["wavelength_map_um"] = quantity(
                "wavelength_map_um",
                np.stack(
                    [column(kw.WAVE_DATA, f"DATA{i+1}", "m")[0] for i in range(24)]
                ),
            )
            if kw.WAVE_FIT in hdus:
                values.update(
                    fit_residual_um=quantity(
                        "fit_residual_um", column(kw.WAVE_FIT, "RESIDUAL", "m")
                    ),
                    fit_x_pixels=column(kw.WAVE_FIT, "X", "pixel"),
                    fit_y_pixels=column(kw.WAVE_FIT, "Y", "pixel"),
                )
        elif actual is WavelengthReference:
            values.update(
                wavelength_um=quantity(
                    "wavelength_um", column(kw.LINE_LIST, "WAVELENGTH", "m")
                ),
                origin=tuple(str(v).strip() for v in hdus[kw.LINE_LIST].data["ORIGIN"]),
            )
        else:
            values["wavelength_um"] = quantity(
                "wavelength_um", column(kw.WAVELENGTH, "WAVELENGTH", "m")
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
                    ("p2vm", kw.P2VM_INVERSE, ""),
                    ("opd_um", kw.OPD, "m"),
                    ("group_delay_um", kw.GROUP_DELAY, "m"),
                    ("ellipse_phase_rad", kw.ELLIPSE_PHASE, "rad"),
                    ("ellipse_amplitude", kw.ELLIPSE_AMPLITUDE, ""),
                    ("fit_rms_adu", kw.P2VM_FIT_RMS, "adu"),
                ]:
                    if ext in hdus:
                        v = image(ext, unit)
                        values[name] = quantity(name, v) if unit == "m" else v
            else:
                nw = len(values["wavelength_um"])
                nf = len(hdus[kw.SCIENCE_FLUX].data) // 4
                for ext, count in [(kw.SCIENCE_FLUX, 4), (kw.SCIENCE_VISIBILITY, 6)]:
                    if not np.array_equal(
                        column(ext, "FRAME"), np.repeat(np.arange(1, nf + 1), count)
                    ):
                        raise ValueError("Invalid frame row order")
                if not np.array_equal(
                    column(kw.SCIENCE_FLUX, "STA_INDEX"),
                    np.tile(np.asarray(TELESCOPES, dtype=int), nf),
                ):
                    raise ValueError("Invalid telescope identity")
                expected = np.tile(
                    np.array([[int(v) for v in name] for name in BASELINES]), (nf, 1)
                )
                if not np.array_equal(
                    column(kw.SCIENCE_VISIBILITY, "STA_INDEX"), expected
                ):
                    raise ValueError("Invalid baseline identity")
                values.update(
                    photometric_flux_adu=column(
                        kw.SCIENCE_FLUX, "FLUXDATA", "adu"
                    ).reshape(nf, 4, nw),
                    coherent_flux_adu=column(
                        kw.SCIENCE_VISIBILITY, "VISDATA", "adu"
                    ).reshape(nf, 6, nw),
                    visibility_amplitude=column(
                        kw.SCIENCE_VISIBILITY, "VISAMP"
                    ).reshape(nf, 6, nw),
                    flags=column(kw.SCIENCE_VISIBILITY, "FLAG").reshape(nf, 6, nw),
                    group_delay_um=quantity(
                        "group_delay_um",
                        column(kw.SCIENCE_VISIBILITY, "GDELAY", "m").reshape(nf, 6),
                    ),
                    differential_phase_deg=column(
                        kw.SCIENCE_VISIBILITY, "DIFFPHASE", "deg"
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


def write_product(path: Path, product):
    """Explicitly save one scientific product, preserving all existing destinations."""
    encoded = encode_product(product)
    with destinations([path], verify=read_product) as temporary:
        encoded.writeto(temporary[0], checksum=True)
