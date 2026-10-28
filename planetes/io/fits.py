"""Science-product FITS with GRAVITY-derived vocabulary and metre-valued arrays."""

from dataclasses import asdict
from pathlib import Path

import numpy as np
from astropy.io import fits as afits

from .. import __version__
from ..instrument import (
    ABCD_ORDER,
    BASELINES,
    TELESCOPES,
    DetectorSetup,
    _check_baselines,
)
from ..products import (
    BadPixelMap,
    Dark,
    Flat,
    P2VM,
    P2VMRed,
    Wave,
    WavelengthReference,
)
from . import keywords as kw
from .hdu import _verify_checksums, _json_hdu, _native, _read_json
from .keywords import CATEGORIES
from .publication import publish_outputs


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


def _detector_table(product):
    labels = [f"REG{i+1:02d}" for i in range(24)]
    if isinstance(product, (P2VM, P2VMRed)):
        _check_baselines(product.baseline_outputs, product.baseline_telescopes)
        for baseline, outputs in product.baseline_outputs.items():
            for output, phase in zip(outputs, ABCD_ORDER, strict=True):
                labels[output] = BASELINES[baseline] + "-" + phase
    return _table(kw.DETECTOR_DATA, [_column("REGNAME", np.asarray(labels))])


def encode_product(product, header=None, *, quality=None):
    """Encode a product; FITS units and ESO keywords stay in this I/O layer."""
    if type(product) not in CATEGORIES:
        raise TypeError(
            "Only complete DARK/BAD/calibration/science objects can be saved"
        )
    h = afits.Header() if header is None else header.copy()
    for key in list(h):
        if key.startswith("ESO DPR ") or key == "PLANETES CONTENT SHA256":
            del h[key]
    h["INSTRUME"] = "PLANETES"
    h["HIERARCH " + kw.FORMAT_KEY] = kw.PRODUCT_FORMAT
    h["HIERARCH ESO PRO CATG"] = CATEGORIES[type(product)]
    h["HIERARCH ESO PRO TYPE"] = (
        "STATIC" if isinstance(product, WavelengthReference) else "REDUCED"
    )
    h["HIERARCH ESO PRO SCIENCE"] = isinstance(product, P2VMRed)
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

    def one_based_pixels(values):
        return (
            np.where(values < 0, values, values + 1)
            if values.dtype.kind in "iu"
            else values + 1
        )

    hdus = [afits.PrimaryHDU(header=h)]
    if isinstance(product, Dark):
        if (
            product.n_frames < 2
            or product.mean_adu.shape != product.temporal_rms_adu.shape
        ):
            raise ValueError("Saved DARK requires measured RMS and at least two frames")
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
    elif isinstance(product, Flat):
        hdus += [
            _detector_table(product),
            _image(kw.FLAT_IMAGE, product.flat_map_adu, "adu"),
        ]
        columns = [
            _column("REGNAME", hdus[1].data["REGNAME"]),
            _column("CENTERY", one_based_pixels(product.trace_y_pixels), "pixel"),
        ]
        if product.profile_sigma_pixels is not None:
            columns.append(_column("WIDTH", product.profile_sigma_pixels, "pixel"))
        hdus += [
            _table(kw.PROFILE_PARAMS, columns),
            _table(
                kw.SPARSE_PROFILE,
                [
                    _column(
                        "PIXEL_X", one_based_pixels(product.profile_x_pixels), "pixel"
                    ),
                    _column(
                        "PIXEL_Y", one_based_pixels(product.profile_y_pixels), "pixel"
                    ),
                ],
            ),
        ]
        if product.telescope_flux_adu is not None:
            hdus.append(_image(kw.FLAT_FLUX, product.telescope_flux_adu, "adu"))
    elif isinstance(product, Wave):
        hdus += [
            _detector_table(product),
            _table(
                kw.WAVE_DATA,
                [
                    _column(f"DATA{i+1}", row[None], "m")
                    for i, row in enumerate(product.wavelength_map_m)
                ],
            ),
        ]
        if product.fit_residual_m is not None:
            hdus.append(
                _table(
                    kw.WAVE_FIT,
                    [
                        _column("RESIDUAL", product.fit_residual_m, "m"),
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
                    _column("WAVELENGTH", product.wavelength_m, "m"),
                    _column("ORIGIN", np.asarray(product.origin)),
                ],
            )
        )
    else:
        hdus += [
            _detector_table(product),
            _table(
                kw.WAVELENGTH,
                [_column("WAVELENGTH", product.wavelength_m, "m")],
            ),
        ]
        if isinstance(product, P2VM):
            columns = [
                _column("REGNAME", hdus[1].data["REGNAME"]),
                _column("TRANSMISSION", product.transmission),
                _column("COHERENCE", product.coherence_amplitude),
                _column("PHASE", product.coherence_phase_rad, "rad"),
            ]
            if product.dc_term_adu is not None:
                columns.append(_column("C_MATRIX", product.dc_term_adu, "adu"))
            hdus.append(_table(kw.P2VM_DATA, columns))
            for name, ext, unit in [
                ("matrix", kw.P2VM_INVERSE, ""),
                ("opd_m", kw.OPD, "m"),
                ("group_delay_m", kw.GROUP_DELAY, "m"),
                ("ellipse_phase_rad", kw.ELLIPSE_PHASE, "rad"),
                ("ellipse_amplitude", kw.ELLIPSE_AMPLITUDE, ""),
                ("fit_rms_adu", kw.P2VM_FIT_RMS, "adu"),
            ]:
                if getattr(product, name) is not None:
                    hdus.append(_image(ext, getattr(product, name), unit))
        else:
            if product.group_delay_m is None or product.differential_phase_deg is None:
                raise ValueError("Saved P2VMRED requires GD and differential phase")
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
                        _column("GDELAY", product.group_delay_m.reshape(-1), "m"),
                        _column(
                            "DIFFPHASE",
                            product.differential_phase_deg.reshape(-1, nw),
                            "deg",
                        ),
                    ],
                )
            )
    # The raw/bench FITS dictionary is unfinished. Store only useful setup
    # information here; ESO DFS owns input-file and recipe provenance.
    metadata = {
        "software_version": __version__,
        "setup": asdict(product.setup)
        if getattr(product, "setup", None) is not None
        else None,
    }
    if quality is not None:
        measurements, missing = quality
        for metric, (number, comment) in measurements.items():
            hdus[0].header["HIERARCH " + kw.QC_KEYWORDS[metric]] = (number, comment)
            if metric in (
                "wavelength_min_m",
                "wavelength_max_m",
                "wavelength_residual_std_m",
            ):
                key = kw.QC_KEYWORDS[metric].replace(" SC", " UM SC")
                hdus[0].header["HIERARCH " + key] = (
                    number * 1e6,
                    comment.replace("[m]", "[um]"),
                )
        metadata["qc_status"] = {
            kw.QC_STATUS_KEYS.get(metric, kw.QC_KEYWORDS.get(metric, metric)): reason
            for metric, reason in missing.items()
        }
    hdus.append(_json_hdu(kw.META_EXT, metadata))
    return afits.HDUList(hdus)


def read_product(path: Path, cls=None):
    """Read a PLANETES FITS product, retaining physical units and identities."""
    with afits.open(path, checksum=True) as hdus:
        h = hdus[0].header
        if (
            h.get(kw.FORMAT_KEY) not in (4, kw.PRODUCT_FORMAT)
            or h.get("INSTRUME") != "PLANETES"
        ):
            raise ValueError("Unsupported PLANETES product format")
        actual = {v: k for k, v in CATEGORIES.items()}.get(h.get(kw.PRO_CATEGORY))
        if actual is None or (cls is not None and actual is not cls):
            raise ValueError("Unexpected product category")
        _verify_checksums(hdus)
        meta = _read_json(hdus, kw.META_EXT)
        values = {}

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

        def zero_based_pixels(values):
            return (
                np.where(values < 0, values, values - 1)
                if values.dtype.kind in "iu"
                else values - 1
            )

        if actual is Dark:
            values.update(
                mean_adu=image(kw.RAW_IMAGE, "adu"),
                temporal_rms_adu=image(kw.DARK_RMS, "adu"),
                n_frames=h["NFRAMES"],
            )
        elif actual is BadPixelMap:
            if hdus[kw.RAW_IMAGE].header.get(kw.BAD_BITS) != "hot=1,dead=2,noisy=4":
                raise ValueError("Invalid BAD flag definition")
            values["bad_pixel_flags"] = image(kw.RAW_IMAGE, "")
        elif actual is Flat:
            values.update(
                flat_map_adu=image(kw.FLAT_IMAGE, "adu"),
                trace_y_pixels=zero_based_pixels(
                    column(kw.PROFILE_PARAMS, "CENTERY", "pixel")
                ),
                profile_x_pixels=zero_based_pixels(
                    column(kw.SPARSE_PROFILE, "PIXEL_X", "pixel")
                ),
                profile_y_pixels=zero_based_pixels(
                    column(kw.SPARSE_PROFILE, "PIXEL_Y", "pixel")
                ),
            )
            if "WIDTH" in hdus[kw.PROFILE_PARAMS].columns.names:
                values["profile_sigma_pixels"] = column(
                    kw.PROFILE_PARAMS, "WIDTH", "pixel"
                )
            if kw.FLAT_FLUX in hdus:
                values["telescope_flux_adu"] = image(kw.FLAT_FLUX, "adu")
        elif actual is Wave:
            values["wavelength_map_m"] = np.stack(
                [column(kw.WAVE_DATA, f"DATA{i+1}", "m")[0] for i in range(24)]
            )
            if kw.WAVE_FIT in hdus:
                values.update(
                    fit_residual_m=column(kw.WAVE_FIT, "RESIDUAL", "m"),
                    fit_x_pixels=column(kw.WAVE_FIT, "X", "pixel"),
                    fit_y_pixels=column(kw.WAVE_FIT, "Y", "pixel"),
                )
        elif actual is WavelengthReference:
            values.update(
                wavelength_m=column(kw.LINE_LIST, "WAVELENGTH", "m"),
                origin=tuple(str(v).strip() for v in hdus[kw.LINE_LIST].data["ORIGIN"]),
            )
        else:
            values["wavelength_m"] = column(kw.WAVELENGTH, "WAVELENGTH", "m")
            labels = list(hdus[kw.DETECTOR_DATA].data["REGNAME"])
            values["baseline_outputs"] = {
                i: [labels.index(name + "-" + a) for a in ABCD_ORDER]
                for i, name in enumerate(BASELINES)
            }
            values["baseline_telescopes"] = {
                i: (TELESCOPES.index(name[0]), TELESCOPES.index(name[1]))
                for i, name in enumerate(BASELINES)
            }
            if actual is P2VM:
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
                    ("matrix", kw.P2VM_INVERSE, ""),
                    ("opd_m", kw.OPD, "m"),
                    ("group_delay_m", kw.GROUP_DELAY, "m"),
                    ("ellipse_phase_rad", kw.ELLIPSE_PHASE, "rad"),
                    ("ellipse_amplitude", kw.ELLIPSE_AMPLITUDE, ""),
                    ("fit_rms_adu", kw.P2VM_FIT_RMS, "adu"),
                ]:
                    if ext in hdus:
                        values[name] = image(ext, unit)
            else:
                nw = len(values["wavelength_m"])
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
                    group_delay_m=column(kw.SCIENCE_VISIBILITY, "GDELAY", "m").reshape(
                        nf, 6
                    ),
                    differential_phase_deg=column(
                        kw.SCIENCE_VISIBILITY, "DIFFPHASE", "deg"
                    ).reshape(nf, 6, nw),
                )
        if actual in (Dark, Flat) and meta["setup"] is not None:
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
        return actual(**values)


def write_product(path: Path, product, *, include_qc: bool = True):
    """Save a product with optional QC; preserve existing output files."""
    from ..qc import measure_quality

    quality = measure_quality(product) if include_qc else None
    encoded = encode_product(product, quality=quality)
    with publish_outputs([path], verify=read_product) as temporary:
        encoded.writeto(temporary[0], checksum=True)
