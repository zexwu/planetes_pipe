"""Science-product FITS with GRAVITY-derived vocabulary and metre-valued arrays."""

from pathlib import Path

import numpy as np
from astropy.io import fits as afits

from ..instrument import (
    ABCD_ORDER,
    BASELINES,
    BASELINE_PAIRS,
    TELESCOPES,
    DetectorSetup,
    _check_baselines,
)
from ..products import Bad, Dark, Flat, P2VM, P2VMRed, Wave, WavelengthReference
from . import keywords as kw
from .hdu import _verify_checksums, _native, _storage_array, new_fits_file
from .keywords import CATEGORIES


def _image(name, value, unit=""):
    hdu = afits.ImageHDU(_storage_array(value), name=name)
    hdu.header["BUNIT"] = unit
    return hdu


def _column(name, values, unit=""):
    """Encode one column, keeping trailing array axes in FITS order."""
    values = _storage_array(values)
    if values.dtype.kind in "US":
        width = max((len(value) for value in values), default=1)
        return afits.Column(name=name, array=values, format=f"{max(1, width)}A")
    kind = {"f": "E", "c": "C", "b": "L", "i": "K", "u": "K"}[values.dtype.kind]
    count = int(np.prod(values.shape[1:])) if values.ndim > 1 else 1
    dim = None
    if values.ndim > 1:
        dim = "(" + ",".join(map(str, values.shape[:0:-1])) + ")"
    return afits.Column(
        name=name, array=values, format=f"{count}{kind}", unit=unit or None, dim=dim
    )


def _table(name, columns):
    return afits.BinTableHDU.from_columns(columns, name=name)


def _detector_table(product):
    labels = [f"REG{i + 1:02d}" for i in range(24)]
    if isinstance(product, (P2VM, P2VMRed)):
        _check_baselines(product.baseline_outputs)
        for baseline, outputs in product.baseline_outputs.items():
            for output, phase in zip(outputs, ABCD_ORDER, strict=True):
                labels[output] = BASELINES[baseline] + "-" + phase
    return _table(kw.DETECTOR_DATA, [_column("REGNAME", np.asarray(labels))])


def encode_product(product, header=None, *, quality=None):
    """Encode an in-memory scientific product as a FITS HDU list.

    Parameters
    ----------
    product : Dark, Bad, Flat, Wave, WavelengthReference, P2VM or P2VMRed
        Product to encode. Saved DARK requires measured RMS and n_frames >= 2;
        P2VMRED requires both GD and differential phase.
    header : astropy.io.fits.Header or None, optional
        Starting primary header, copied before modification. Recipes pass the
        DFS header to retain input/parameter provenance. Raw DPR cards are removed.
    quality : mapping or None, optional
        Explicit metric -> (value, comment) entries from measure_quality.
        None writes no QC. This function does not calculate quality statistics.

    Returns
    -------
    hdus : astropy.io.fits.HDUList
        Encoded extensions and ESO product metadata, ready for the writer.
        Floating payloads use float32/complex64; valid extraction pixel coordinates
        become one-based on disk. No file is written and no checksum is added here.

    Notes
    -----
    Wavelength and GD remain metres; response phase is radians and differential
    phase is degrees. FITS encoding does not perform scientific calibration.
    """
    if type(product) not in CATEGORIES:
        raise TypeError(
            "Only complete DARK/BAD/calibration/science objects can be saved"
        )
    hdr = afits.Header() if header is None else header.copy()
    for key in list(hdr):
        if key.startswith("ESO DPR "):
            del hdr[key]
    hdr["INSTRUME"] = "PLANETES"
    hdr["HIERARCH " + kw.PRO_CATEGORY] = CATEGORIES[type(product)]
    hdr["HIERARCH " + kw.PRO_TYPE] = (
        "STATIC" if isinstance(product, WavelengthReference) else "REDUCED"
    )
    hdr["HIERARCH " + kw.PRO_SCIENCE] = isinstance(product, P2VMRed)
    setup = getattr(product, "setup", None)
    if setup is not None:
        for name, key in kw.ROI_KEYS.items():
            v = getattr(setup, name)
            if v is not None:
                hdr["HIERARCH " + key] = v + (1 if name.startswith("origin_") else 0)
        if setup.detector is not None:
            hdr["HIERARCH " + kw.DETECTOR_ID] = setup.detector
        if setup.exposure_s is not None:
            hdr["HIERARCH " + kw.DIT] = (setup.exposure_s, "[s] Integration time")

    def one_based_pixels(values):
        return (
            np.where(values < 0, values, values + 1)
            if values.dtype.kind in "iu"
            else values + 1
        )

    hdus = [afits.PrimaryHDU(header=hdr)]
    if isinstance(product, Dark):
        if (
            product.temporal_rms_adu is None
            or product.n_frames is None
            or product.n_frames < 2
            or product.mean_adu.shape != product.temporal_rms_adu.shape
        ):
            raise ValueError("Saved DARK requires measured RMS and at least two frames")
        hdu = hdus[0]
        hdu.header["NFRAMES"] = product.n_frames
        hdus += [
            _image(kw.RAW_IMAGE, product.mean_adu, "adu"),
            _image(kw.DARK_RMS, product.temporal_rms_adu, "adu"),
        ]
    elif isinstance(product, Bad):
        hdus.append(_image(kw.RAW_IMAGE, product.bad_pixel_flags))
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
                    _column(f"DATA{i + 1}", row[None], "m")
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
            _table(kw.WAVELENGTH, [_column("WAVELENGTH", product.wavelength_m, "m")]),
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
            # fmt: off
            for name, ext, unit in [
                ("matrix",            kw.P2VM_INVERSE,      ""),
                ("opd_m",             kw.OPD,               "m"),
                ("group_delay_m",     kw.GROUP_DELAY,       "m"),
                ("ellipse_phase_rad", kw.ELLIPSE_PHASE,     "rad"),
                ("ellipse_amplitude", kw.ELLIPSE_AMPLITUDE, ""),
                ("fit_rms_adu",       kw.P2VM_FIT_RMS,      "adu"),
            ]:
                if getattr(product, name) is not None:
                    hdus.append(_image(ext, getattr(product, name), unit))
            # fmt: on
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
                [[int(TELESCOPES[t]) for t in BASELINE_PAIRS[b]] for b in range(6)]
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
    if quality is not None:
        for metric, (number, comment) in quality.items():
            hdus[0].header["HIERARCH " + kw.QC_KEYWORDS[metric]] = (number, comment)
            if metric in kw.QC_UM_KEYWORDS:
                key = kw.QC_UM_KEYWORDS[metric]
                hdus[0].header["HIERARCH " + key] = (
                    number * 1e6,
                    comment.replace("[m]", "[um]"),
                )
    return afits.HDUList(hdus)


def read_product(path: Path, cls=None):
    """Read a PLANETES FITS product and check its stored identity, units and checksums.

    Parameters
    ----------
    path : Path
        Product FITS file with valid CHECKSUM and DATASUM in every HDU.
    cls : product class or None, optional
        Expected class, such as Dark or P2VM. A mismatch raises ValueError.
        None chooses the class from ESO PRO CATG.

    Returns
    -------
    product : Dark, Bad, Flat, Wave, WavelengthReference, P2VM or P2VMRed
        Decoded product. Floating arrays are native float64/complex128, integers
        and flags retain their types, and extraction coordinates are zero-based.

    Notes
    -----
    Promoting stored float32 arrays does not recover discarded precision.
    This reader checks file representation and physical identities; numerical
    entry points check the geometry and wavelength coverage needed for calculation.
    """
    with afits.open(path) as hdus:
        hdr = hdus[0].header
        if hdr.get("INSTRUME") != "PLANETES":
            raise ValueError("Expected a PLANETES product")
        actual = {v: k for k, v in CATEGORIES.items()}.get(hdr.get(kw.PRO_CATEGORY))
        if actual is None or (cls is not None and actual is not cls):
            raise ValueError("Unexpected product category")
        _verify_checksums(hdus)
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
                n_frames=hdr["NFRAMES"],
            )
        elif actual is Bad:
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
                [column(kw.WAVE_DATA, f"DATA{i + 1}", "m")[0] for i in range(24)]
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
                # fmt: off
                for name, ext, unit in [
                    ("matrix",            kw.P2VM_INVERSE,      ""),
                    ("opd_m",             kw.OPD,               "m"),
                    ("group_delay_m",     kw.GROUP_DELAY,       "m"),
                    ("ellipse_phase_rad", kw.ELLIPSE_PHASE,     "rad"),
                    ("ellipse_amplitude", kw.ELLIPSE_AMPLITUDE, ""),
                    ("fit_rms_adu",       kw.P2VM_FIT_RMS,      "adu"),
                ]:
                    if ext in hdus:
                        values[name] = image(ext, unit)
                # fmt: on
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
        if actual in (Dark, Flat):
            setup = {name: hdr.get(key) for name, key in kw.ROI_KEYS.items()}
            for name in ("origin_x", "origin_y"):
                if setup[name] is not None:
                    setup[name] -= 1  # FITS origins are one-based.
            setup["exposure_s"] = hdr.get(kw.DIT)
            setup["detector"] = hdr.get(kw.DETECTOR_ID)
            if any(value is not None for value in setup.values()):
                values["setup"] = DetectorSetup(**setup)
        return actual(**values)


def product_filename(reference: Path, product) -> str:
    """Build a GRAVITY-style product name from its explicitly selected raw input.

    Parameters
    ----------
    reference : Path
        Reference raw filename. Its complete stem, including a recorded timestamp
        or an explicit UNKNOWN marker, is retained without interpretation.
    product : Dark, Bad, Flat, Wave, WavelengthReference, P2VM or P2VMRed
        Product whose ESO PRO CATG defines the suffix.

    Returns
    -------
    filename : str
        <raw stem>_<lowercase category without underscores>.fits, without a
        directory. For example, PLANETES.2026-02-05T210609_singlescip2vmred.fits.

    Notes
    -----
    This uses the input name, not the current time or file modification time.
    It does not create DATE-OBS or infer a timezone. For multiple-input products,
    callers explicitly choose the reference; SOF ordering need not define it.
    """
    category = CATEGORIES[type(product)].lower().replace("_", "")
    return f"{reference.stem}_{category}.fits"


def write_product(path: Path, product, *, quality=None):
    """Save one scientific product without replacing any existing destination.

    Parameters
    ----------
    path : Path
        New output FITS path. Parent directories are created if necessary;
        an existing destination raises OSError and is left intact.
    product : Dark, Bad, Flat, Wave, WavelengthReference, P2VM or P2VMRed
        Complete product accepted by encode_product; see its unit/storage contract.
    quality : mapping or None, optional
        Explicit metric -> (value, comment) entries, normally from measure_quality.
        None omits QC; the writer does not compute it.

    Notes
    -----
    Write checksummed FITS to a temporary file in the destination directory, then
    publish it with a non-overwriting hard link. Each file is independent. Plain
    Python output does not record input paths or recipe parameters; ESO recipes
    use the separate DFS writer for those records. Returns None.
    """
    encoded = encode_product(product, quality=quality)
    with new_fits_file(path) as temporary:
        encoded.writeto(temporary, checksum=True)
