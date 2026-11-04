"""FITS files: raw cubes, product read/write, QC cards and provenance.

Each product module (dark, flat, wave, p2vm, science) only describes its own
extensions; everything else about files is here. No CPL is used.
"""

from contextlib import contextmanager
from dataclasses import fields
from datetime import UTC, datetime
from pathlib import Path

import astropy
import numpy as np
from astropy.io import fits as afits

from planetes_pipe import __version__
from planetes_pipe.instrument import DetectorSetup, Raw, layout

# Raw SOF tag -> ESO DPR TYPE of the raw file.
RAW_TYPES = {
    "DARK_RAW": "DARK",
    "FLAT_RAW": "FLAT",
    "P2VM_RAW": "P2VM",
    "WAVELAMP_RAW": "WAVE,LAMP",
    "SINGLE_SCI_RAW": "OBJECT,SINGLE",
}


def is_raw(tag: str) -> bool:
    """RAW (not calibration) input in the DICD and CPL frame-group sense."""
    return tag.endswith("_RAW") or tag == "PHASE_REFERENCE"


# Relative round-to-nearest error bound for the on-disk real component.
STORAGE_RTOL = np.finfo(np.float32).eps / 2


def _native(values):
    """Load floating FITS data at computation precision; preserve integer/flag types."""
    dtype = {"f": np.float64, "c": np.complex128}.get(
        values.dtype.kind, values.dtype.newbyteorder("=")
    )
    return np.array(values, dtype=dtype, copy=True)


def as_compute_array(values):
    """Convert a raw/Python array to native calculation precision at its input boundary.

    Parameters
    ----------
    values : array-like
        Real or complex samples; units and array shape are unchanged.

    Returns
    -------
    array : ndarray
        float64 for real input, complex128 for complex input. A copy is made only
        when required, so the result may share memory with a compatible input.
    """
    return np.asarray(
        values, dtype=np.complex128 if np.iscomplexobj(values) else np.float64
    )


def _storage_array(values):
    """Store floating payloads as float32/complex64; retain integers, flags and labels."""
    values = np.asarray(values)
    dtype = {"f": np.float32, "c": np.complex64}.get(values.dtype.kind)
    return values if dtype is None else values.astype(dtype, copy=False)


def _verify_checksums(hdus):
    for h in hdus:
        if h.verify_checksum() != 1 or h.verify_datasum() != 1:
            raise ValueError("Invalid or missing FITS checksum")


@contextmanager


def set_card(header, key: str, value, comment: str = "") -> None:
    """Set a card, trimming the comment so the card fits 80 columns untruncated."""
    name = "HIERARCH " + key if " " in key or len(key) > 8 else key
    header[name] = value
    used = len(header.cards[name].image.rstrip())
    room = 80 - used - 3
    header[name] = (value, comment[:room] if room > 0 else "")


# ---------------------------------------------------------------- raw files


def read_raw(path: Path, category: str, *, hdu: int | str = "IMAGING_DATA_SC") -> Raw:
    """Open a raw cube (memory-mapped) with its shutters and detector setup.

    Parameters
    ----------
    path : Path
        Raw FITS file containing a (frame, y, x) detector cube in ADU.
    category : str
        Input role chosen by the caller, e.g. DARK_RAW or FLAT_RAW.
    hdu : int or str, optional
        Image HDU; default IMAGING_DATA_SC, 0 for primary-HDU data. No search.

    Notes
    -----
    Width/height come from the cube; FITS one-based ROI origins become zero-based.
    The file is read only. Frames are read from disk as they are used.
    """
    with afits.open(path, memmap=True) as hdus:
        header = hdus[0].header
        cube = hdus[hdu].data
    setup = read_setup(header) or DetectorSetup()
    setup = DetectorSetup(**{
        **{f.name: getattr(setup, f.name) for f in fields(setup)},
        "height": cube.shape[1],
        "width": cube.shape[2],
    })
    return Raw(cube, category, read_shutters(header), setup)


def find_raw(directory: Path, category: str) -> list[Path]:
    """Raw files of one category in a directory, by ESO DPR TYPE (not filename)."""
    return [path for path in sorted(Path(directory).glob("*.fits"))
            if (header := afits.getheader(path)).get("ESO DPR TYPE") == RAW_TYPES[category]
            and "ESO PRO CATG" not in header]


def read_setup(hdr) -> DetectorSetup | None:
    """Decode detector setup; FITS origins are one-based, Python origins zero-based."""
    x, y = hdr.get("ESO DET WIN1 STRX"), hdr.get("ESO DET WIN1 STRY")
    setup = DetectorSetup(
        origin_x=None if x is None else x - 1,
        origin_y=None if y is None else y - 1,
        width=hdr.get("ESO DET WIN1 NX"),
        height=hdr.get("ESO DET WIN1 NY"),
        binning_x=hdr.get("ESO DET WIN1 BINX"),
        binning_y=hdr.get("ESO DET WIN1 BINY"),
        exposure_s=hdr.get("ESO DET WIN1 DIT1", hdr.get("ESO DET WIN1 UIT1")),
        detector=hdr.get("ESO DET ID"),
    )
    return setup if any(getattr(setup, f.name) is not None for f in fields(setup)) else None


def write_setup(header, setup: DetectorSetup):
    """Write known detector settings, converting Python ROI origins to FITS."""
    for key, value in (
        ("ESO DET WIN1 STRX", None if setup.origin_x is None else setup.origin_x + 1),
        ("ESO DET WIN1 STRY", None if setup.origin_y is None else setup.origin_y + 1),
        ("ESO DET WIN1 NX", setup.width),
        ("ESO DET WIN1 NY", setup.height),
        ("ESO DET WIN1 BINX", setup.binning_x),
        ("ESO DET WIN1 BINY", setup.binning_y),
    ):
        if value is not None:
            header["HIERARCH " + key] = value
    if setup.detector is not None:
        header["HIERARCH ESO DET ID"] = setup.detector
    if setup.exposure_s is not None:
        header["HIERARCH ESO DET WIN1 DIT1"] = (
            setup.exposure_s,
            "[s] Integration time",
        )


def read_shutters(header):
    """Shutter states in physical T1..T4 order.

    Files written before 2026-11 use the non-ESO 'PLANETES INS SHUTn ST' names.
    """
    return tuple(header.get(f"ESO INS SHUT{n} ST", header.get(f"PLANETES INS SHUT{n} ST"))
                 for n in range(1, 5))


def write_shutters(header, shutters) -> None:
    """Write shutter states in physical T1..T4 order."""
    for n, state in enumerate(shutters, 1):
        header[f"HIERARCH ESO INS SHUT{n} ST"] = state


# ---------------------------------------------------------------- extensions


def image_hdu(name, value, unit=""):
    hdu = afits.ImageHDU(_storage_array(value), name=name)
    hdu.header["BUNIT"] = unit
    return hdu


def column(name, values, unit=""):
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


def table(name, columns):
    return afits.BinTableHDU.from_columns(columns, name=name)


def detector_table(baseline_outputs=None):
    labels = [f"REG{i + 1:02d}" for i in range(24)]
    if baseline_outputs is not None:
        for baseline, outputs in baseline_outputs.items():
            for output, phase in zip(outputs, layout.abcd_order, strict=True):
                labels[output] = layout.baselines[baseline] + "-" + phase
    return table("IMAGING_DETECTOR_SC", [column("REGNAME", np.asarray(labels))])


def baseline_outputs(labels) -> dict[int, list[int]]:
    """Decode IMAGING_DETECTOR_SC REGNAME labels ('43-A', ...) to baseline -> A,C,B,D outputs."""
    labels = list(labels)
    return {i: [labels.index(name + "-" + phase) for phase in layout.abcd_order]
            for i, name in enumerate(layout.baselines)}


def read_image(hdus, ext):
    """An image extension at computation precision."""
    return _native(hdus[ext].data)


def read_column(hdus, ext, name):
    """A table column at computation precision."""
    return _native(hdus[ext].data[name])


def one_based_pixels(values):
    """Convert valid pixel coordinates to FITS one-based indices."""
    return (
        np.where(values < 0, values, values + 1)
        if values.dtype.kind in "iu"
        else values + 1
    )


def zero_based_pixels(values):
    """Restore zero-based pixel coordinates, preserving negative sentinels."""
    return (
        np.where(values < 0, values, values - 1)
        if values.dtype.kind in "iu"
        else values - 1
    )


# ---------------------------------------------------------------- products


def write_qc(header, values: dict | None) -> None:
    """Write {keyword: (value, comment)} from a product's qc(); unmeasured values are omitted."""
    for key, (number, comment) in (values or {}).items():
        if number is not None and np.isfinite(number):
            set_card(header, key, number, comment)


def save_product(path: Path, category: str, extensions: list, *, header=None,
                 qc: dict | None = None, setup: DetectorSetup | None = None) -> None:
    """Write a product: primary header, QC, extensions and checksums. Existing files raise OSError.

    header is a provenance() header or None. DPR keywords are removed; PRO
    CATG/TYPE/SCIENCE, PIPEFILE and the detector setup are written.
    """
    hdr = afits.Header() if header is None else header.copy()
    for key in list(hdr):
        if key.startswith("ESO DPR "):
            del hdr[key]
    hdr["INSTRUME"] = "PLANETES"
    hdr["HIERARCH ESO PRO CATG"] = category
    hdr["HIERARCH ESO PRO TYPE"] = "REDUCED"
    hdr["HIERARCH ESO PRO SCIENCE"] = category == "SINGLE_SCI_P2VMRED"
    set_card(hdr, "PIPEFILE", Path(path).name, "Filename of data product")
    if setup is not None:
        write_setup(hdr, setup)
    write_qc(hdr, qc)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    afits.HDUList([afits.PrimaryHDU(header=hdr), *extensions]).writeto(path, checksum=True)


@contextmanager
def open_product(path: Path, category: str):
    """Open a product, checking its category and checksums."""
    with afits.open(path) as hdus:
        if hdus[0].header.get("ESO PRO CATG") != category:
            raise ValueError(f"Expected a {category} product: {path}")
        _verify_checksums(hdus)
        yield hdus


def product_filename(reference: Path, category: str) -> str:
    """<reference raw stem>_<category without underscores>.fits, e.g. ..._singlescip2vmred.fits.

    The reference stem is kept as is (timestamp or UNKNOWN marker); no time is invented.
    """
    return f"{Path(reference).stem}_{category.lower().replace('_', '')}.fits"


# ---------------------------------------------------------------- provenance

PRO_DID = "PRO-1.17"


# Removed from the inherited raw header, as CPL DFS does.
_STRUCTURAL = {"SIMPLE", "BITPIX", "NAXIS", "EXTEND", "COMMENT", "HISTORY", ""}


_REMOVED = {"CHECKSUM", "DATASUM", "ARCFILE", "ORIGFILE", "DATAMD5", "PIPEFILE"}


def _format(value) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return ",".join(f"{key}:{_format(gain)}" for key, gain in value.items())
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def provenance(recipe: str, reference, inputs: dict, **options) -> afits.Header:
    """Primary header for the products of one step call: inherited raw cards + PRO provenance.

    Parameters
    ----------
    recipe : str
        Recipe identifier, e.g. 'planetes_dark'.
    reference : path
        Raw input whose primary header is inherited (DPR and file identity removed).
    inputs : dict
        SOF tag -> list of paths actually used. Raw and calibration files are
        numbered separately (PRO REC1 RAWi / CALi), reference raw first.
    **options
        Recipe options by Python keyword, recorded as PRO REC1 PARAMi with the
        command-line name (underscores become dashes).

    Notes
    -----
    The keywords follow CPL DFS. PRO CATG and PIPEFILE are added by save_product;
    DATAMD5 is added when EsoRex signs the products.
    """
    reference = Path(reference)
    source = afits.getheader(reference)
    header = afits.Header()
    for card in source.cards:
        key = card.keyword
        if key in _STRUCTURAL or key in _REMOVED or key.startswith("NAXIS"):
            continue
        if " " in key and (not key.startswith("ESO ") or key.startswith(("ESO DPR ", "ESO PRO "))):
            continue
        set_card(header, key, card.value, card.comment)
    set_card(header, "DATE", datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S"),
             "file creation date (YYYY-MM-DDThh:mm:ss UT)")
    files = [(Path(path), tag) for tag, paths in inputs.items() for path in paths]
    files.sort(key=lambda item: item[0] != reference)
    raws = [(path, tag) for path, tag in files if is_raw(tag)]
    calibrations = [(path, tag) for path, tag in files if not is_raw(tag)]
    pro = [
        ("DID", PRO_DID, "Data dictionary for PRO"),
        ("TYPE", "REDUCED", "Product type"),
        ("TECH", source.get("ESO DPR TECH", "UNDEFINED"), "Observation technique"),
        ("REC1 ID", recipe, "Pipeline recipe (unique) identifier"),
        ("REC1 DRS ID", f"astropy-{astropy.__version__}", "Data Reduction System identifier"),
        ("REC1 PIPE ID", f"planetes-pipe/{__version__}", "Pipeline (unique) identifier"),
    ]
    for n, (path, tag) in enumerate(raws, 1):
        pro += [(f"REC1 RAW{n} NAME", path.name, "File name of raw frame"),
                (f"REC1 RAW{n} CATG", tag, "Category of raw frame")]
    pro.append(("DATANCOM", len(raws), "Number of combined frames"))
    for n, (path, tag) in enumerate(calibrations, 1):
        pro += [(f"REC1 CAL{n} NAME", path.name, "File name of calibration frame"),
                (f"REC1 CAL{n} CATG", tag, "Category of calibration frame")]
    for n, (name, value) in enumerate(options.items(), 1):
        pro += [(f"REC1 PARAM{n} NAME", name.replace("_", "-"), ""),
                (f"REC1 PARAM{n} VALUE", _format(value), "")]
    for key, value, comment in pro:
        set_card(header, f"ESO PRO {key}", value, comment)
    return header


def output_paths(output_dir, reference, *categories: str) -> dict[str, Path]:
    """{category: output_dir/<reference stem>_<category>.fits} for one step call."""
    return {category: Path(output_dir) / product_filename(reference, category)
            for category in categories}
