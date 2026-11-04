"""Product FITS files: writing, reading, extensions, QC cards and DICD provenance.

Each step module describes its own extensions with column()/image_hdu() and calls
save_product()/open_product(); everything shared between products is here.
Floating data are stored as computed (float64/complex128). No CPL is used.
"""

from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import astropy
import numpy as np
from astropy.io import fits as afits

from planetes_pipe import __version__
from planetes_pipe.instrument import DetectorSetup, layout
from planetes_pipe.raw import write_setup


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
    """Open a product, checking that it is of the expected category."""
    with afits.open(path) as hdus:
        if hdus[0].header.get("ESO PRO CATG") != category:
            raise ValueError(f"Expected a {category} product: {path}")
        yield hdus


def product_path(output_dir, reference, category: str) -> Path:
    """output_dir/<reference raw stem>_<category without underscores>.fits.

    E.g. PLANETES.2026-02-05T210609_singlescip2vmred.fits. The raw stem is kept as is
    (timestamp or UNKNOWN marker); no time is invented.
    """
    return Path(output_dir) / f"{Path(reference).stem}_{category.lower().replace('_', '')}.fits"


# numpy dtype (without byte order) -> FITS binary-table format letter.
FITS_FORMATS = {"f4": "E", "f8": "D", "c8": "C", "c16": "M", "b1": "L"}


def image_hdu(name, value, unit=""):
    hdu = afits.ImageHDU(np.asarray(value), name=name)
    hdu.header["BUNIT"] = unit
    return hdu


def column(name, values, unit=""):
    """Encode one column as stored in memory, keeping trailing array axes in FITS order."""
    values = np.asarray(values)
    if values.dtype.kind in "US":
        width = max((len(value) for value in values), default=1)
        return afits.Column(name=name, array=values, format=f"{max(1, width)}A")
    kind = FITS_FORMATS.get(values.dtype.str[1:], "K")  # Integers are stored as int64.
    count = int(np.prod(values.shape[1:])) if values.ndim > 1 else 1
    dim = None
    if values.ndim > 1:
        dim = "(" + ",".join(map(str, values.shape[:0:-1])) + ")"
    return afits.Column(
        name=name, array=values, format=f"{count}{kind}", unit=unit or None, dim=dim
    )


def read_image(hdus, ext):
    """An image extension, copied into native byte order (FITS is big-endian; numba needs native)."""
    data = hdus[ext].data
    return np.array(data, dtype=data.dtype.newbyteorder("="))


def read_column(hdus, ext, name):
    """A table column, copied into native byte order."""
    data = hdus[ext].data[name]
    return np.array(data, dtype=data.dtype.newbyteorder("="))


def detector_table(baseline_outputs=None):
    labels = [f"REG{i + 1:02d}" for i in range(24)]
    if baseline_outputs is not None:
        for baseline, outputs in baseline_outputs.items():
            for output, phase in zip(outputs, layout.abcd_order, strict=True):
                labels[output] = layout.baselines[baseline] + "-" + phase
    return afits.BinTableHDU.from_columns([column("REGNAME", np.asarray(labels))],
                                          name="IMAGING_DETECTOR_SC")


def baseline_outputs(labels) -> dict[int, list[int]]:
    """Decode IMAGING_DETECTOR_SC REGNAME labels ('43-A', ...) to baseline -> A,C,B,D outputs."""
    labels = list(labels)
    return {i: [labels.index(name + "-" + phase) for phase in layout.abcd_order]
            for i, name in enumerate(layout.baselines)}


def write_qc(header, values: dict | None) -> None:
    """Write {keyword: (value, comment)} from a product's qc(); unmeasured values are omitted."""
    for key, (number, comment) in (values or {}).items():
        if number is not None and np.isfinite(number):
            set_card(header, key, number, comment)


def set_card(header, key: str, value, comment: str = "") -> None:
    """Set a card, trimming the comment so the card fits 80 columns untruncated."""
    name = "HIERARCH " + key if " " in key or len(key) > 8 else key
    header[name] = value
    used = len(header.cards[name].image.rstrip())
    room = 80 - used - 3
    header[name] = (value, comment[:room] if room > 0 else "")


def is_raw(tag: str) -> bool:
    """RAW (not calibration) input in the DICD and CPL frame-group sense."""
    return tag.endswith("_RAW") or tag == "PHASE_REFERENCE"


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
    entries = [(Path(path), tag) for tag, paths in inputs.items() for path in paths]
    entries.sort(key=lambda item: item[0] != reference)
    raws = [(path, tag) for path, tag in entries if is_raw(tag)]
    calibrations = [(path, tag) for path, tag in entries if not is_raw(tag)]
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
