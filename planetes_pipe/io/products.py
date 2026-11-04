"""Product FITS files: write and read, the product header, and extension helpers.

Order: the file API (save_product, open_product, product_path); the primary
header (provenance: inherited raw cards + DICD §4.7.1 PRO records); helpers that
step modules use to describe their extensions. Floating data are stored as
computed (float64/complex128). Keywords come from keywords.py. No CPL is used.
"""

from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import astropy
import numpy as np
from astropy.io import fits as afits

from planetes_pipe import __version__
from planetes_pipe.instrument import layout
from planetes_pipe.io.keywords import (
    DATE,
    DPR_TECH,
    INHERIT,
    INSTRUME,
    PIPEFILE,
    PRO_CAL_CATG,
    PRO_CAL_NAME,
    PRO_CATG,
    PRO_DATANCOM,
    PRO_DID,
    PRO_DRS,
    PRO_PARAM_NAME,
    PRO_PARAM_VALUE,
    PRO_PIPELINE,
    PRO_RAW_CATG,
    PRO_RAW_NAME,
    PRO_RECIPE,
    PRO_SCIENCE,
    PRO_TECH,
    PRO_TYPE,
    sorted_by_category,
)


def save_product(path: Path, category: str, extensions: list, *, header=None,
                 qc: dict | None = None, detector: afits.Header | None = None) -> None:
    """Write a product file; an existing file raises OSError.

    header is a provenance() header, or None for a product made outside a recipe.
    qc maps QC Keywords to measured values; None and non-finite values are omitted.
    detector is the DET cards of the raw data the product was made from.
    """
    primary = afits.Header() if header is None else header.copy()
    INSTRUME.set(primary, "PLANETES")
    PIPEFILE.set(primary, Path(path).name)
    PRO_CATG.set(primary, category)
    PRO_TYPE.set(primary, "REDUCED")
    PRO_SCIENCE.set(primary, category == "SINGLE_SCI_P2VMRED")
    PRO_DID.set(primary, "PRO-1.17")
    if detector is not None:
        primary.extend(detector.cards, update=True)
    for keyword, value in (qc or {}).items():
        if value is not None and np.isfinite(value):
            keyword.set(primary, value)
    for extension in extensions:
        INHERIT.set(extension.header, True)
    hdus = afits.HDUList([afits.PrimaryHDU(header=sorted_by_category(primary)), *extensions])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    hdus.writeto(path, checksum=True)


@contextmanager
def open_product(path: Path, category: str):
    """Open a product, checking that it is of the expected category."""
    with afits.open(path) as hdus:
        if PRO_CATG.get(hdus[0].header) != category:
            raise ValueError(f"Expected a {category} product: {path}")
        yield hdus


def product_path(output_dir, reference, category: str) -> Path:
    """output_dir/<reference raw stem>_<category without underscores>.fits.

    E.g. PLANETES.2026-02-05T210609_singlescip2vmred.fits. The raw stem is kept as is
    (timestamp or UNKNOWN marker); no time is invented.
    """
    return Path(output_dir) / f"{Path(reference).stem}_{category.lower().replace('_', '')}.fits"


def provenance(recipe: str, reference, inputs: dict, **options) -> afits.Header:
    """Header for the products of one step call: the reference raw's cards plus PRO records.

    Parameters
    ----------
    recipe : str
        Recipe identifier, e.g. 'planetes_dark'.
    reference : path
        Raw input whose observation cards the products inherit.
    inputs : dict
        SOF tag -> list of paths actually used. Raw and calibration files are
        numbered separately (PRO.REC1.RAWi / CALi), reference raw first.
    **options
        Recipe options by Python keyword, recorded as PRO.REC1.PARAMi with the
        command-line name (underscores become dashes).

    Notes
    -----
    Follows DICD §4.7.1 and CPL DFS. save_product adds PRO.CATG/TYPE/SCIENCE/DID
    and PIPEFILE; EsoRex adds DATAMD5 when it signs the products.
    """
    reference = Path(reference)
    raw_header = afits.getheader(reference)
    header = _inherited(raw_header)
    DATE.set(header, datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3])
    PRO_TECH.set(header, DPR_TECH.get(raw_header, "UNDEFINED"))
    PRO_RECIPE.set(header, recipe)
    PRO_DRS.set(header, f"astropy-{astropy.__version__}")
    PRO_PIPELINE.set(header, f"planetes-pipe/{__version__}")

    used = sorted(((Path(path), tag) for tag, paths in inputs.items() for path in paths),
                  key=lambda item: item[0] != reference)
    raws = [(path, tag) for path, tag in used if is_raw(tag)]
    calibrations = [(path, tag) for path, tag in used if not is_raw(tag)]
    for n, (path, tag) in enumerate(raws, 1):
        PRO_RAW_NAME.at(n).set(header, path.name)
        PRO_RAW_CATG.at(n).set(header, tag)
    PRO_DATANCOM.set(header, len(raws))
    for n, (path, tag) in enumerate(calibrations, 1):
        PRO_CAL_NAME.at(n).set(header, path.name)
        PRO_CAL_CATG.at(n).set(header, tag)
    for n, (name, value) in enumerate(options.items(), 1):
        PRO_PARAM_NAME.at(n).set(header, name.replace("_", "-"))
        PRO_PARAM_VALUE.at(n).set(header, _format(value))
    return header


# Raw primary cards a product does not inherit (as CPL DFS): structure, checksums,
# file identity. DPR, PRO and non-ESO hierarchical cards are not inherited either.
_NOT_INHERITED = {"SIMPLE", "BITPIX", "NAXIS", "EXTEND", "COMMENT", "HISTORY", "",
                  "CHECKSUM", "DATASUM", "ARCFILE", "ORIGFILE", "DATAMD5", "PIPEFILE"}


def _inherited(raw_header: afits.Header) -> afits.Header:
    """The reference raw's observation cards, as a product header starts from."""
    header = afits.Header()
    for card in raw_header.cards:
        key = card.keyword
        if key in _NOT_INHERITED or key.startswith("NAXIS"):
            continue
        hierarchical = " " in key
        if hierarchical and (not key.startswith("ESO ") or key.startswith(("ESO DPR ", "ESO PRO "))):
            continue
        header.append(card)
    return header


def is_raw(tag: str) -> bool:
    """RAW (not calibration) input in the DICD and CPL frame-group sense."""
    return tag.endswith("_RAW") or tag == "PHASE_REFERENCE"


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
