"""DICD processing (PRO) keywords for every PLANETES product.

Python calls and PyEsoRex recipes both write products through this header; the
keywords follow what CPL DFS writes. EsoRex additionally signs the files
(DATAMD5) and registers them as products.
"""

from datetime import UTC, datetime
from pathlib import Path

import astropy
from astropy.io import fits as afits

from planetes_pipe import __version__

PRO_DID = "PRO-1.17"
# Removed from the inherited raw header, as CPL DFS does.
_STRUCTURAL = {"SIMPLE", "BITPIX", "NAXIS", "EXTEND", "COMMENT", "HISTORY", ""}
_REMOVED = {"CHECKSUM", "DATASUM", "ARCFILE", "ORIGFILE", "DATAMD5", "PIPEFILE"}


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


def provenance_header(
    reference: Path,
    *,
    recipe: str,
    category: str,
    inputs: list[tuple[Path, str]],
    parameters: list[tuple[str, object, object, str]],
    filename: str,
) -> afits.Header:
    """Inherit the reference raw header and add DICD PRO provenance keywords.

    Parameters
    ----------
    reference : Path
        Raw input whose primary header is inherited (DPR and file identity removed).
    recipe : str
        Recipe identifier, e.g. 'planetes_dark'.
    category : str
        ESO PRO CATG of the product.
    inputs : list of (path, tag)
        Inputs actually used, reference first; raw and calibration files are
        numbered separately, as CPL does.
    parameters : list of (option name, value, default, help)
        Recipe parameters in declaration order.
    filename : str
        Published product filename, recorded as PIPEFILE.

    Returns
    -------
    header : astropy.io.fits.Header
        Primary header ready for the product encoder. DATAMD5 is not written;
        archive signing is done by EsoRex.
    """
    source = afits.getheader(reference)
    header = afits.Header()
    for card in source.cards:
        key = card.keyword
        if key in _STRUCTURAL or key in _REMOVED or key.startswith("NAXIS"):
            continue
        if " " in key and (not key.startswith("ESO ") or key.startswith(("ESO DPR ", "ESO PRO "))):
            continue
        header["HIERARCH " + key if " " in key else key] = (card.value, card.comment)
    header["DATE"] = (
        datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S"),
        "file creation date (YYYY-MM-DDThh:mm:ss UT)",
    )
    header["PIPEFILE"] = (filename, "Filename of data product")
    pro = [
        ("CATG", category, "Category of pipeline product frame"),
        ("DID", PRO_DID, "Data dictionary for PRO"),
        ("TYPE", "REDUCED", "Product type"),
        ("TECH", source.get("ESO DPR TECH", "UNDEFINED"), "Observation technique"),
        ("SCIENCE", False, "Scientific product if T"),
        ("REC1 ID", recipe, "Pipeline recipe (unique) identifier"),
        ("REC1 DRS ID", f"astropy-{astropy.__version__}", "Data Reduction System identifier"),
        ("REC1 PIPE ID", f"planetes-pipe/{__version__}", "Pipeline (unique) identifier"),
    ]
    raws = [(path, tag) for path, tag in inputs if is_raw(tag)]
    calibrations = [(path, tag) for path, tag in inputs if not is_raw(tag)]
    for n, (path, tag) in enumerate(raws, 1):
        pro += [
            (f"REC1 RAW{n} NAME", path.name, "File name of raw frame"),
            (f"REC1 RAW{n} CATG", tag, "Category of raw frame"),
        ]
    pro.append(("DATANCOM", len(raws), "Number of combined frames"))
    for n, (path, tag) in enumerate(calibrations, 1):
        pro += [
            (f"REC1 CAL{n} NAME", path.name, "File name of calibration frame"),
            (f"REC1 CAL{n} CATG", tag, "Category of calibration frame"),
        ]
    for n, (name, value, default, description) in enumerate(parameters, 1):
        pro += [
            (f"REC1 PARAM{n} NAME", name, description),
            (f"REC1 PARAM{n} VALUE", _format(value), f"Default: {_format(default)}"),
        ]
    for key, value, comment in pro:
        header[f"HIERARCH ESO PRO {key}"] = (value, comment)
    return header
