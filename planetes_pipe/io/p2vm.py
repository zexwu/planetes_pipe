"""P2VM FITS layout, units and coordinate conversions."""

from pathlib import Path
from collections.abc import Mapping
import numpy as np
from numpy.typing import NDArray
from astropy.io import fits as afits
from .common import primary_header, write_quality, _verify_checksums, new_fits_file

from planetes_pipe.model.instrument import layout
from planetes_pipe.model.products import P2VM
from .common import _image, _column, _table, _detector_table, read_image, read_column


def encode(product: P2VM, header=None, *, quality=None) -> afits.HDUList:
    """Encode this product as FITS HDUs with the supplied header and QC."""
    primary = afits.PrimaryHDU(header=primary_header(product, header))
    hdus = []
    hdus += [
        _detector_table(product.baseline_outputs),
        _table(
            "PLANETES_WAVELENGTH", [_column("WAVELENGTH", product.wavelength_m, "m")]
        ),
    ]
    columns = [
        _column("REGNAME", hdus[0].data["REGNAME"]),
        _column("TRANSMISSION", product.transmission),
        _column("COHERENCE", product.coherence_amplitude),
        _column("PHASE", product.coherence_phase_rad, "rad"),
    ]
    if product.dc_term_adu is not None:
        columns.append(_column("C_MATRIX", product.dc_term_adu, "adu"))
    hdus.append(_table("P2VM_SC", columns))
    for name, ext, unit in [
        ("matrix", "PLANETES_P2VM_INVERSE", ""),
        ("opd_m", "PLANETES_OPD", "m"),
        ("group_delay_m", "PLANETES_GD", "m"),
        ("ellipse_phase_rad", "PLANETES_ELLIPSE_PHASE", "rad"),
        ("ellipse_amplitude", "PLANETES_ELLIPSE_AMPLITUDE", ""),
        ("fit_rms_adu", "PLANETES_P2VM_FIT_RMS", "adu"),
    ]:
        if getattr(product, name) is not None:
            hdus.append(_image(ext, getattr(product, name), unit))
    hdus = afits.HDUList([primary, *hdus])
    write_quality(primary.header, quality)
    return hdus


def decode(hdus: afits.HDUList) -> P2VM:
    """Decode this product from an already opened FITS file."""
    values = {}
    values["wavelength_m"] = read_column(hdus, "PLANETES_WAVELENGTH", "WAVELENGTH", "m")
    labels = list(hdus["IMAGING_DETECTOR_SC"].data["REGNAME"])
    values["baseline_outputs"] = {
        i: [labels.index(name + "-" + a) for a in layout.abcd_order]
        for i, name in enumerate(layout.baselines)
    }
    if list(hdus["P2VM_SC"].data["REGNAME"]) != labels:
        raise ValueError("P2VM and detector REGNAME mismatch")
    values.update(
        transmission=read_column(hdus, "P2VM_SC", "TRANSMISSION"),
        coherence_amplitude=read_column(hdus, "P2VM_SC", "COHERENCE"),
        coherence_phase_rad=read_column(hdus, "P2VM_SC", "PHASE", "rad"),
    )
    if "C_MATRIX" in hdus["P2VM_SC"].columns.names:
        values["dc_term_adu"] = read_column(hdus, "P2VM_SC", "C_MATRIX", "adu")
    for name, ext, unit in [
        ("matrix", "PLANETES_P2VM_INVERSE", ""),
        ("opd_m", "PLANETES_OPD", "m"),
        ("group_delay_m", "PLANETES_GD", "m"),
        ("ellipse_phase_rad", "PLANETES_ELLIPSE_PHASE", "rad"),
        ("ellipse_amplitude", "PLANETES_ELLIPSE_AMPLITUDE", ""),
        ("fit_rms_adu", "PLANETES_P2VM_FIT_RMS", "adu"),
    ]:
        if ext in hdus:
            values[name] = read_image(hdus, ext, unit)
    return P2VM(**values)


def read(path: Path) -> P2VM:
    """Read a checksummed P2VM product at computation precision."""
    with afits.open(path) as hdus:
        if (
            hdus[0].header.get("INSTRUME") != "PLANETES"
            or hdus[0].header.get("ESO PRO CATG") != "P2VM"
        ):
            raise ValueError("Expected a PLANETES P2VM product")
        _verify_checksums(hdus)
        return decode(hdus)


def write(path: Path, product: P2VM, *, quality=None) -> None:
    """Save this product without overwriting a file or recomputing QC."""
    with encode(product, quality=quality) as hdus:
        with new_fits_file(path) as temporary:
            hdus.writeto(temporary, checksum=True)


def read_scan_opd(
    baseline_cubes: Mapping[str, Path], dl_gain: Mapping[str, float]
) -> NDArray:
    """Read per-frame DL_SCAN positions and explicitly convert mechanical mm to OPD m."""
    if set(dl_gain) != set(layout.baselines):
        raise ValueError("dl_gain requires all six physical baselines")
    positions = []
    for baseline in layout.baselines:
        table, header = afits.getdata(baseline_cubes[baseline], "DL_SCAN", header=True)
        if header["MOVINGDL"] != int(baseline[1]):
            raise ValueError(f"DL gain assumes the second arm moves: {baseline}")
        counts = table["NFRAMES"]
        if not np.array_equal(table["FIRST_FRAME"], np.cumsum(counts) - counts):
            raise ValueError(f"Noncontiguous DL_SCAN frames: {baseline}")
        position = np.repeat((table["BEFORE_MM"] + table["AFTER_MM"]) / 2, counts)
        positions.append(position * 1e-3 * dl_gain[baseline])
    return np.asarray(positions)
