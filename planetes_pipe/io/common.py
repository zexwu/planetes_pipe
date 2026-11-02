"""I/O precision boundary, FITS metadata and checksums."""

from dataclasses import fields
import os
from contextlib import contextmanager
from pathlib import Path
from shutil import rmtree
from uuid import uuid4

import numpy as np
from astropy.io import fits as afits

from planetes_pipe.model.instrument import layout, DetectorSetup, _check_baselines
from planetes_pipe.model.products import Bad, Dark, Flat, Wave, P2VM, P2VMRed
from planetes_pipe.model.quality import WaveQC

CATEGORIES = {
    Dark: "DARK",
    Bad: "BAD",
    Flat: "FLAT",
    Wave: "WAVE",
    P2VM: "P2VM",
    P2VMRed: "SINGLE_SCI_P2VMRED",
}

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
def new_fits_file(path: Path):
    """Stage one new FITS file beside its destination and publish without overwriting.

    Parameters
    ----------
    path : Path
        Requested final path. Parent directories are created as needed.

    Yields
    ------
    temporary : Path
        New temporary filename. The caller must write and close the complete FITS
        file here before leaving the context, including any required signatures.

    Notes
    -----
    On successful exit, a hard link publishes the completed file. An existing
    destination raises OSError. Temporary files are removed on success or failure;
    previously published independent products are not rolled back.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    directory = path.parent / f".planetes-{uuid4().hex}"
    # Normal mkdir inherits the destination ACL; TemporaryDirectory restricts it on Windows.
    directory.mkdir()
    try:
        temporary = directory / path.name
        yield temporary
        os.link(temporary, path)
    finally:
        rmtree(directory)


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


def _detector_table(baseline_outputs=None):
    labels = [f"REG{i + 1:02d}" for i in range(24)]
    if baseline_outputs is not None:
        _check_baselines(baseline_outputs)
        for baseline, outputs in baseline_outputs.items():
            for output, phase in zip(outputs, layout.abcd_order, strict=True):
                labels[output] = layout.baselines[baseline] + "-" + phase
    return _table("IMAGING_DETECTOR_SC", [_column("REGNAME", np.asarray(labels))])


def read_image(hdus, ext, unit):
    """Read an image with its expected unit at computation precision."""
    item = hdus[ext]
    if item.header.get("BUNIT", "") != unit:
        raise ValueError(f"Incompatible units: {ext}")
    return _native(item.data)


def read_column(hdus, ext, name, unit=""):
    """Read a table column, checking its unit and logical encoding."""
    table = hdus[ext]
    if (table.columns[name].unit or "") != unit:
        raise ValueError(f"Incompatible units: {ext}.{name}")
    if table.columns[name].format.endswith("L"):
        raw = table.data._get_raw_data()[name]
        if not np.isin(raw, [ord("T"), ord("F")]).all():
            raise ValueError("Invalid FITS logical flag encoding")
    return _native(table.data[name])


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


def read_setup(hdr):
    """Decode detector setup; FITS origins are one-based, Python origins zero-based."""
    x, y = hdr.get("ESO DET WIN1 STRX"), hdr.get("ESO DET WIN1 STRY")
    setup = DetectorSetup(
        origin_x=None if x is None else x - 1,
        origin_y=None if y is None else y - 1,
        width=hdr.get("ESO DET WIN1 NX"),
        height=hdr.get("ESO DET WIN1 NY"),
        binning_x=hdr.get("ESO DET WIN1 BINX"),
        binning_y=hdr.get("ESO DET WIN1 BINY"),
        exposure_s=hdr.get("ESO DET WIN1 DIT1"),
        detector=hdr.get("ESO DET ID"),
    )
    return (
        setup
        if any(getattr(setup, f.name) is not None for f in fields(setup))
        else None
    )


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
    """Read recorded shutter states in physical T1,T2,T3,T4 order."""
    return tuple(header.get(f"PLANETES INS SHUT{n} ST") for n in range(1, 5))


def write_shutters(header, shutters):
    """Write shutter states in physical T1,T2,T3,T4 order."""
    for n, state in enumerate(shutters, 1):
        header[f"HIERARCH PLANETES INS SHUT{n} ST"] = state


def primary_header(product, header=None):
    """Copy common product identity and recorded detector setup into the header."""
    hdr = afits.Header() if header is None else header.copy()
    for key in list(hdr):
        if key.startswith("ESO DPR "):
            del hdr[key]
    hdr["INSTRUME"] = "PLANETES"
    hdr["HIERARCH ESO PRO CATG"] = CATEGORIES[type(product)]
    hdr["HIERARCH ESO PRO TYPE"] = "REDUCED"
    hdr["HIERARCH ESO PRO SCIENCE"] = isinstance(product, P2VMRed)
    setup = getattr(product, "setup", None)
    if setup is not None:
        write_setup(hdr, setup)

    return hdr


def write_quality(header, quality):
    """Write supplied QC values without computing any statistics."""
    if quality is not None:
        for metric in fields(quality):
            number = getattr(quality, metric.name)
            key = metric.metadata.get("keyword")
            if key is None or number is None or not np.isfinite(number):
                continue
            comment = metric.metadata["comment"]
            if isinstance(quality, WaveQC):
                comment = comment.format(n=quality.residual_count)
            header["HIERARCH " + key] = (number, comment)
            if "um_keyword" in metric.metadata:
                header["HIERARCH " + metric.metadata["um_keyword"]] = (
                    number * 1e6, comment.replace("[m]", "[um]"))


def product_filename(reference: Path, product) -> str:
    """Build a GRAVITY-style product name from its explicitly selected raw input.

    Parameters
    ----------
    reference : Path
        Reference raw filename. Its complete stem, including a recorded timestamp
        or an explicit UNKNOWN marker, is retained without interpretation.
    product : Dark, Bad, Flat, Wave, P2VM or P2VMRed
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
