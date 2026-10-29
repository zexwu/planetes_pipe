"""I/O precision boundary, FITS metadata and checksums."""

import os
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

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
    with TemporaryDirectory(prefix=".planetes-", dir=path.parent) as directory:
        temporary = Path(directory) / path.name
        yield temporary
        os.link(temporary, path)
