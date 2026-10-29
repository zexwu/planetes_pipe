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
    """Adapt raw/Python image input to native float64 or complex128 once."""
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
    """Yield a temporary FITS path, then publish without replacing an existing file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".planetes-", dir=path.parent) as directory:
        temporary = Path(directory) / path.name
        yield temporary
        os.link(temporary, path)
