"""Stateless FITS cube reading."""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class CubeSelection:
    """Frame and detector slices applied while reading a cube."""

    frames: slice
    y: slice
    x: slice


def iter_cube(
    path: Path,
    selection: CubeSelection,
    chunk_frames: int = 16,
) -> Iterator[NDArray[np.float32]]:
    """Yield a selected three-dimensional FITS cube in frame chunks."""
    if not path.exists():
        raise FileNotFoundError(f"FITS file not found: {path}")
    if chunk_frames <= 0:
        raise ValueError("chunk_frames must be positive")

    with fits.open(path, memmap=True) as hdul:
        data = next((hdu.data for hdu in hdul if hdu.data is not None), None)
        if data is None:
            raise ValueError(f"FITS file contains no data HDU: {path}")
        if data.ndim != 3:
            raise ValueError(f"Expected a 3D FITS cube in {path}, got shape {data.shape}")

        start = selection.frames.start or 0
        stop = (
            len(data)
            if selection.frames.stop is None
            else min(selection.frames.stop, len(data))
        )
        for chunk_start in range(start, stop, chunk_frames):
            chunk_stop = min(chunk_start + chunk_frames, stop)
            yield np.asarray(
                data[chunk_start:chunk_stop, selection.y, selection.x],
                dtype=np.float32,
            )
