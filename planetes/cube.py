"""Stateless reading of raw detector FITS cubes."""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
from astropy.io import fits
from numpy.typing import NDArray

INPUT_HDU = 1
CHUNK_FRAMES = 16


def iter_cube(path: Path) -> Iterator[NDArray[np.float32]]:
    """Yield a complete three-dimensional detector cube in frame chunks."""
    with fits.open(path, memmap=True) as hdul:
        try:
            data = hdul[INPUT_HDU].data
        except (IndexError, KeyError) as exc:
            raise ValueError(f"FITS HDU {INPUT_HDU} not found in {path}") from exc
        if data is None:
            raise ValueError(f"FITS HDU {INPUT_HDU} contains no data in {path}")
        if data.ndim != 3:
            raise ValueError(
                f"Expected a 3D FITS cube in {path}, got shape {data.shape}"
            )

        for chunk_start in range(0, len(data), CHUNK_FRAMES):
            chunk_stop = min(chunk_start + CHUNK_FRAMES, len(data))
            yield np.asarray(
                data[chunk_start:chunk_stop],
                dtype=np.float32,
            )
