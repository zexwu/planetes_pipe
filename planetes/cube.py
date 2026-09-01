"""Stateless reading of raw detector FITS cubes."""

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
    hdu: int | str | None = None,
) -> Iterator[NDArray[np.float32]]:
    """Yield a selected three-dimensional FITS cube in frame chunks."""
    if not path.exists():
        raise FileNotFoundError(f"FITS file not found: {path}")
    if chunk_frames <= 0:
        raise ValueError("chunk_frames must be positive")

    with fits.open(path, memmap=True) as hdul:
        if hdu is None:
            data = next((item.data for item in hdul if item.data is not None), None)
        else:
            try:
                data = hdul[hdu].data
            except (IndexError, KeyError) as exc:
                raise ValueError(f"FITS HDU {hdu!r} not found in {path}") from exc
        if data is None:
            raise ValueError(f"FITS HDU {hdu!r} contains no data in {path}")
        if data.ndim != 3:
            raise ValueError(f"Expected a 3D FITS cube in {path}, got shape {data.shape}")

        start, stop = _bounds(selection.frames, len(data), "frame", path)
        y_start, y_stop = _bounds(selection.y, data.shape[1], "y", path)
        x_start, x_stop = _bounds(selection.x, data.shape[2], "x", path)
        for chunk_start in range(start, stop, chunk_frames):
            chunk_stop = min(chunk_start + chunk_frames, stop)
            yield np.asarray(
                data[chunk_start:chunk_stop, y_start:y_stop, x_start:x_stop],
                dtype=np.float32,
            )


def _bounds(value: slice, size: int, axis: str, path: Path) -> tuple[int, int]:
    if value.step not in (None, 1):
        raise ValueError(f"{axis} selection step must be 1")
    start = value.start or 0
    stop = size if value.stop is None else value.stop
    if start < 0 or stop > size:
        raise ValueError(
            f"{axis} selection [{start}, {stop}) exceeds size {size} in {path}"
        )
    return start, stop
