"""Raw detector frames in ADU and their recorded acquisition metadata."""

from collections.abc import Iterator
from dataclasses import dataclass
from numpy.typing import NDArray
from .instrument import DetectorSetup


@dataclass(frozen=True, slots=True)
class RawMetadata:
    """Raw category, physical shutter states and detector settings."""

    category: str
    shutters: tuple[bool | None, bool | None, bool | None, bool | None]
    setup: DetectorSetup


@dataclass(slots=True)
class Raw:
    """One-pass float64 frame chunks, valid inside the raw reader's with block.

    Shutters are in physical T1,T2,T3,T4 order. Unknown metadata stays None.
    No FITS objects or filenames enter the scientific calculations.
    """

    frames: Iterator[NDArray]
    category: str
    shutters: tuple[bool | None, bool | None, bool | None, bool | None]
    setup: DetectorSetup


def shutter_identity(shutters: tuple[bool | None, ...]) -> str:
    """Open shutter numbers in descending physical order, e.g. '4' or '43'."""
    if None in shutters:
        raise ValueError("Missing shutter states; cannot identify calibration input")
    return "".join(str(n) for n in range(4, 0, -1) if shutters[n - 1])
