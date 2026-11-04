"""Raw data: the detector cube, its recorded header, and how to read it.

    from planetes_pipe.io.raw import read_raw, read_raws
    lamp = read_raw("lamp.fits", "WAVELAMP_RAW")
    flats = read_raws(flat_files, "FLAT_RAW", layout.telescopes)   # {"4": Raw, ...}

The detector setup of a raw file is its DET cards (DICD §4.4.2.7), used as
recorded: a calibration carries the DET cards of the raw it was made from, and
check_calibrations compares those cards directly. The bench writes the same cards.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits as afits
from numpy.typing import NDArray

from planetes_pipe.io.keywords import DIT, DPR_TYPE, PRO_CATG, SHUTTER, WINDOW

# Raw SOF tag -> ESO DPR TYPE of the raw file.
RAW_TYPES = {
    "DARK_RAW": "DARK",
    "FLAT_RAW": "FLAT",
    "P2VM_RAW": "P2VM",
    "WAVELAMP_RAW": "WAVE,LAMP",
    "SINGLE_SCI_RAW": "OBJECT,SINGLE",
}


@dataclass(frozen=True, slots=True)
class Raw:
    """One raw detector cube with its primary header.

    cube is the (frame, y, x) ADU image as stored, memory-mapped by read_raw so
    that large files are not loaded at once.
    """

    cube: NDArray
    category: str
    header: afits.Header
    path: Path | None = None

    @property
    def shutters(self) -> tuple:
        """Recorded shutter states, physical T1..T4 order; None where not recorded."""
        return read_shutters(self.header)

    @property
    def detector(self) -> afits.Header:
        """The DET cards of this raw file."""
        return detector_cards(self.header)

    def chunks(self, size: int = 16) -> Iterator[NDArray]:
        """Frames in order, as float64 blocks of up to `size` frames."""
        for start in range(0, len(self.cube), size):
            yield np.asarray(self.cube[start : start + size], dtype=np.float64)

    def mean(self) -> NDArray:
        """Equally weighted temporal mean over all frames, float64 (y, x) in ADU."""
        total = None
        count = 0
        for chunk in self.chunks():
            if total is None:
                total = np.zeros(chunk.shape[1:])
            total += chunk.sum(axis=0)
            count += len(chunk)
        if total is None:
            raise ValueError("Cube contains no frames")
        return total / count


def read_raw(path: Path, category: str, *, hdu: int | str = "IMAGING_DATA_SC") -> Raw:
    """Open a raw cube (memory-mapped) with its primary header.

    category is the input role chosen by the caller (e.g. DARK_RAW). hdu is the
    image HDU; default IMAGING_DATA_SC, 0 for primary-HDU data. The file is read only.
    """
    with afits.open(path, memmap=True) as hdus:
        return Raw(hdus[hdu].data, category, hdus[0].header.copy(), Path(path))


def read_raws(paths, category: str, identities: tuple[str, ...], *,
              hdu: int | str = "IMAGING_DATA_SC") -> dict[str, Raw]:
    """Read a calibration sequence keyed by physical identity, e.g. {"4": Raw, "3": Raw, ...}.

    Identity comes from the recorded shutters ('4' = only T4 open, '43' = T4 and T3),
    never from the filename or input order. There must be exactly one file per
    identity: a missing, repeated or unexpected shutter configuration raises.
    """
    raws = [read_raw(path, category, hdu=hdu) for path in paths]
    found = [shutter_identity(raw.shutters) for raw in raws]
    if sorted(found) != sorted(identities):
        raise ValueError(f"Expected one {category} per shutter configuration {identities}; "
                         f"found {tuple(found)}")
    by_identity = dict(zip(found, raws))
    return {identity: by_identity[identity] for identity in identities}


def find_raw(directory: Path, category: str) -> list[Path]:
    """Raw files of one category in a directory, by DPR.TYPE (never by filename)."""
    found = []
    for path in sorted(Path(directory).glob("*.fits")):
        header = afits.getheader(path)
        if DPR_TYPE.get(header) == RAW_TYPES[category] and PRO_CATG.get(header) is None:
            found.append(path)
    return found


def detector_cards(header: afits.Header) -> afits.Header:
    """The DET cards of a header, as recorded."""
    return afits.Header([card for card in header.cards if card.keyword.startswith("ESO DET ")])


def check_match(header: afits.Header, calibration: afits.Header | None, keywords, role: str) -> None:
    """Raise if a calibration was recorded with a different value of any of `keywords`.

    calibration holds the calibration's DET cards, or None when they are unknown.
    A keyword recorded on only one side is not compared.
    """
    if calibration is None:
        return
    for keyword in keywords:
        actual, expected = keyword.get(header), keyword.get(calibration)
        if actual is not None and expected is not None and actual != expected:
            raise ValueError(f"{role} has {keyword.name} = {expected!r}, the raw data {actual!r}")


def check_calibrations(raw: Raw, *, dark=None, flat=None) -> None:
    """A DARK must share the raw's detector window and DIT; a FLAT its window."""
    if dark is not None:
        check_match(raw.header, dark.detector, (*WINDOW, DIT), "DARK")
    if flat is not None:
        check_match(raw.header, flat.detector, WINDOW, "FLAT")


def read_shutters(header: afits.Header) -> tuple:
    """Shutter states in physical T1..T4 order; None where not recorded.

    Files written before 2026-11 use the non-ESO names 'PLANETES INS SHUTn ST'.
    """
    return tuple(SHUTTER.at(n).get(header, header.get(f"PLANETES INS SHUT{n} ST"))
                 for n in range(1, 5))


def write_shutters(header: afits.Header, shutters) -> None:
    """Write shutter states in physical T1..T4 order."""
    for n, state in enumerate(shutters, 1):
        SHUTTER.at(n).set(header, state)


def shutter_identity(shutters: tuple) -> str:
    """Open shutter numbers in descending physical order, e.g. '4' or '43'."""
    if None in shutters:
        raise ValueError("Missing shutter states; cannot identify calibration input")
    return "".join(str(n) for n in range(4, 0, -1) if shutters[n - 1])
