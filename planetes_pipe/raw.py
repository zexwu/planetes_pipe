"""Raw data: the detector cube, its header (shutters, detector setup) and how to read it.

    from planetes_pipe.raw import read_raw, read_raws
    lamp = read_raw("lamp.fits", "WAVELAMP_RAW")
    flats = read_raws(flat_files, "FLAT_RAW", layout.telescopes)   # {"4": Raw, ...}

The DET and shutter header encoding here is also used for product headers and by
the bench when it writes raw files.
"""

from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from astropy.io import fits as afits
from numpy.typing import NDArray

from planetes_pipe.instrument import DetectorSetup, check_setup, shutter_identity
from planetes_pipe.keywords import (
    DET_ID,
    DIT,
    DPR_TYPE,
    PRO_CATG,
    SHUTTER,
    UIT,
    WIN_BINX,
    WIN_BINY,
    WIN_NX,
    WIN_NY,
    WIN_STRX,
    WIN_STRY,
)

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
    """One raw detector cube and its recorded metadata.

    cube is the (frame, y, x) ADU image HDU as stored, memory-mapped by
    read_raw, so large files are not loaded at once. Shutters are in
    physical T1,T2,T3,T4 order; unknown metadata stays None.
    """

    cube: NDArray
    category: str
    shutters: tuple[bool | None, bool | None, bool | None, bool | None]
    setup: DetectorSetup
    path: Path | None = None

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
    """Open a raw cube (memory-mapped) with its shutters and detector setup.

    Parameters
    ----------
    path : Path
        Raw FITS file containing a (frame, y, x) detector cube in ADU.
    category : str
        Input role chosen by the caller, e.g. DARK_RAW or FLAT_RAW.
    hdu : int or str, optional
        Image HDU; default IMAGING_DATA_SC, 0 for primary-HDU data. No search.

    Notes
    -----
    Width/height come from the cube; FITS one-based ROI origins become zero-based.
    The file is read only. Frames are read from disk as they are used.
    """
    with afits.open(path, memmap=True) as hdus:
        header = hdus[0].header
        cube = hdus[hdu].data
    setup = replace(read_setup(header), height=cube.shape[1], width=cube.shape[2])
    return Raw(cube, category, read_shutters(header), setup, Path(path))


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


def check_calibrations(raw: Raw, *, dark=None, flat=None) -> None:
    """Reject a DARK whose known DIT/ROI/detector, or a FLAT whose ROI/detector, differs from raw."""
    if dark is not None:
        check_setup(raw.setup, dark.setup, exposure=True, role="DARK")
    if flat is not None:
        check_setup(raw.setup, flat.setup, exposure=False, role="FLAT")


def read_setup(header) -> DetectorSetup:
    """Detector setup from the DET keywords; unknown fields stay None.

    FITS window origins are one-based, Python origins zero-based.
    """
    x, y = WIN_STRX.get(header), WIN_STRY.get(header)
    return DetectorSetup(
        origin_x=x - 1 if x is not None else None,
        origin_y=y - 1 if y is not None else None,
        width=WIN_NX.get(header),
        height=WIN_NY.get(header),
        binning_x=WIN_BINX.get(header),
        binning_y=WIN_BINY.get(header),
        exposure_s=DIT.get(header, UIT.get(header)),
        detector=DET_ID.get(header),
    )


def write_setup(header, setup: DetectorSetup) -> None:
    """Write the known detector settings; window origins become one-based."""
    values = {
        WIN_STRX: setup.origin_x + 1 if setup.origin_x is not None else None,
        WIN_STRY: setup.origin_y + 1 if setup.origin_y is not None else None,
        WIN_NX: setup.width,
        WIN_NY: setup.height,
        WIN_BINX: setup.binning_x,
        WIN_BINY: setup.binning_y,
        DET_ID: setup.detector,
        DIT: setup.exposure_s,
    }
    for keyword, value in values.items():
        if value is not None:
            keyword.set(header, value)


def read_shutters(header) -> tuple:
    """Shutter states in physical T1..T4 order; None where not recorded.

    Files written before 2026-11 use the non-ESO names 'PLANETES INS SHUTn ST'.
    """
    return tuple(SHUTTER.at(n).get(header, header.get(f"PLANETES INS SHUT{n} ST"))
                 for n in range(1, 5))


def write_shutters(header, shutters) -> None:
    """Write shutter states in physical T1..T4 order."""
    for n, state in enumerate(shutters, 1):
        SHUTTER.at(n).set(header, state)
