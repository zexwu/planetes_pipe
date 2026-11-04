"""Raw data: the detector cube, its header (shutters, detector setup) and how to read it.

    from planetes_pipe.raw import read_raw, read_raws
    lamp = read_raw("lamp.fits", "WAVELAMP_RAW")
    flats = read_raws(flat_files, "FLAT_RAW", layout.telescopes)   # {"4": Raw, ...}

The DET and shutter header encoding here is also used for product headers and by
the bench when it writes raw files.
"""

from collections.abc import Iterator
from dataclasses import dataclass, fields, replace
from pathlib import Path

import numpy as np
from astropy.io import fits as afits
from numpy.typing import NDArray

from planetes_pipe.instrument import DetectorSetup, check_setup, shutter_identity

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
    setup = replace(read_setup(header) or DetectorSetup(), height=cube.shape[1], width=cube.shape[2])
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
    """Raw files of one category in a directory, by ESO DPR TYPE (not filename)."""
    return [path for path in sorted(Path(directory).glob("*.fits"))
            if (header := afits.getheader(path)).get("ESO DPR TYPE") == RAW_TYPES[category]
            and "ESO PRO CATG" not in header]


def check_calibrations(raw: Raw, *, dark=None, flat=None) -> None:
    """Reject a DARK whose known DIT/ROI/detector, or a FLAT whose ROI/detector, differs from raw."""
    if dark is not None:
        check_setup(raw.setup, dark.setup, exposure=True, role="DARK")
    if flat is not None:
        check_setup(raw.setup, flat.setup, exposure=False, role="FLAT")


def read_setup(hdr) -> DetectorSetup | None:
    """Decode detector setup; FITS origins are one-based, Python origins zero-based."""
    x, y = hdr.get("ESO DET WIN1 STRX"), hdr.get("ESO DET WIN1 STRY")
    setup = DetectorSetup(
        origin_x=None if x is None else x - 1,
        origin_y=None if y is None else y - 1,
        width=hdr.get("ESO DET WIN1 NX"),
        height=hdr.get("ESO DET WIN1 NY"),
        binning_x=hdr.get("ESO DET WIN1 BINX"),
        binning_y=hdr.get("ESO DET WIN1 BINY"),
        exposure_s=hdr.get("ESO DET WIN1 DIT1", hdr.get("ESO DET WIN1 UIT1")),
        detector=hdr.get("ESO DET ID"),
    )
    return setup if any(getattr(setup, f.name) is not None for f in fields(setup)) else None


def write_setup(header, setup: DetectorSetup) -> None:
    """Write the known detector settings; ROI origins become one-based."""
    origin_x = setup.origin_x + 1 if setup.origin_x is not None else None
    origin_y = setup.origin_y + 1 if setup.origin_y is not None else None
    cards = {
        "ESO DET WIN1 STRX": origin_x,
        "ESO DET WIN1 STRY": origin_y,
        "ESO DET WIN1 NX": setup.width,
        "ESO DET WIN1 NY": setup.height,
        "ESO DET WIN1 BINX": setup.binning_x,
        "ESO DET WIN1 BINY": setup.binning_y,
        "ESO DET ID": setup.detector,
    }
    for key, value in cards.items():
        if value is not None:
            header["HIERARCH " + key] = value
    if setup.exposure_s is not None:
        header["HIERARCH ESO DET WIN1 DIT1"] = (setup.exposure_s, "[s] Integration time")


def read_shutters(header):
    """Shutter states in physical T1..T4 order.

    Files written before 2026-11 use the non-ESO 'PLANETES INS SHUTn ST' names.
    """
    return tuple(header.get(f"ESO INS SHUT{n} ST", header.get(f"PLANETES INS SHUT{n} ST"))
                 for n in range(1, 5))


def write_shutters(header, shutters) -> None:
    """Write shutter states in physical T1..T4 order."""
    for n, state in enumerate(shutters, 1):
        header[f"HIERARCH ESO INS SHUT{n} ST"] = state
