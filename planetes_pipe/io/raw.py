"""Raw data: the detector cube, its recorded header, and how to read it.

    from planetes_pipe.io.raw import read_raw, read_raws
    lamp = read_raw("lamp.fits", "WAVELAMP_RAW")
    flats = read_raws(flat_files, "FLAT_RAW", layout.telescopes)   # {"4": Raw, ...}

The detector setup of a raw file is its DET cards (DICD §4.4.2.7), used as
recorded: a calibration carries the DET cards of the raw it was made from, and
check_calibrations compares those cards directly. The bench writes the same cards.
"""

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits as afits
from numpy.typing import NDArray

from planetes_pipe.io.keywords import (
    DIT,
    DPR_TYPE,
    MJD_OBS,
    PRO_CATG,
    QC_SATURATED,
    SHUTTER,
    TPL_START,
    WINDOW,
)

log = logging.getLogger(__name__)

# Raw value at which the detector's 12-bit readout clips. A clipped sample is not
# a measurement: fringes and fluxes computed from it are biased low.
SATURATION_ADU = 4095
# A DARK must share the raw's detector window and DIT (check_calibrations).
DARK_SETUP = (*WINDOW, DIT)

# Raw SOF tag -> ESO DPR TYPE of the raw file.
RAW_TYPES = {
    "DARK_RAW": "DARK",
    "FLAT_RAW": "FLAT",
    "P2VM_RAW": "P2VM",
    "WAVELAMP_RAW": "WAVE,LAMP",
    "SINGLE_SCI_RAW": "OBJECT,SINGLE",
}


class CompressedCube:
    """A tile-compressed raw cube, decompressed only for the frames asked for.

    Supports what Raw needs: len, shape, dtype, slicing (cube[start:stop],
    cube[f, y, x]) and np.asarray (which decompresses everything). The file is
    opened for each access, so it is never held open.
    """

    def __init__(self, path: Path, hdu: int | str, shape: tuple, dtype):
        self.path, self.hdu, self.shape, self.dtype = Path(path), hdu, tuple(shape), dtype

    def __len__(self) -> int:
        return self.shape[0]

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def __getitem__(self, key):
        with afits.open(self.path) as hdus:
            return hdus[self.hdu].section[key]

    def __array__(self, dtype=None, copy=None):
        data = self[:]
        return data if dtype is None else data.astype(dtype)


@dataclass(frozen=True, slots=True)
class Raw:
    """One raw detector cube with its primary header.

    cube is the (frame, y, x) ADU image as stored: memory-mapped, or decompressed
    on demand for tile-compressed files (CompressedCube), so that large files are
    not loaded at once. Read frames through chunks().
    """

    cube: NDArray
    category: str
    header: afits.Header
    path: Path | None = None

    @property
    def shutters(self) -> tuple:
        """Recorded shutter states, physical S1..S4 order; None where not recorded."""
        return read_shutters(self.header)

    @property
    def detector(self) -> afits.Header:
        """The DET cards of this raw file."""
        return detector_cards(self.header)

    def chunks(self, size: int = 16) -> Iterator[NDArray]:
        """Frames in order, as float64 blocks of up to `size` frames."""
        for start in range(0, len(self.cube), size):
            yield np.asarray(self.cube[start : start + size], dtype=np.float64)

    def saturated(self, pixels: NDArray | None = None) -> int:
        """Number of samples at or above SATURATION_ADU, over all frames.

        pixels is an optional boolean (y, x) mask, e.g. Flat.extracted_pixels(),
        to count only the pixels a reduction uses.
        """
        where = True if pixels is None else pixels
        return sum(int(np.count_nonzero((self.cube[start : start + 16] >= SATURATION_ADU) & where))
                   for start in range(0, len(self.cube), 16))

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
    """Open a raw cube with its primary header, without loading the frames.

    category is the input role chosen by the caller (e.g. DARK_RAW). hdu is the
    image HDU; default IMAGING_DATA_SC, 0 for primary-HDU data. Plain images are
    memory-mapped; tile-compressed ones (bench files, Rice) are read in blocks.
    The file is read only.
    """
    with afits.open(path, memmap=True) as hdus:
        image, header = hdus[hdu], hdus[0].header.copy()
        if isinstance(image, afits.CompImageHDU):
            cube = CompressedCube(path, hdu, image.shape, image.section[0:1].dtype)
        else:
            cube = image.data
    return Raw(cube, category, header, Path(path))


def read_raws(paths, category: str, identities: tuple[str, ...], *,
              hdu: int | str = "IMAGING_DATA_SC") -> dict[str, Raw]:
    """Read a calibration sequence keyed by physical identity, e.g. {"4": Raw, "3": Raw, ...}.

    Identity comes from the recorded shutters ('4' = only S4 open, '43' = S4 and S3),
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


def closest_raw(raw_file, candidates, keywords, what: str = "calibration") -> Path:
    """The candidate raw file that best matches raw_file, by header.

    Compatible first: no keyword of `keywords` differs (a value recorded on only one
    side is not compared). Among those, prefer the exact setup, then the same capture
    run (TPL.START), then the closest MJD-OBS. Raise LookupError if none is
    compatible, or if several are and the times needed to choose are not recorded.
    """
    target = afits.getheader(raw_file)
    wanted = [k.get(target) for k in keywords]
    headers = {Path(c): afits.getheader(c) for c in candidates}
    same = [c for c, h in headers.items()
            if all(a is None or b is None or a == b
                   for a, b in zip((k.get(h) for k in keywords), wanted))]
    if not same:
        raise LookupError(f"No {what} with the setup of {Path(raw_file).name}")
    run_start = TPL_START.get(target)
    for preferred in (lambda c: [k.get(headers[c]) for k in keywords] == wanted,
                      lambda c: run_start is not None and TPL_START.get(headers[c]) == run_start):
        same = [c for c in same if preferred(c)] or same
    mjd = MJD_OBS.get(target)
    if len(same) > 1 and (mjd is None or any(MJD_OBS.get(headers[c]) is None for c in same)):
        raise LookupError(f"{len(same)} {what} files match {Path(raw_file).name} but MJD-OBS "
                          f"is not recorded: {', '.join(c.name for c in same)}")
    chosen = min(same, key=lambda c: 0.0 if mjd is None else abs(MJD_OBS.get(headers[c]) - mjd))
    apart = ("" if mjd is None or MJD_OBS.get(headers[chosen]) is None
             else f", {abs(MJD_OBS.get(headers[chosen]) - mjd) * 86400:.0f} s apart")
    dit = DIT.get(headers[chosen])
    log.info("%s for %s: %s (DIT %s%s)", what, Path(raw_file).name, chosen.name,
             "unknown" if dit is None else f"{dit:.4g} s", apart)
    return chosen


def find_dark(raw_file, directory) -> Path:
    """The DARK_RAW in directory for raw_file: same window and DIT, then same run, then closest time."""
    return closest_raw(raw_file, find_raw(directory, "DARK_RAW"), DARK_SETUP, "DARK_RAW")


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
        check_match(raw.header, dark.detector, DARK_SETUP, "DARK")
    if flat is not None:
        check_match(raw.header, flat.detector, WINDOW, "FLAT")


def record_saturation(header: afits.Header, raws: dict[str, Raw], pixels=None) -> None:
    """Write QC.NSAT.<label> for each raw input into a product header; warn if any is clipped.

    raws maps a short label to each input, e.g. {"SC43": scan, "S4": flat}; pixels
    restricts the count to the extracted pixels (Flat.extracted_pixels()) when known.
    """
    for label, raw in raws.items():
        count = raw.saturated(pixels)
        QC_SATURATED.at(label).set(header, count)
        if count:
            name = raw.path.name if raw.path else label
            log.warning(
                "%s: %d samples at the %d ADU saturation level; results from those pixels "
                "are biased. Lower the exposure or the source.", name, count, SATURATION_ADU)


def read_shutters(header: afits.Header) -> tuple:
    """Shutter states in physical S1..S4 order; None where not recorded.

    Files written before 2026-11 use the non-ESO names 'PLANETES INS SHUTn ST'.
    """
    return tuple(SHUTTER.at(n).get(header, header.get(f"PLANETES INS SHUT{n} ST"))
                 for n in range(1, 5))


def write_shutters(header: afits.Header, shutters) -> None:
    """Write shutter states in physical S1..S4 order."""
    for n, state in enumerate(shutters, 1):
        SHUTTER.at(n).set(header, state)


def shutter_identity(shutters: tuple) -> str:
    """Open shutter numbers in descending physical order, e.g. '4' or '43'."""
    if None in shutters:
        raise ValueError("Missing shutter states; cannot identify calibration input")
    return "".join(str(n) for n in range(4, 0, -1) if shutters[n - 1])
