"""Fixed PLANETES layout, detector setup and raw cubes."""

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class InstrumentLayout:
    """Physical output order and the corresponding P2VM matrix axes."""

    telescopes: tuple[str, ...] = ("4", "3", "2", "1")
    baselines: tuple[str, ...] = ("43", "42", "41", "32", "31", "21")
    abcd_order: tuple[str, ...] = ("A", "C", "B", "D")

    @property
    def baseline_pairs(self) -> tuple[tuple[int, int], ...]:
        """Telescope-axis indices for each physical baseline."""
        return tuple((self.telescopes.index(b[0]), self.telescopes.index(b[1]))
                     for b in self.baselines)

    @property
    def n_telescopes(self) -> int:
        return len(self.telescopes)

    @property
    def n_baselines(self) -> int:
        return len(self.baselines)

    @property
    def n_outputs(self) -> int:
        return self.n_baselines * len(self.abcd_order)

    @property
    def n_data(self) -> int:
        return self.n_telescopes + 2 * self.n_baselines

    @property
    def real_slice(self) -> slice:
        return slice(self.n_telescopes, self.n_telescopes + self.n_baselines)

    @property
    def imag_slice(self) -> slice:
        return slice(self.real_slice.stop, self.n_data)


layout = InstrumentLayout()


@dataclass(frozen=True, slots=True)
class DetectorSetup:
    """Physical camera configuration; missing fields remain unknown."""

    exposure_s: float | None = None
    origin_x: int | None = None
    origin_y: int | None = None
    width: int | None = None
    height: int | None = None
    binning_x: int | None = None
    binning_y: int | None = None
    detector: str | None = None


def check_setup(
    raw: DetectorSetup, calibration: DetectorSetup | None, *, exposure: bool, role: str
) -> None:
    """Reject contradictions between known raw and selected calibration settings.

    Parameters
    ----------
    raw : DetectorSetup
        Physical configuration of the current raw cube.
    calibration : DetectorSetup or None
        Configuration associated with the selected calibration; None means unknown.
    exposure : bool
        Also compare integration times when True, as required for direct dark
        subtraction. False compares detector, ROI and binning only.
    role : str
        Calibration label used in error messages, such as 'DARK' or 'FLAT'.

    Notes
    -----
    Missing fields are skipped and are not filled or inferred. Exposure comparison
    allows only floating-point encoding roundoff. Raises ValueError for known
    contradictions, otherwise returns None. Array dimensions require separate checks
    when the calibration lacks metadata.
    """
    names = [
        "detector",
        "origin_x",
        "origin_y",
        "width",
        "height",
        "binning_x",
        "binning_y",
    ]
    if exposure:
        names.append("exposure_s")
    for name in names:
        actual = getattr(raw, name)
        expected = getattr(calibration, name) if calibration is not None else None
        if actual is None or expected is None:
            continue
        if name == "exposure_s":
            # Encoding roundoff only, not a detector/physical matching tolerance.
            if abs(actual - expected) > 4 * abs(
                np.spacing(max(abs(actual), abs(expected)))
            ):
                raise ValueError(
                    f"Incompatible {role} exposure_s: {actual} != {expected}"
                )
        elif actual != expected:
            raise ValueError(f"Incompatible {role} {name}: {actual} != {expected}")


@dataclass(frozen=True, slots=True)
class Raw:
    """One raw detector cube and its recorded metadata.

    cube is the (frame, y, x) ADU image HDU as stored, memory-mapped by
    files.read_raw, so large files are not loaded at once. Shutters are in
    physical T1,T2,T3,T4 order; unknown metadata stays None.
    """

    cube: NDArray
    category: str
    shutters: tuple[bool | None, bool | None, bool | None, bool | None]
    setup: DetectorSetup

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


def shutter_identity(shutters: tuple[bool | None, ...]) -> str:
    """Open shutter numbers in descending physical order, e.g. '4' or '43'."""
    if None in shutters:
        raise ValueError("Missing shutter states; cannot identify calibration input")
    return "".join(str(n) for n in range(4, 0, -1) if shutters[n - 1])


def order_raws(raws: list[Raw], identities: tuple[str, ...]) -> list[Raw]:
    """Order a calibration sequence by recorded shutters, rejecting missing/duplicate inputs."""
    grouped = {}
    for raw in raws:
        identity = shutter_identity(raw.shutters)
        if identity in grouped:
            raise ValueError(f"Duplicate calibration shutter state: {identity}")
        grouped[identity] = raw
    if set(grouped) != set(identities):
        raise ValueError(f"Expected shutter configurations {identities}; found {tuple(grouped)}")
    return [grouped[key] for key in identities]
