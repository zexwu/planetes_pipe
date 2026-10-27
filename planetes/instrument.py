"""Fixed PLANETES detector and output ordering."""

from dataclasses import dataclass

import numpy as np

TELESCOPES = ("4", "3", "2", "1")
BASELINES = ("43", "42", "41", "32", "31", "21")
ABCD_ORDER = ("A", "C", "B", "D")

N_TELESCOPES = len(TELESCOPES)
N_BASELINES = len(BASELINES)
N_OUTPUTS = N_BASELINES * len(ABCD_ORDER)
N_DATA = N_TELESCOPES + 2 * N_BASELINES

REAL_SLICE = slice(N_TELESCOPES, N_TELESCOPES + N_BASELINES)
IMAG_SLICE = slice(REAL_SLICE.stop, N_DATA)


@dataclass(frozen=True, slots=True)
class DetectorSetup:
    """Physical camera configuration; missing legacy fields remain unknown."""

    exposure_s: float | None = None
    exposure_source: str = "readback"
    origin_x: int | None = None
    origin_y: int | None = None
    width: int | None = None
    height: int | None = None
    binning_x: int | None = None
    binning_y: int | None = None
    detector: str | None = None
    orientation: str | None = None
    legacy: bool = False


def check_setup(
    raw: DetectorSetup, calibration: DetectorSetup | None, *, exposure: bool, role: str
) -> tuple[str, ...]:
    """Reject known physical conflicts; only explicitly legacy inputs may lack fields."""
    names = [
        "detector",
        "origin_x",
        "origin_y",
        "width",
        "height",
        "binning_x",
        "binning_y",
        "orientation",
    ]
    if exposure:
        names.append("exposure_s")
    unknown = []
    for name in names:
        actual = getattr(raw, name)
        expected = getattr(calibration, name) if calibration is not None else None
        if actual is None or expected is None:
            if not raw.legacy and not (calibration is not None and calibration.legacy):
                raise ValueError(f"Missing {role} configuration: {name}")
            unknown.append(f"{role}.{name}")
        elif name == "exposure_s":
            # Encoding roundoff only, not a detector/physical matching tolerance.
            if abs(actual - expected) > 4 * abs(
                np.spacing(max(abs(actual), abs(expected)))
            ):
                raise ValueError(
                    f"Incompatible {role} exposure_s: {actual} != {expected}"
                )
        elif actual != expected:
            raise ValueError(f"Incompatible {role} {name}: {actual} != {expected}")
    return tuple(unknown)
