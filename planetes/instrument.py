"""Fixed PLANETES detector and output ordering."""

from dataclasses import dataclass

import numpy as np

# fmt: off
TELESCOPES = ("4", "3", "2", "1")
BASELINES  = ("43", "42", "41", "32", "31", "21")
# fmt: on
# Telescope-axis indices, in the fixed physical baseline order above.
BASELINE_PAIRS = tuple(
    (TELESCOPES.index(pair[0]), TELESCOPES.index(pair[1])) for pair in BASELINES
)
ABCD_ORDER = ("A", "C", "B", "D")

# fmt: off
N_TELESCOPES = len(TELESCOPES)
N_BASELINES  = len(BASELINES)
N_OUTPUTS    = N_BASELINES * len(ABCD_ORDER)
N_DATA       = N_TELESCOPES + 2 * N_BASELINES
# fmt: on

REAL_SLICE = slice(N_TELESCOPES, N_TELESCOPES + N_BASELINES)
IMAG_SLICE = slice(REAL_SLICE.stop, N_DATA)


def _check_baselines(baseline_outputs: dict[int, list[int]]) -> None:
    """Require one ABCD assignment for every output in the fixed baseline order."""
    if (
        set(baseline_outputs) != set(range(N_BASELINES))
        or any(len(outputs) != len(ABCD_ORDER) for outputs in baseline_outputs.values())
        or sorted(output for outputs in baseline_outputs.values() for output in outputs)
        != list(range(N_OUTPUTS))
    ):
        raise ValueError("Incompatible physical telescope/baseline output mapping")


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
