"""Single-frame timing with the same scientific core used by the formal pipeline."""

from pathlib import Path
from time import perf_counter

from planetes.dark import load_dark
from planetes.fits import iter_raw_cube
from planetes.reduction import FrameReducer


def main():
    project = Path(__file__).resolve().parents[1]
    calibrations = project / "reduced/calibrations"
    science = project / "2026_02_05/sci2/sci.fits"
    dark = load_dark(project / "2026_02_05/sci2/sci_dark.fits")
    reducer = FrameReducer.from_directory(calibrations)
    raw = next(iter_raw_cube(science))[0]
    reducer.reduce_frame(
        raw, dark=dark
    )  # Warm JIT before timing the actual frame workload.
    start = perf_counter()
    for _ in range(128):
        result = reducer.reduce_frame(raw, dark=dark)
    print(f"Mean processing time: {(perf_counter()-start)/128*1000:.3f} ms/frame")
    print(f"Invalid visibility samples: {result.flags.sum()}/{result.flags.size}")


if __name__ == "__main__":
    main()
