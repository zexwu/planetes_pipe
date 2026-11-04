"""One science cube + an existing calibration directory -> instrumental observables."""
from pathlib import Path

from astropy.io import fits

from planetes_pipe import science
from planetes_pipe.raw import read_raw

SCIENCE = Path("/path/to/science.fits")
CALIBRATIONS = Path("/path/to/reduced/calibration")
OUTPUT = CALIBRATIONS.parent / "science"


def main():
    headers = {path: fits.getheader(path) for path in CALIBRATIONS.glob("*.fits")}

    def product(category, **keywords):
        """The single calibration of this category matching the given header values."""
        (path,) = (p for p, h in headers.items() if h.get("ESO PRO CATG") == category
                   and all(h.get(k) == v for k, v in keywords.items()))
        return path

    exposure = read_raw(SCIENCE, "SINGLE_SCI_RAW").setup.exposure_s
    if exposure is None:
        raise ValueError("Select a dark explicitly when science exposure is unknown")
    # The step also checks detector/ROI compatibility before subtracting the dark.
    reduced = science.run(
        SCIENCE,
        product("DARK", **{"ESO DET WIN1 DIT1": exposure}),
        product("FLAT"),
        product("WAVE"),
        product("P2VM"),
        output_dir=OUTPUT,
    )
    print(reduced["SINGLE_SCI_P2VMRED"])


if __name__ == "__main__":
    main()
