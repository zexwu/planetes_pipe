"""One science cube + an existing calibration directory -> instrumental observables."""
from pathlib import Path

from astropy.io import fits

from planetes_pipe.io.keywords import DIT, PRO_CATG
from planetes_pipe.io.raw import read_raw
from planetes_pipe.steps import science

SCIENCE = Path("/path/to/science.fits")
CALIBRATIONS = Path("/path/to/reduced/calibration")
OUTPUT = CALIBRATIONS.parent / "science"


def main():
    raw = read_raw(SCIENCE, "SINGLE_SCI_RAW")
    headers = {path: fits.getheader(path) for path in sorted(CALIBRATIONS.glob("*.fits"))}

    def product(category, *same):
        """The single product of this category; `same` keywords must equal the science's."""
        (path,) = (p for p, h in headers.items() if PRO_CATG.get(h) == category
                   and all(k.get(h) == k.get(raw.header) for k in same))
        return path

    if DIT.get(raw.header) is None:
        raise ValueError("Select a dark explicitly when the science DIT is not recorded")
    # The step also checks the detector window before subtracting the dark.
    reduced = science.run(SCIENCE, product("DARK", DIT), product("FLAT"), product("WAVE"),
                          product("P2VM"), output_dir=OUTPUT)
    print(reduced["SINGLE_SCI_P2VMRED"])


if __name__ == "__main__":
    main()
