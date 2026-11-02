"""One science cube + an existing calibration directory -> instrumental observables."""
from astropy.io import fits
from pathlib import Path

from planetes_pipe import io
from planetes_pipe.io.common import product_filename
from planetes_pipe.io.raw import read_raw_metadata
from planetes_pipe.model.products import Dark, Flat, Wave, P2VM
from planetes_pipe.processing.science import reduce_science, measure_quality

SCIENCE = Path("/path/to/science.fits")
CALIBRATIONS = Path("/path/to/reduced/calibration")
OUTPUT = CALIBRATIONS.parent / "science"


def main():
    paths = list(CALIBRATIONS.glob("*.fits"))
    categories = {path: fits.getheader(path).get("ESO PRO CATG") for path in paths}
    (flat_path,) = (path for path in paths if categories[path] == "FLAT")
    (wave_path,) = (path for path in paths if categories[path] == "WAVE")
    (p2vm_path,) = (path for path in paths if categories[path] == "P2VM")
    flat, wave, p2vm = (
        io.flat.read(flat_path),
        io.wave.read(wave_path),
        io.p2vm.read(p2vm_path),
    )
    setup = read_raw_metadata(SCIENCE, category="SINGLE_SCI_RAW").setup
    if setup.exposure_s is None:
        raise ValueError("Select a dark explicitly when science exposure is unknown")
    darks = [io.dark.read(path) for path in paths if categories[path] == "DARK"]
    (dark,) = (
        p
        for p in darks
        if p.setup is not None and p.setup.exposure_s == setup.exposure_s
    )
    # The step also checks detector/ROI compatibility before subtracting the dark.
    with io.raw.read(SCIENCE, category="SINGLE_SCI_RAW") as raw_input:
        result = reduce_science(raw_input, dark, flat, wave, p2vm)
    path = OUTPUT / product_filename(SCIENCE, result)
    io.science.write(path, result, quality=measure_quality(result))
    print(path)


if __name__ == "__main__":
    main()
