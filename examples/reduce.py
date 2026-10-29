"""Reduce the standardized 2026-02-05 bench data with the current FITS products.

Run from this repository: .venv/bin/python examples/reduce.py
The sibling dataset supplies raw/calibration and raw/science. Choose a fresh
output directory below; existing raw and reduced files are never overwritten.
"""

# %% Select this night's raw inputs and a new output directory.
from pathlib import Path

from astropy.io import fits

from planetes_pipe import configure_logging
from planetes_pipe.dark import make_dark_products
from planetes_pipe.flat import make_flat
from planetes_pipe.instrument import TELESCOPES
from planetes_pipe.io.fits import product_filename, read_product, write_product
from planetes_pipe.io.keywords import DPR_TYPE
from planetes_pipe.io.raw import read_raw_metadata
from planetes_pipe.p2vm import make_p2vm
from planetes_pipe.products import Dark, Flat, P2VM, Wave, WavelengthReference
from planetes_pipe.qc import measure_quality
from planetes_pipe.reduction import reduce_science
from planetes_pipe.wavelength import make_wave, wavelength_reference_path

night = Path(__file__).resolve().parents[2] / "2026_02_05"
calibration_raw = night / "raw/calibration"
science_raw = night / "raw/science"
output = night / "reduced_example"

# These are explicit exposure selections for this dataset. UNKNOWN.0001 is an
# import identifier, not an inferred exposure time or shutter configuration.
main_dark_path = calibration_raw / "PLANETES.2026-02-05TUNKNOWN.0001.fits"
lamp_dark_path = calibration_raw / "PLANETES.2026-02-05T210014.fits"
science_dark_path = science_raw / "PLANETES.2026-02-05T210637.fits"
lamp_path = calibration_raw / "PLANETES.2026-02-05T205537.fits"
science_path = science_raw / "PLANETES.2026-02-05T210609.fits"

# %% Identify the four telescope flats and six baseline scans from their headers.
telescope_flats = {}
baseline_cubes = {}
for path in sorted(calibration_raw.glob("*.fits")):
    dpr_type = fits.getheader(path)[DPR_TYPE]
    if dpr_type == "FLAT":
        category, selected = "FLAT_RAW", telescope_flats
    elif dpr_type == "P2VM":
        category, selected = "P2VM_RAW", baseline_cubes
    else:
        continue
    metadata = read_raw_metadata(path, category=category)
    if None in metadata.shutters:
        raise ValueError(f"Missing physical shutter states: {path}")
    identity = "".join(tel for tel in TELESCOPES if metadata.shutters[int(tel) - 1])
    if identity in selected:
        raise ValueError(
            f"Multiple {category} files for {identity}; select one sequence"
        )
    selected[identity] = path

# %% Measure the three selected darks, then fit extraction geometry and wavelength.
output.mkdir(parents=True, exist_ok=False)
configure_logging()
main_dark, bad = make_dark_products(main_dark_path)
lamp_dark, _ = make_dark_products(lamp_dark_path)
science_dark, _ = make_dark_products(science_dark_path)
flat = make_flat(telescope_flats, main_dark, bad)
reference = read_product(wavelength_reference_path(), WavelengthReference)
wave = make_wave(lamp_path, lamp_dark, flat, reference=reference)

# %% Fit the instrumental P2VM. This dataset has no selected phase reference.
p2vm = make_p2vm(
    flat,
    wave,
    main_dark,
    telescope_flats=telescope_flats,
    baseline_cubes=baseline_cubes,
)

# %% Save calibrations with the complete raw stem and the product category.
for reference_path, product in [
    (main_dark_path, main_dark),
    (main_dark_path, bad),
    (lamp_dark_path, lamp_dark),
    (science_dark_path, science_dark),
    (telescope_flats["4"], flat),
    (lamp_path, wave),
    (baseline_cubes["43"], p2vm),
]:
    path = output / product_filename(reference_path, product)
    write_product(path, product, quality=measure_quality(product))
    print(path.name)

# %% Use the same on-disk calibrations as the bench viewer, including float32 storage.
science_dark = read_product(
    output / product_filename(science_dark_path, science_dark), Dark
)
flat = read_product(output / product_filename(telescope_flats["4"], flat), Flat)
wave = read_product(output / product_filename(lamp_path, wave), Wave)
p2vm = read_product(output / product_filename(baseline_cubes["43"], p2vm), P2VM)
reduced = reduce_science(science_path, science_dark, flat, wave, p2vm)
path = output / product_filename(science_path, reduced)
write_product(path, reduced, quality=measure_quality(reduced))
print(path.name)
