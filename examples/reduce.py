"""Edit the three directories below, then run this file for instrumental reduction."""

# %% Select raw inputs and a new output directory (bench values follow the ADU convention).
from pathlib import Path

from planetes.dark import calibrate_dark_file
from planetes.flat import calibrate_flat_file
from planetes.io.fits import write_product
from planetes.p2vm import calibrate_p2vm_file
from planetes.reduction import reduce_file
from planetes.wavelength import calibrate_wavelength_file

calibration_raw = Path("/absolute/calibration-raw")
science_raw = Path("/absolute/science-raw")
output = Path("/absolute/new-output")
output.mkdir(parents=True, exist_ok=False)

# %% Measure the dark appropriate to each data group.
main_dark, bad = calibrate_dark_file(calibration_raw / "dark.fits")
lamp_dark, _ = calibrate_dark_file(calibration_raw / "wave_dark.fits")
science_dark, _ = calibrate_dark_file(science_raw / "sci_dark.fits")

# %% Calibrate extraction geometry, wavelengths and the instrumental response.
# These dictionaries explicitly assign physical identities, even without shutter headers.
telescope_flats = {n: calibration_raw / f"tel{n}.fits" for n in ("4", "3", "2", "1")}
baseline_cubes = {
    n: calibration_raw / f"bsl{n}.fits" for n in ("43", "42", "41", "32", "31", "21")
}
flat = calibrate_flat_file(telescope_flats, main_dark, bad)
wave = calibrate_wavelength_file(calibration_raw / "wave.fits", lamp_dark, flat)
# Add an explicit phase_reference and matching phase_dark here when available.
p2vm = calibrate_p2vm_file(
    flat,
    wave,
    main_dark,
    telescope_flats=telescope_flats,
    baseline_cubes=baseline_cubes,
)

# %% Extract all science frames on the saved P2VM wavelength grid.
frames = reduce_file(science_raw / "sci.fits", science_dark, flat, wave, p2vm)

# %% Save the selected calibrations and per-frame instrumental observables.
for filename, product in [
    ("planetes_dark.fits", main_dark),
    ("planetes_bad.fits", bad),
    ("planetes_lamp_dark.fits", lamp_dark),
    ("planetes_science_dark.fits", science_dark),
    ("planetes_flat.fits", flat),
    ("planetes_wave.fits", wave),
    ("planetes_p2vm.fits", p2vm),
    ("planetes_reduced.fits", frames),
]:
    write_product(output / filename, product, include_qc=True)
