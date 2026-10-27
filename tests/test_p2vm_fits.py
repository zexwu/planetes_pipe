"""Scientific invariants and the pilot calibration FITS boundary."""

from dataclasses import fields, replace
from pathlib import Path
import tempfile
import unittest

import numpy as np
from astropy.io import fits

from planetes.p2vm import build_v2pm_matrix, compute_p2vm
from planetes.instrument import N_DATA
from planetes.fits import read_product, write_product
from p2vm_fixture import calibration_arrays, calibration_spectra


class P2VMFitsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        arrays = calibration_arrays(n_wave=15)
        cls.spectra = calibration_spectra(n_wave=15)
        cls.flat, cls.wave = arrays[-2:]
        cls.p2vm = compute_p2vm(cls.spectra, phase_correction=False)

    def test_p2vm_inverts_known_full_rank_instrument_response(self):
        matrix = build_v2pm_matrix(
            self.p2vm.transmission,
            self.p2vm.coherence_amplitude,
            self.p2vm.coherence_phase_rad,
        )
        for wave in range(len(self.p2vm.wavelength_um)):
            np.testing.assert_allclose(
                self.p2vm.p2vm[:, :, wave] @ matrix[:, :, wave],
                np.eye(N_DATA),
                atol=1e-12,
            )

    def test_fits_round_trip_preserves_all_arrays_and_mappings(self):
        for product in [self.flat, self.wave, self.p2vm]:
            with (
                self.subTest(product=type(product).__name__),
                tempfile.TemporaryDirectory() as tmp,
            ):
                path = Path(tmp) / "calibration.fits"
                write_product(path, product)
                loaded = read_product(path, type(product))
                for field in fields(product):
                    if field.name == "provenance":
                        continue
                    before, after = (
                        getattr(product, field.name),
                        getattr(loaded, field.name),
                    )
                    if isinstance(before, np.ndarray):
                        np.testing.assert_array_equal(before, after)
                        self.assertTrue(after.dtype.isnative)
                    else:
                        self.assertEqual(before, after)

    def test_fits_reader_rejects_changed_wavelength_units(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "wave.fits"
            write_product(path, self.wave)
            with fits.open(path, mode="update") as hdus:
                hdus["WAVELENGTH_MAP_UM"].header["BUNIT"] = "nm"
                hdus["WAVELENGTH_MAP_UM"].add_checksum()
            with self.assertRaisesRegex(ValueError, "units or axes"):
                read_product(path, type(self.wave))

    def test_fits_writer_rejects_changed_telescope_identity(self):
        pairs = dict(self.p2vm.baseline_telescopes)
        pairs[0] = tuple(reversed(pairs[0]))
        product = replace(self.p2vm, baseline_telescopes=pairs)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "identities"):
                write_product(Path(tmp) / "bad.fits", product)

    def test_fits_writer_preserves_existing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "wave.fits"
            write_product(path, self.wave)
            original = path.read_bytes()
            with self.assertRaises(OSError):
                write_product(path, self.wave)
            self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
