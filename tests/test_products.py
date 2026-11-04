"""products.py: product FITS round trips, file names and no-overwrite."""

import tempfile
import unittest
from dataclasses import fields
from pathlib import Path

import numpy as np
from astropy.io import fits
from fixtures import (
    calibration_arrays,
    calibration_spectra,
    write_product,
)

from planetes_pipe.flat import Flat, read_flat
from planetes_pipe.p2vm import P2VM, fit_p2vm, read_p2vm
from planetes_pipe.products import product_path
from planetes_pipe.wave import Wave, read_wave, write_wave

READERS = {Flat: read_flat, Wave: read_wave, P2VM: read_p2vm}


class CalibrationFitsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        arrays = calibration_arrays(n_wave=15)
        cls.spectra = calibration_spectra(n_wave=15)
        cls.flat, cls.wave = arrays[-2:]
        cls.p2vm = fit_p2vm(cls.spectra)
        cls.p2vm_diagnostics = fit_p2vm(cls.spectra, keep_diagnostics=True)

    def test_round_trip_is_exact_and_keeps_mappings(self):
        for product in [self.flat, self.wave, self.p2vm, self.p2vm_diagnostics]:
            with (
                self.subTest(product=type(product).__name__),
                tempfile.TemporaryDirectory() as tmp,
            ):
                path = Path(tmp) / "calibration.fits"
                write_product(path, product)
                with fits.open(path, mode="update") as hdus:
                    self.assertNotIn("PLANETES_META", hdus)
                    self.assertNotIn("PLANETES FORMAT", hdus[0].header)
                    # Product layout, not a private version number, determines reading.
                    hdus[0].header["HIERARCH PLANETES FORMAT"] = -1
                    hdus[0].add_checksum()
                loaded = READERS[type(product)](path)
                for field in fields(product):
                    before, after = (
                        getattr(product, field.name),
                        getattr(loaded, field.name),
                    )
                    if isinstance(before, np.ndarray):
                        np.testing.assert_array_equal(before, after)  # Stored as computed.
                        self.assertEqual(after.dtype, before.dtype)
                        self.assertTrue(after.dtype.isnative)
                    else:
                        self.assertEqual(before, after)

    def test_fits_writer_preserves_existing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "wave.fits"
            write_wave(path, self.wave)
            header = fits.getheader(path)
            self.assertEqual(header["ESO QC MINWAVE SC"],
                             float(self.wave.wavelength_map_m.min(axis=1).max()))
            original = path.read_bytes()
            with self.assertRaises(OSError):
                write_wave(path, self.wave)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(Path(tmp).glob(".planetes-*")), [])


class ProductPathTests(unittest.TestCase):
    def test_product_path_preserves_reference_identity(self):
        for stem in (
            "PLANETES.2026-02-05T210609",
            "PLANETES.2026-02-05T21:06:09.123",
            "PLANETES.2026-02-05TUNKNOWN.0006",
        ):
            with self.subTest(stem=stem):
                reference = Path("/unread/raw") / f"{stem}.fits"
                self.assertEqual(
                    product_path("out", reference, "SINGLE_SCI_P2VMRED").name,
                    f"{stem}_singlescip2vmred.fits",
                )
                self.assertEqual(
                    product_path("out", reference, "DARK").name,
                    f"{stem}_dark.fits",
                )


if __name__ == "__main__":
    unittest.main()
