"""Scientific FITS round trips, raw reading and non-overwriting publication."""

import tempfile
import unittest
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
from astropy.io import fits
from p2vm_fixture import calibration_arrays, calibration_spectra

from planetes.io.fits import read_product, write_product
from planetes.io.raw import iter_raw_cube as iter_cube
from planetes.io.raw import (
    read_raw_header,
    read_raw_metadata,
)
from planetes.p2vm import compute_p2vm
from planetes.products import ReducedObservables
from planetes.reduction import reduce_spectra


class ScienceProductTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spectra = calibration_spectra(n_wave=15)
        p2vm = compute_p2vm(spectra, phase_correction=False)
        science = spectra.phase_reference.copy()
        science[:, 0] = 0
        cls.product = reduce_spectra(science, p2vm)

    def test_complex_flux_and_invalid_flags_round_trip_exactly(self):
        coherent = self.product.coherent_flux_adu.copy()
        coherent[0, 0, 0] = complex(-0.0, -0.0)
        coherent[0, 0, 1] = complex(np.nan, np.nan)
        product = replace(self.product, coherent_flux_adu=coherent)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "science.fits"
            write_product(path, product)
            loaded = read_product(path, ReducedObservables)
            for field in fields(product):
                if field.name == "provenance":
                    continue
                before, after = (
                    getattr(product, field.name),
                    getattr(loaded, field.name),
                )
                if isinstance(before, np.ndarray):
                    expected = before
                    np.testing.assert_array_equal(expected, after)
                else:
                    self.assertEqual(before, after)
            np.testing.assert_array_equal(
                coherent.view(np.uint64), loaded.coherent_flux_adu.view(np.uint64)
            )
            self.assertTrue(loaded.flags[0].all())
            self.assertEqual(loaded.flags.dtype, bool)


class CalibrationFitsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        arrays = calibration_arrays(n_wave=15)
        cls.spectra = calibration_spectra(n_wave=15)
        cls.flat, cls.wave = arrays[-2:]
        cls.p2vm = compute_p2vm(cls.spectra, phase_correction=False)

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
                        expected = before
                        np.testing.assert_array_equal(expected, after)
                        self.assertTrue(after.dtype.isnative)
                    else:
                        self.assertEqual(before, after)

    def test_fits_writer_preserves_existing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "wave.fits"
            write_product(path, self.wave)
            original = path.read_bytes()
            with self.assertRaises(OSError):
                write_product(path, self.wave)
            self.assertEqual(path.read_bytes(), original)


class RawFitsTests(unittest.TestCase):
    def test_bench_cubes_keep_missing_metadata_and_all_frames(self):
        cube = np.arange(17 * 3 * 5, dtype=np.int16).reshape(17, 3, 5)
        for primary in (True, False):
            with self.subTest(primary=primary), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "bench.fits"
                hdus = (
                    [fits.PrimaryHDU(cube)]
                    if primary
                    else [fits.PrimaryHDU(), fits.ImageHDU(cube)]
                )
                fits.HDUList(hdus).writeto(path)
                original = path.read_bytes()
                raw = read_raw_metadata(path, category="DARK_RAW")
                self.assertIsNone(raw.shutters)
                self.assertIsNone(raw.setup.exposure_s)
                self.assertEqual((raw.setup.height, raw.setup.width), (3, 5))
                self.assertTrue(raw.setup.legacy)
                self.assertIn("RAW.BUNIT", raw.original["unverified"])
                self.assertNotIn("ESO DET WIN1 DIT1", read_raw_header(path))
                chunks = list(iter_cube(path))
                self.assertEqual([len(chunk) for chunk in chunks], [16, 1])
                np.testing.assert_array_equal(np.concatenate(chunks), cube)
                self.assertEqual(path.read_bytes(), original)
