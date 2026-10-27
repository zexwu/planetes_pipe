"""Scientific FITS round trips, raw reading and non-overwriting publication."""

import tempfile
import unittest
from dataclasses import fields, replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
from astropy.io import fits
from p2vm_fixture import calibration_arrays, calibration_spectra

from planetes.dark import frame_statistics
from planetes.fits import (
    fill_dfs_product,
    product_destination,
    read_product,
    write_product,
)
from planetes.fits import iter_raw_cube as iter_cube
from planetes.models import ReducedObservables
from planetes.p2vm import compute_p2vm
from planetes.reduction import compute_reduce
from planetes.wavelength import load_wavelength_reference


class ScienceProductTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spectra = calibration_spectra(n_wave=15)
        p2vm = compute_p2vm(spectra, phase_correction=False)
        science = spectra.phase_reference.copy()
        science[:, 0] = 0
        cls.product = compute_reduce(science, p2vm)

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
                before, after = (
                    getattr(product, field.name),
                    getattr(loaded, field.name),
                )
                if isinstance(before, np.ndarray):
                    np.testing.assert_array_equal(before, after)
                else:
                    self.assertEqual(before, after)
            np.testing.assert_array_equal(
                coherent.view(np.uint64), loaded.coherent_flux_adu.view(np.uint64)
            )
            self.assertTrue(loaded.flags[0].all())
            self.assertEqual(loaded.flags.dtype, bool)

    def test_corrupt_boolean_encoding_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "science.fits"
            write_product(path, self.product)
            with fits.open(path, mode="update") as hdus:
                hdus["FLAGS"].data[0, 0, 0] = 2
                hdus["FLAGS"].add_checksum()
            with self.assertRaisesRegex(ValueError, "boolean visibility flags"):
                read_product(path, ReducedObservables)

    def test_scientific_edit_with_valid_checksums_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "science.fits"
            write_product(path, self.product)
            with fits.open(path, mode="update") as hdus:
                hdus["VISIBILITY_AMPLITUDE"].data[0, 0, 0] = 42.0
                hdus["VISIBILITY_AMPLITUDE"].add_checksum()
            with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
                read_product(path)

    def test_baseline_table_exposes_zero_based_instrument_mapping(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "science.fits"
            write_product(path, self.product)
            with fits.open(path) as hdus:
                table = hdus["BASELINE_MAP"]
                self.assertTrue(table.header["INDEX0"])
                self.assertEqual(table.header["PLTABCD"], "A,C,B,D")
                for row in table.data:
                    baseline = int(row["BASELINE_INDEX"])
                    self.assertEqual(
                        list(row["OUTPUT_INDICES"]),
                        self.product.baseline_regions[baseline],
                    )
                    self.assertEqual(
                        tuple(row["TELESCOPE_INDICES"]),
                        self.product.baseline_telescopes[baseline],
                    )

    def test_mapping_dictionary_order_cannot_change_baseline_identity(self):
        product = replace(
            self.product,
            baseline_regions=dict(
                reversed(list(self.product.baseline_regions.items()))
            ),
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "science.fits"
            write_product(path, product)
            actual = read_product(path)
            self.assertEqual(actual.baseline_regions, product.baseline_regions)
            self.assertEqual(actual.baseline_telescopes, product.baseline_telescopes)

    def test_append_cannot_modify_a_raw_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "raw.fits"
            fits.PrimaryHDU(np.ones((3, 4))).writeto(path)
            original = path.read_bytes()
            with self.assertRaisesRegex(ValueError, "new empty DFS"):
                fill_dfs_product(path, self.product)
            self.assertEqual(path.read_bytes(), original)

    def test_static_reference_retains_original_values(self):
        lines = load_wavelength_reference().wavelength_um
        self.assertEqual(len(lines), 30)
        self.assertAlmostEqual(lines[0], 0.965780)
        self.assertAlmostEqual(lines[-1], 1.409370)


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


class RawFitsTests(unittest.TestCase):
    def test_fits_cube_iteration_and_statistics_use_the_full_cube(self) -> None:
        cube = np.arange(17 * 3 * 5, dtype=np.int16).reshape(17, 3, 5)

        with tempfile.TemporaryDirectory() as tmp:
            cube_path = Path(tmp) / "cube.fits"
            fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(cube)]).writeto(cube_path)

            chunks = list(iter_cube(cube_path))
            mean, std, _ = frame_statistics(chunks)
            expected = cube.astype(np.float32)

            self.assertEqual([len(chunk) for chunk in chunks], [16, 1])
            np.testing.assert_array_equal(np.concatenate(chunks), expected)
            np.testing.assert_allclose(mean, expected.mean(axis=0), rtol=1e-6)
            np.testing.assert_allclose(std, expected.std(axis=0), rtol=1e-6)

    def test_fits_input_must_be_a_3d_extension(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "image.fits"
            fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(np.ones((3, 5)))]).writeto(
                path
            )

            with self.assertRaisesRegex(
                ValueError, "three-dimensional raw detector cube"
            ):
                list(iter_cube(path))


class PublicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.wave = calibration_arrays(n_wave=15)[-1]

    def test_failure_during_write_leaves_no_final_file_and_can_retry(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "wave.fits"
            with patch(
                "planetes.fits.read_product",
                side_effect=ValueError("injected readback failure"),
            ):
                with self.assertRaisesRegex(ValueError, "injected"):
                    write_product(path, self.wave)
            self.assertFalse(path.exists())
            self.assertEqual(list(path.parent.iterdir()), [])
            write_product(path, self.wave)
            self.assertIsNotNone(read_product(path))

    def test_concurrent_publication_never_overwrites_the_winner(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "wave.fits"
            with self.assertRaises(FileExistsError):
                with product_destination(path) as candidate:
                    write_product(candidate, self.wave)
                    path.write_bytes(b"existing winner")
            self.assertEqual(path.read_bytes(), b"existing winner")

    def test_dangling_symlink_is_also_an_existing_target(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "wave.fits"
            path.symlink_to("missing")
            with self.assertRaises(FileExistsError):
                write_product(path, self.wave)
            self.assertTrue(path.is_symlink())
