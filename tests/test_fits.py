"""Scientific FITS round trips, raw reading and non-overwriting publication."""

import tempfile
import unittest
from dataclasses import fields, replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
from astropy.io import fits
from p2vm_fixture import calibration_arrays, calibration_spectra, metadata

from planetes.dark import frame_statistics
from planetes.fits import (
    destinations,
    fill_dfs_product,
    product_destination,
    read_product,
    write_product,
    write_raw,
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
                if field.name == "provenance":
                    continue
                before, after = (
                    getattr(product, field.name),
                    getattr(loaded, field.name),
                )
                if isinstance(before, np.ndarray):
                    expected = (
                        before / 1e6 * 1e6
                        if field.metadata and field.metadata["unit"] == "um"
                        else before
                    )
                    np.testing.assert_array_equal(expected, after)
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
                hdus["PLANETES_VIS"].data["FLAG"][0, 0] = False
                hdus["PLANETES_VIS"].add_checksum()
            with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
                read_product(path, ReducedObservables)

    def test_scientific_edit_with_valid_checksums_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "science.fits"
            write_product(path, self.product)
            with fits.open(path, mode="update") as hdus:
                hdus["PLANETES_VIS"].data["VISAMP"][0, 0] = 42.0
                hdus["PLANETES_VIS"].add_checksum()
            with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
                read_product(path)

    def test_science_rows_use_physical_station_ids_and_frame_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "science.fits"
            write_product(path, self.product)
            with fits.open(path) as hdus:
                np.testing.assert_array_equal(
                    hdus["PLANETES_VIS"].data["STA_INDEX"][:6],
                    [[4, 3], [4, 2], [4, 1], [3, 2], [3, 1], [2, 1]],
                )
                np.testing.assert_array_equal(
                    hdus["PLANETES_FLUX"].data["STA_INDEX"][:4], [4, 3, 2, 1]
                )
                self.assertTrue((hdus["PLANETES_VIS"].data["FRAME"][:6] == 1).all())

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
                        expected = (
                            before / 1e6 * 1e6
                            if field.metadata and field.metadata["unit"] == "um"
                            else before
                        )
                        np.testing.assert_array_equal(expected, after)
                        self.assertTrue(after.dtype.isnative)
                    else:
                        self.assertEqual(before, after)

    def test_fits_reader_rejects_changed_wavelength_units(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "wave.fits"
            write_product(path, self.wave)
            with fits.open(path, mode="update") as hdus:
                hdus["WAVE_DATA_SC"].header["TUNIT1"] = "nm"
                hdus["WAVE_DATA_SC"].add_checksum()
            with self.assertRaisesRegex(ValueError, "units"):
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
            write_raw(cube_path, cube, metadata("DARK_RAW", n_wave=5, height=3))

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
            fits.HDUList(
                [
                    fits.PrimaryHDU(),
                    fits.ImageHDU(np.ones((3, 5)), name="IMAGING_DATA_SC"),
                ]
            ).writeto(path)

            with self.assertRaisesRegex(ValueError, "new PLANETES raw FITS"):
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

    def test_second_publication_failure_rolls_back_only_our_first_file(self):
        import os

        link = os.link
        with tempfile.TemporaryDirectory() as temporary:
            paths = [Path(temporary) / "dark.fits", Path(temporary) / "bad.fits"]

            def fail_second(source, target):
                if Path(target) == paths[1]:
                    raise OSError("injected second publication failure")
                return link(source, target)

            with patch("planetes.fits.os.link", side_effect=fail_second):
                with self.assertRaisesRegex(OSError, "second publication"):
                    with destinations(paths) as staged:
                        # Stage directly to avoid counting nested single-product publication.
                        from planetes.fits import encode_product

                        for path in staged:
                            encode_product(self.wave).writeto(path, checksum=True)
            self.assertFalse(any(path.exists() for path in paths))


class CanonicalAndRawTests(unittest.TestCase):
    def test_rewrites_retain_valid_metres_and_inplace_changes_invalidate_hash(self):
        from planetes.models import fingerprint, retain_representation

        wave = calibration_arrays(n_wave=15)[-1]
        metres = np.nextafter(wave.wavelength_map_um / 1e6, np.inf)
        wave = replace(wave, wavelength_map_um=metres * 1e6)
        wave = retain_representation(wave, {"wavelength_map_um": metres})
        original = fingerprint(wave)
        with tempfile.TemporaryDirectory() as temporary:
            for i in range(3):
                path = Path(temporary) / f"wave{i}.fits"
                write_product(path, wave)
                wave = read_product(path)
                self.assertEqual(fingerprint(wave), original)
                with fits.open(path) as hdus:
                    np.testing.assert_array_equal(
                        hdus["WAVE_DATA_SC"].data["DATA1"][0], metres[0]
                    )
            wave.wavelength_map_um[0, 0] += 1e-6
            self.assertNotEqual(fingerprint(wave), original)
            self.assertNotEqual(
                fingerprint(
                    replace(
                        wave, provenance=replace(wave.provenance, parameters={"x": 1})
                    )
                ),
                original,
            )

    def test_raw_pixels_metadata_and_duplicate_shutters(self):
        from planetes.fits import read_raw_metadata, identify_raw

        data = np.array([[[40000.0, -2.0, 3.5]], [[1.0, 2.0, 3.0]]], dtype=np.float32)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "anything.fits"
            raw = metadata("FLAT_RAW", "4", 3, 1)
            write_raw(path, data, raw)
            self.assertEqual(read_raw_metadata(path), raw)
            with fits.open(path) as hdus:
                self.assertNotIn("DATE-OBS", hdus[0].header)
                self.assertEqual(hdus[0].header["ESO DET WIN1 DIT1"], 0.01)
                self.assertEqual(hdus[0].header["EXPTIME"], 0.02)
                np.testing.assert_array_equal(hdus["IMAGING_DATA_SC"].data, data)
                self.assertEqual(hdus["IMAGING_DATA_SC"].data.dtype.kind, "f")
            with self.assertRaisesRegex(ValueError, "Duplicate shutter"):
                identify_raw([path, path], "FLAT_RAW")
            with fits.open(path, mode="update") as hdus:
                hdus[0].header["ESO DET WIN1 STRX"] = 2
                hdus[0].add_checksum()
            with self.assertRaisesRegex(ValueError, "disagree"):
                read_raw_metadata(path)

    def test_operator_exposure_is_requested_and_legacy_missing_stays_unknown(self):
        from planetes.fits import read_raw_metadata

        with tempfile.TemporaryDirectory() as temporary:
            data = np.zeros((2, 1, 3), dtype=np.uint16)
            raw = metadata("DARK_RAW", n_wave=3, height=1)
            raw = replace(raw, setup=replace(raw.setup, exposure_source="operator"))
            path = Path(temporary) / "operator.fits"
            write_raw(path, data, raw)
            with fits.open(path) as hdus:
                self.assertIn("ESO DET WIN1 UIT1", hdus[0].header)
                self.assertNotIn("EXPTIME", hdus[0].header)
            raw = replace(
                raw,
                configuration_source="legacy-role-map",
                setup=replace(
                    raw.setup,
                    legacy=True,
                    exposure_s=None,
                    exposure_source="unknown",
                    binning_x=None,
                ),
            )
            path = Path(temporary) / "legacy.fits"
            write_raw(path, data, raw)
            self.assertIsNone(read_raw_metadata(path).setup.exposure_s)

    def test_invalid_fits_logical_bytes_are_rejected(self):
        spectra = calibration_spectra(n_wave=15)
        product = compute_reduce(
            spectra.phase_reference, compute_p2vm(spectra, phase_correction=False)
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "science.fits"
            write_product(path, product)
            with fits.open(path, mode="update") as hdus:
                hdus["PLANETES_VIS"].data._get_raw_data()["FLAG"][0, 0] = ord("X")
                hdus["PLANETES_VIS"].add_checksum()
            with self.assertRaisesRegex(ValueError, "logical flag"):
                read_product(path)

    def test_rollback_preserves_a_concurrent_replacement(self):
        import os
        from planetes.fits import encode_product

        wave = calibration_arrays(n_wave=15)[-1]
        link = os.link
        with tempfile.TemporaryDirectory() as temporary:
            paths = [Path(temporary) / "one.fits", Path(temporary) / "two.fits"]

            def replace_first_then_fail(source, target):
                if Path(target) == paths[1]:
                    paths[0].unlink()
                    paths[0].write_bytes(b"concurrent winner")
                    raise OSError("injected second failure")
                return link(source, target)

            with patch("planetes.fits.os.link", side_effect=replace_first_then_fail):
                with self.assertRaisesRegex(OSError, "injected"):
                    with destinations(paths) as staged:
                        for path in staged:
                            encode_product(wave).writeto(path, checksum=True)
            self.assertEqual(paths[0].read_bytes(), b"concurrent winner")
            self.assertFalse(paths[1].exists())
