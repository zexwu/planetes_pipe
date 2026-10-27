"""Calibration relationships, grids, independent darks and live/file equivalence."""

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
from astropy.io import fits
from p2vm_fixture import calibration_arrays, calibration_spectra

from planetes.dark import calibrate_dark_file, load_dark
from planetes.extraction import prepare_extraction
from planetes.fits import read_product, write_product
from planetes.flat import compute_flat
from planetes.models import (
    DarkCalibration,
    DarkImage,
    fingerprint,
    validate_gd_range,
    wavelength_grid,
)
from planetes.p2vm import compute_p2vm
from planetes.reduction import FrameReducer, reduce_file


class CalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.values = calibration_arrays(n_wave=15)
        cls.flat, cls.wave = cls.values[-2:]
        cls.p2vm = compute_p2vm(calibration_spectra(n_wave=15), phase_correction=False)
        cls.dark = DarkImage(np.zeros_like(cls.flat.flat_map_adu, dtype=np.float32))

    def test_same_shape_different_flat_is_rejected(self):
        wrong = replace(self.flat, flat_map_adu=self.flat.flat_map_adu + 1)
        with self.assertRaisesRegex(ValueError, "dependency"):
            prepare_extraction(wrong, self.wave, self.p2vm.wavelength_um)

    def test_same_shape_different_wave_is_rejected(self):
        wrong = replace(
            self.wave, wavelength_map_um=self.wave.wavelength_map_um + 0.001
        )
        with self.assertRaisesRegex(ValueError, "dependency"):
            FrameReducer(self.flat, wrong, self.p2vm)

    def test_coverage_is_strict_and_target_grid_is_explicit(self):
        with self.assertRaisesRegex(ValueError, "Output 0.*outside"):
            prepare_extraction(self.flat, self.wave, np.linspace(1.1, 1.3, 15))
        grid = np.linspace(1.16, 1.29, 17)
        prepared = prepare_extraction(self.flat, self.wave, grid)
        np.testing.assert_array_equal(prepared.wavelength_um, grid)
        self.assertEqual(
            prepared.extract(np.ones((1, 24, 15), dtype=np.float32), self.dark).shape,
            (24, 1, 17),
        )

    def test_live_preparation_owns_geometry_grid_and_inverse_snapshots(self):
        flat = replace(self.flat, profile_x_pixels=self.flat.profile_x_pixels.copy())
        parent = replace(
            self.p2vm,
            p2vm=self.p2vm.p2vm.copy(),
            wavelength_um=self.p2vm.wavelength_um.copy(),
        )
        reducer = FrameReducer(flat, self.wave, parent)
        geometry = reducer.extractor.profile_x_pixels.copy()
        inverse = reducer.p2vm.p2vm.copy()
        grid = reducer.wavelength_um.copy()
        flat.profile_x_pixels[:] = -1
        parent.wavelength_um[:] = 2
        parent.p2vm[:] = 0
        np.testing.assert_array_equal(reducer.extractor.profile_x_pixels, geometry)
        np.testing.assert_array_equal(reducer.p2vm.p2vm, inverse)
        np.testing.assert_array_equal(reducer.wavelength_um, grid)
        for values in (
            reducer.extractor.profile_x_pixels,
            reducer.p2vm.p2vm,
            reducer.wavelength_um,
        ):
            self.assertFalse(values.flags.writeable)

    def test_live_and_file_match_with_an_independent_dark(self):
        signal = np.transpose(self.values[2], (1, 0, 2)).astype(np.float32)
        signal[0] = 0
        dark = DarkImage(np.full(signal.shape[1:], 7, dtype=np.float32))
        raw = signal + dark.mean_adu
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "science.fits"
            fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(raw)]).writeto(path)
            complete = reduce_file(path, dark, self.flat, self.wave, self.p2vm)
            reducer = FrameReducer(self.flat, self.wave, self.p2vm)
            for index in (0, 1, len(raw) - 1):
                one = reducer.reduce_frame(raw[index], dark=dark)
                for name in (
                    "photometric_flux_adu",
                    "coherent_flux_adu",
                    "visibility_amplitude",
                    "flags",
                ):
                    a, b = (
                        getattr(complete, name)[index : index + 1],
                        getattr(one, name),
                    )
                    self.assertEqual(a.tobytes(), b.tobytes(), name)
                self.assertFalse(hasattr(one, "group_delay_um"))
            self.assertTrue(complete.flags[0].all())

    def test_2d_dark_has_no_invented_noise_and_cannot_build_flat(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "dark.fits"
            fits.PrimaryHDU(np.full((24, 15), 7, dtype=np.float32)).writeto(path)
            dark = load_dark(path)
            self.assertIsInstance(dark, DarkImage)
            self.assertFalse(hasattr(dark, "temporal_rms_adu"))
            with self.assertRaisesRegex(ValueError, "complete dark"):
                compute_flat(dark, [np.ones((24, 15))])

    def test_raw_stack_and_gui_loader_use_identical_statistics(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "dark.fits"
            data = np.arange(17 * 3 * 5, dtype=np.float32).reshape(17, 3, 5)
            fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(data)]).writeto(path)
            a, b = calibrate_dark_file(path), load_dark(path)
            for name in ("mean_adu", "temporal_rms_adu", "bad_pixel_flags"):
                self.assertEqual(getattr(a, name).tobytes(), getattr(b, name).tobytes())
            self.assertEqual(a.n_frames, 17)

    def test_single_frame_cube_is_subtraction_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "dark.fits"
            fits.HDUList(
                [fits.PrimaryHDU(), fits.ImageHDU(np.ones((1, 3, 5), dtype=np.float32))]
            ).writeto(path)
            self.assertIsInstance(load_dark(path), DarkImage)
            with self.assertRaisesRegex(ValueError, "at least two"):
                calibrate_dark_file(path)

    def test_wrong_category_cannot_be_loaded_as_dark(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "not_dark.fits"
            write_product(path, self.wave)
            with self.assertRaisesRegex(ValueError, "Expected dark"):
                load_dark(path)

    def test_fingerprint_changes_with_data_but_not_filename_or_producer(self):
        info = replace(
            self.flat.provenance,
            producer="pyesorex:test",
            inputs={"raw": "another_path"},
        )
        self.assertEqual(
            fingerprint(self.flat), fingerprint(replace(self.flat, provenance=info))
        )
        self.assertNotEqual(
            fingerprint(self.flat),
            fingerprint(replace(self.flat, flat_map_adu=self.flat.flat_map_adu + 1)),
        )

    def test_parameter_domains_do_not_allow_degenerate_gd_grids(self):
        for n in (0, 1, 2, True, 3.5):
            with self.subTest(channels=n), self.assertRaises(ValueError):
                wavelength_grid(channels=n)
        for bounds in ((1, 0), (0, np.inf), (np.nan, 1)):
            with self.subTest(bounds=bounds), self.assertRaises(ValueError):
                validate_gd_range(bounds)

    def test_calibration_gd_parameter_reaches_all_searches(self):
        from planetes.reduction import compute_gdelay

        calls = []

        def measured(data, wavelength, **kwargs):
            calls.append(kwargs["search_range"])
            return compute_gdelay(data, wavelength, **kwargs)

        with patch("planetes.p2vm.compute_gdelay", side_effect=measured):
            compute_p2vm(
                calibration_spectra(n_wave=15),
                phase_correction=True,
                gd_search_min_um=-12.0,
                gd_search_max_um=12.0,
            )
        self.assertGreater(len(calls), 6)
        self.assertTrue(all(value == (-12.0, 12.0) for value in calls))

    def test_nondefault_width_pads_profiles_without_changing_pixels(self):
        from planetes.extraction import extract_spec_sparse

        for center in (4.5, 0.5):
            ny, nx = 14, 128
            image = np.zeros((ny, nx), dtype=np.float32)
            image[1 if center < 1 else 4] = 100
            image[8] = 100
            bad = np.zeros((ny, nx), dtype=np.int16)
            bad[1 if center < 1 else 4, 10] = 1
            dark = DarkCalibration(np.zeros_like(image), np.ones_like(image), bad, 2)
            traces = [
                (np.full(nx, center), np.zeros(nx)),
                (np.full(nx, 8.0), np.zeros(nx)),
            ]
            with patch("planetes.flat.trace_spectrum", side_effect=traces):
                flat = compute_flat(dark, [image], n_outputs=2, profile_width=3.5)
            Y, X = np.indices((ny, nx))
            actual = extract_spec_sparse(
                np.ones((1, ny, nx), dtype=np.float32),
                flat.profile_y_pixels,
                flat.profile_x_pixels,
            )
            for output in range(2):
                mask = (abs(Y - flat.trace_y_pixels[output][X]) < 3.5 / 2) & (bad == 0)
                np.testing.assert_array_equal(actual[output, 0], mask.sum(axis=0))
            self.assertTrue(
                ((flat.profile_x_pixels == -1) == (flat.profile_y_pixels == -1)).all()
            )
            self.assertTrue((flat.profile_x_pixels == -1).any())
            if center > 1:
                self.assertTrue((flat.profile_x_pixels[-1] == -1).any())

    def test_newton_refinement_can_leave_coarse_initialization_bounds(self):
        from planetes.reduction import compute_gdelay

        wave = np.linspace(1.15, 1.3, 151)
        signal = np.exp(2j * np.pi * 2.0 / wave)[None]
        actual = compute_gdelay(signal, wave, search_range=(0.0, 1.0))
        np.testing.assert_allclose(actual, [2.0], atol=1e-8)
        self.assertGreater(actual[0], 1.0)

    def test_phase_dark_is_independent_from_main_dark(self):
        from p2vm_fixture import write_recipe_inputs

        from planetes.extraction import extract_calibration_files
        from planetes.models import wavelength_grid

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = write_recipe_inputs(root / "inputs")
            reference = inputs["PHASE_REFERENCE"]
            with fits.open(reference, mode="update") as hdus:
                hdus[1].data += 7
                hdus[1].add_checksum()
            flat, wave = (
                read_product(inputs["PLANETES_FLAT"]),
                read_product(inputs["PLANETES_WAVE"]),
            )
            main = read_product(inputs["DARK_CALIB"])
            phase = DarkImage(np.full(main.mean_adu.shape, 7, dtype=np.float32))
            spectra = extract_calibration_files(
                flat,
                wave,
                main,
                telescope_flats={
                    n: inputs[f"TEL_FLAT_{n}"] for n in ("4", "3", "2", "1")
                },
                baseline_cubes={
                    n: inputs[f"P2VM_RAW_{n}"]
                    for n in ("43", "42", "41", "32", "31", "21")
                },
                wavelength_um=wavelength_grid(),
                phase_reference=reference,
                phase_dark=phase,
            )
            raw = np.array(fits.getdata(reference), dtype=np.float32)
            corrected = raw - 7
            prepared = prepare_extraction(flat, wave, wavelength_grid())
            expected = prepared.extract(
                corrected, DarkImage(np.zeros_like(phase.mean_adu))
            )
            np.testing.assert_array_equal(spectra.phase_reference, expected)
            self.assertEqual(
                spectra.provenance.dependencies["phase_dark"], fingerprint(phase)
            )
            self.assertNotEqual(fingerprint(main), fingerprint(phase))
