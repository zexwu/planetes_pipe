"""Calibration relationships, grids, independent darks and live/file equivalence."""

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
from astropy.io import fits
from p2vm_fixture import (
    calibration_arrays,
    calibration_spectra,
    metadata,
    setup,
    write_recipe_inputs,
)

from planetes.dark import calibrate_dark_file, compute_badpix, load_dark
from planetes.extraction import prepare_extraction
from planetes.flat import compute_flat
from planetes.io.fits import read_product, write_product
from planetes.io.raw import write_raw
from planetes.p2vm import (
    calibrate_p2vm_file,
    compute_p2vm,
    extract_calibration_files,
    wavelength_grid,
)
from planetes.products import (
    DarkCalibration,
    DarkImage,
    fingerprint,
    validate_gd_range,
)
from planetes.reduction import FrameReducer, reduce_file


class CalibrationTests(unittest.TestCase):
    def test_p2vm_file_interface_accepts_explicit_bench_identities(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = write_recipe_inputs(root / "versioned")
            flat, wave = read_product(paths["FLAT"]), read_product(paths["WAVE"])
            dark = read_product(paths["DARK"])
            options = dict(wavelength_channels=15, phase_correction=False)
            expected = calibrate_p2vm_file(
                flat,
                wave,
                dark,
                telescope_flats=paths["FLAT_RAW"],
                baseline_cubes=paths["P2VM_RAW"],
                **options,
            )

            def bench_files(role, identities):
                result = {}
                for identity, path in zip(identities, paths[role], strict=True):
                    target = root / f"{role}_{identity}.fits"
                    with fits.open(path) as hdus:
                        fits.PrimaryHDU(hdus[1].data).writeto(target)
                    result[identity] = target
                return result

            actual = calibrate_p2vm_file(
                flat,
                wave,
                dark,
                telescope_flats=bench_files("FLAT_RAW", ("4", "3", "2", "1")),
                baseline_cubes=bench_files(
                    "P2VM_RAW", ("43", "42", "41", "32", "31", "21")
                ),
                **options,
            )
            np.testing.assert_array_equal(actual.p2vm, expected.p2vm)
            self.assertEqual(actual.baseline_regions, expected.baseline_regions)
            self.assertEqual(fingerprint(actual), fingerprint(expected))
            self.assertIn("DARK.exposure_s", actual.provenance.unverified)
            self.assertIn("RAW.BUNIT", actual.provenance.unverified)

    @classmethod
    def setUpClass(cls):
        cls.values = calibration_arrays(n_wave=15)
        cls.flat, cls.wave = cls.values[-2:]
        cls.p2vm = compute_p2vm(calibration_spectra(n_wave=15), phase_correction=False)
        cls.dark = DarkImage(np.zeros_like(cls.flat.flat_map_adu, dtype=np.float32))

    def test_same_shape_different_flat_is_rejected(self):
        wrong = replace(self.flat, flat_map_adu=self.flat.flat_map_adu + 1)
        with self.assertRaisesRegex(ValueError, "dependency"):
            prepare_extraction(wrong, self.wave, self.p2vm.wavelength_m)

    def test_same_shape_different_wave_is_rejected(self):
        wrong = replace(self.wave, wavelength_map_m=self.wave.wavelength_map_m + 0.001)
        with self.assertRaisesRegex(ValueError, "dependency"):
            FrameReducer(self.flat, wrong, self.p2vm)

    def test_coverage_is_strict_and_target_grid_is_explicit(self):
        with self.assertRaisesRegex(ValueError, "Output 0.*outside"):
            prepare_extraction(self.flat, self.wave, np.linspace(1.1e-6, 1.3e-6, 15))
        grid = np.linspace(1.16e-6, 1.29e-6, 17)
        prepared = prepare_extraction(self.flat, self.wave, grid)
        np.testing.assert_array_equal(prepared.wavelength_m, grid)
        self.assertEqual(
            prepared.extract(np.ones((1, 24, 15), dtype=np.float32), self.dark).shape,
            (24, 1, 17),
        )

    def test_live_preparation_owns_geometry_grid_and_inverse_snapshots(self):
        flat = replace(self.flat, profile_x_pixels=self.flat.profile_x_pixels.copy())
        parent = replace(
            self.p2vm,
            p2vm=self.p2vm.p2vm.copy(),
            wavelength_m=self.p2vm.wavelength_m.copy(),
        )
        reducer = FrameReducer(flat, self.wave, parent)
        geometry = reducer.extractor.profile_x_pixels.copy()
        inverse = reducer.p2vm.p2vm.copy()
        grid = reducer.wavelength_m.copy()
        flat.profile_x_pixels[:] = -1
        parent.wavelength_m[:] = 2
        parent.p2vm[:] = 0
        np.testing.assert_array_equal(reducer.extractor.profile_x_pixels, geometry)
        np.testing.assert_array_equal(reducer.p2vm.p2vm, inverse)
        np.testing.assert_array_equal(reducer.wavelength_m, grid)
        for values in (
            reducer.extractor.profile_x_pixels,
            reducer.p2vm.p2vm,
            reducer.wavelength_m,
        ):
            self.assertFalse(values.flags.writeable)

    def test_live_and_file_match_with_an_independent_dark(self):
        signal = np.transpose(self.values[2], (1, 0, 2)).astype(np.float32)
        signal[0] = 0
        dark = DarkCalibration(
            np.full(signal.shape[1:], 7, dtype=np.float32),
            np.zeros(signal.shape[1:], dtype=np.float32),
            32,
            setup=setup(15),
        )
        raw = signal + dark.mean_adu
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "science.fits"
            write_raw(path, raw, metadata("SINGLE_SCI_RAW", "1234", 15))
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
                self.assertFalse(hasattr(one, "group_delay_m"))
            self.assertTrue(complete.flags[0].all())

    def test_2d_dark_has_no_invented_noise_and_cannot_build_flat(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "dark.fits"
            fits.PrimaryHDU(np.full((24, 15), 7, dtype=np.float32)).writeto(path)
            dark = load_dark(path)
            self.assertIsInstance(dark, DarkImage)
            self.assertFalse(hasattr(dark, "temporal_rms_adu"))
            with self.assertRaisesRegex(ValueError, "complete dark"):
                compute_flat(dark, None, [np.ones((24, 15))])

    def test_raw_stack_and_gui_loader_use_identical_statistics(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "dark.fits"
            data = np.arange(17 * 3 * 5, dtype=np.float32).reshape(17, 3, 5)
            write_raw(path, data, metadata("DARK_RAW", n_wave=5, height=3))
            (a, _bad), b = calibrate_dark_file(path), load_dark(path)
            for name in ("mean_adu", "temporal_rms_adu"):
                self.assertEqual(getattr(a, name).tobytes(), getattr(b, name).tobytes())
            self.assertEqual(a.n_frames, 17)

    def test_single_frame_cube_is_subtraction_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "dark.fits"
            write_raw(
                path,
                np.ones((1, 3, 5), dtype=np.float32),
                metadata("DARK_RAW", n_wave=5, height=3),
            )
            self.assertIsInstance(load_dark(path), DarkImage)
            with self.assertRaisesRegex(ValueError, "at least two"):
                calibrate_dark_file(path)

    def test_wrong_category_cannot_be_loaded_as_dark(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "not_dark.fits"
            write_product(path, self.wave)
            with self.assertRaisesRegex(ValueError, "Unexpected product category"):
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
                gd_search_min_m=-12e-6,
                gd_search_max_m=12e-6,
            )
        self.assertGreater(len(calls), 6)
        self.assertTrue(all(value == (-12e-6, 12e-6) for value in calls))

    def test_nondefault_width_pads_profiles_without_changing_pixels(self):
        from planetes.extraction import extract_spec_sparse

        for center in (4.5, 0.5):
            ny, nx = 14, 128
            image = np.zeros((ny, nx), dtype=np.float32)
            image[1 if center < 1 else 4] = 100
            image[8] = 100
            bad = np.zeros((ny, nx), dtype=np.int16)
            bad[1 if center < 1 else 4, 10] = 1
            dark = DarkCalibration(np.zeros_like(image), np.ones_like(image), 2)
            bpm = replace(compute_badpix(dark), bad_pixel_flags=bad)
            traces = [
                (np.full(nx, center), np.zeros(nx), np.ones(nx)),
                (np.full(nx, 8.0), np.zeros(nx), np.ones(nx)),
            ]
            with patch("planetes.flat.trace_spectrum", side_effect=traces):
                flat = compute_flat(dark, bpm, [image], n_outputs=2, profile_width=3.5)
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

        wave = np.linspace(1.15e-6, 1.3e-6, 151)
        signal = np.exp(2j * np.pi * 2e-6 / wave)[None]
        actual = compute_gdelay(signal, wave, search_range=(0.0, 1e-6))
        np.testing.assert_allclose(actual, [2e-6], atol=1e-14)
        self.assertGreater(actual[0], 1e-6)

    def test_phase_dark_is_independent_from_main_dark(self):
        from p2vm_fixture import write_recipe_inputs

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = write_recipe_inputs(root / "inputs")
            reference = inputs["PHASE_REFERENCE"]
            with fits.open(reference, mode="update") as hdus:
                hdus[1].data += 7
                hdus[1].add_checksum()
            flat, wave = (
                read_product(inputs["FLAT"]),
                read_product(inputs["WAVE"]),
            )
            main = read_product(inputs["DARK"])
            phase = replace(
                main, mean_adu=np.full(main.mean_adu.shape, 7, dtype=np.float32)
            )
            spectra = extract_calibration_files(
                flat,
                wave,
                main,
                telescope_flats=inputs["FLAT_RAW"],
                baseline_cubes=inputs["P2VM_RAW"],
                wavelength_m=wavelength_grid(),
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


class SetupAndQualityTests(unittest.TestCase):
    def test_known_exposure_roi_conflicts_and_explicit_legacy_unknown(self):
        from planetes.instrument import DetectorSetup, check_setup

        known = setup(15)
        for field, value in [
            ("exposure_s", 0.02),
            ("origin_x", 1),
            ("detector", "other"),
        ]:
            with (
                self.subTest(field=field),
                self.assertRaisesRegex(ValueError, "Incompatible"),
            ):
                check_setup(
                    known, replace(known, **{field: value}), exposure=True, role="DARK"
                )
        with self.assertRaisesRegex(ValueError, "Missing"):
            check_setup(known, DetectorSetup(), exposure=True, role="DARK")
        unknown = check_setup(
            known, DetectorSetup(legacy=True), exposure=True, role="DARK"
        )
        self.assertIn("DARK.exposure_s", unknown)

    def test_bad_requires_the_dark_that_actually_created_it(self):
        dark = DarkCalibration(np.zeros((24, 15)), np.ones((24, 15)), 2)
        bad = compute_badpix(dark)
        with self.assertRaisesRegex(ValueError, "dependency"):
            compute_flat(replace(dark, mean_adu=np.ones((24, 15))), bad, [])

    def test_qc_uses_population_dark_rms_and_sample_wave_residual_stdev(self):
        from planetes.products import WavelengthCalibration
        from planetes.qc import measure_quality

        dark = DarkCalibration(
            np.array([[1.0, 2.0], [3.0, 40.0]]), np.array([[2.0, 4.0], [6.0, 80.0]]), 4
        )
        quality, _ = measure_quality(dark)
        self.assertEqual(quality["dark_median_adu"][0], 2.5)
        self.assertEqual(quality["dark_rms_median_adu"][0], 5.0)
        wave = replace(
            calibration_arrays(n_wave=15)[-1],
            wavelength_map_m=np.vstack(
                [np.linspace(1e-6 + 1e-9 * i, 2e-6 + 1e-9 * i, 15) for i in range(24)]
            ),
            fit_residual_m=np.array([1e-6, 2e-6, 4e-6]),
            fit_x_pixels=np.arange(3.0),
            fit_y_pixels=np.arange(3.0),
        )
        self.assertIsInstance(wave, WavelengthCalibration)
        quality, _ = measure_quality(wave)
        self.assertEqual(quality["wavelength_min_m"][0], 1.023 / 1e6)
        self.assertEqual(quality["wavelength_max_m"][0], 2.0 / 1e6)
        self.assertAlmostEqual(
            quality["wavelength_residual_std_m"][0], np.sqrt(7 / 3) * 1e-6
        )
        _, missing = measure_quality(replace(wave, fit_residual_m=np.empty(0)))
        self.assertIn("wavelength_residual_std", missing)

    def test_p2vm_coherence_qc_has_independent_normalization_and_quantiles(self):
        from planetes.qc import measure_quality

        product = compute_p2vm(calibration_spectra(n_wave=15), phase_correction=False)
        transmission = np.full_like(product.transmission, 2.0)
        amplitude = np.broadcast_to(
            np.arange(1, 16.0), product.coherence_amplitude.shape
        ).copy()
        quality, _ = measure_quality(
            replace(product, transmission=transmission, coherence_amplitude=amplitude)
        )
        self.assertEqual(quality["p2vm_coherence_mean_b43"][0], 2.0)
        self.assertAlmostEqual(quality["p2vm_coherence_min_b43"][0], 0.25)
        self.assertAlmostEqual(quality["p2vm_coherence_max_b43"][0], 3.75)
        self.assertEqual(quality["p2vm_transmission_mean"][0], 1.0)
