"""Scientific invariants for the retained extraction, matrix and visibility definitions."""

import unittest
from dataclasses import replace

import numpy as np
from p2vm_fixture import calibration_arrays, calibration_spectra

from planetes_pipe.extraction import extract_sparse_spectra
from planetes_pipe.instrument import N_DATA
from planetes_pipe.p2vm import build_v2pm, fit_p2vm, fit_abcd_ellipse
from planetes_pipe.reduction import normalize_coherent_flux, reduce_spectra
from planetes_pipe.wavelength import match_peaks_grid


class ScienceTests(unittest.TestCase):
    def test_ellipse_radius_is_normalized_independently_of_fringe_contrast(self):
        phase_rad = np.linspace(0, 2 * np.pi, 64, endpoint=False)[:, None]
        phase_rad = phase_rad + np.array([0.0, 0.25, 0.5])
        abcd_phase_rad = np.array([0.0, np.pi, np.pi / 2, 3 * np.pi / 2])
        expected_phasor = np.exp(-1j * (phase_rad - phase_rad[0]))
        for contrast_adu in (50.0, 400.0):
            spectra = 1000 + contrast_adu * np.cos(
                phase_rad[None, :, :] + abcd_phase_rad[:, None, None]
            )
            ellipse_phase_rad, ellipse_radius = fit_abcd_ellipse(spectra)
            np.testing.assert_allclose(ellipse_radius, 1.0, rtol=0, atol=1e-12)
            # The retained A,C,B,D differences give phase decreasing with input OPD.
            np.testing.assert_allclose(
                np.exp(1j * (ellipse_phase_rad - ellipse_phase_rad[0])),
                expected_phasor,
                rtol=0,
                atol=1e-12,
            )

    def test_sparse_extractor_skips_invalid_sentinels(self) -> None:
        cube = np.ones((1, 2, 3))
        cube[0, -1, -1] = 1000.0
        ys = np.array([[0, -1]])
        xs = np.array([[0, -1]])

        batch = extract_sparse_spectra(cube, ys, xs)

        self.assertEqual(batch[0, 0, 0], 1.0)
        self.assertEqual(batch.sum(), 1.0)

    def test_grid_line_matching_is_one_to_one(self) -> None:
        theoretical = np.array([0.0, 0.15, 1.0, 2.0])
        observed = np.array([0.05, 1.0, 2.0])
        matches, _ = match_peaks_grid(
            theoretical,
            observed,
            tol=0.2,
            min_inliers=3,
            offset_limit=0.5,
            slope_bounds=(0.8, 1.2),
            curve_limit=0.1,
            steps_per_unit=2,
        )
        self.assertEqual(len(matches), len(np.unique(matches[:, 1])))

    def test_visibility_normalization_flags_invalid_flux_products(self) -> None:
        coherent_flux = np.tile(np.array([[[6 + 8j, 1 + 0j]]]), (1, 6, 1))
        flux = np.array([[[4.0, -1.0], [25.0, 2.0], [25.0, 2.0], [25.0, 2.0]]])

        visamp, flag = normalize_coherent_flux(coherent_flux, flux)

        self.assertEqual(visamp[0, 0, 0], 1.0)
        self.assertFalse(flag[0, 0, 0])
        self.assertTrue(np.isnan(visamp[0, 0, 1]))
        self.assertTrue(flag[0, 0, 1])

    def test_p2vm_inverts_known_full_rank_instrument_response(self):
        spectra = calibration_spectra(n_wave=15)
        p2vm = fit_p2vm(spectra)
        diagnostic = fit_p2vm(spectra, keep_diagnostics=True)
        self.assertIsNone(p2vm.opd_m)
        self.assertIsNotNone(diagnostic.opd_m)
        self.assertEqual(p2vm.matrix.tobytes(), diagnostic.matrix.tobytes())
        matrix = build_v2pm(
            p2vm.transmission, p2vm.coherence_amplitude, p2vm.coherence_phase_rad
        )
        for wave in range(len(p2vm.wavelength_m)):
            np.testing.assert_allclose(
                p2vm.matrix[:, :, wave] @ matrix[:, :, wave], np.eye(N_DATA), atol=1e-12
            )
        outputs = {**p2vm.baseline_outputs, 0: p2vm.baseline_outputs[1]}
        reference = calibration_arrays(n_wave=15)[2]
        with self.assertRaisesRegex(ValueError, "mapping"):
            reduce_spectra(reference, replace(p2vm, baseline_outputs=outputs))
        with self.assertRaisesRegex(ValueError, "dimensions"):
            reduce_spectra(reference[..., :1], p2vm)
