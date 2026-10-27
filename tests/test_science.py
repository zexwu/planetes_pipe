"""Scientific invariants for the retained extraction, matrix and visibility definitions."""

import unittest

import numpy as np
from p2vm_fixture import calibration_spectra

from planetes.extraction import extract_spec_sparse
from planetes.instrument import (
    IMAG_SLICE,
    N_BASELINES,
    N_DATA,
    N_OUTPUTS,
    N_TELESCOPES,
    REAL_SLICE,
)
from planetes.p2vm import build_v2pm_matrix, compute_p2vm
from planetes.reduction import normalize_visibility
from planetes.wavelength import match_lines_grid


class ScienceTests(unittest.TestCase):
    def test_v2pm_terms_expand_to_cartesian_matrix(self) -> None:
        transmission = np.arange(N_OUTPUTS * N_TELESCOPES * 2, dtype=float).reshape(
            N_OUTPUTS, N_TELESCOPES, 2
        )
        amplitude = np.full((N_OUTPUTS, N_BASELINES, 2), 2.0)
        phase = np.full((N_OUTPUTS, N_BASELINES, 2), np.pi / 3)

        matrix = build_v2pm_matrix(transmission, amplitude, phase)

        np.testing.assert_array_equal(
            matrix[:, :N_TELESCOPES], transmission / N_OUTPUTS
        )
        np.testing.assert_array_equal(
            matrix[:, REAL_SLICE], amplitude * np.cos(phase) / N_OUTPUTS
        )
        np.testing.assert_array_equal(
            matrix[:, IMAG_SLICE], amplitude * np.sin(phase) / N_OUTPUTS
        )

    def test_sparse_extractor_skips_invalid_sentinels(self) -> None:
        cube = np.ones((1, 2, 3), dtype=np.float32)
        cube[0, -1, -1] = 1000.0
        ys = np.array([[0, -1]])
        xs = np.array([[0, -1]])

        batch = extract_spec_sparse(cube, ys, xs)

        self.assertEqual(batch[0, 0, 0], 1.0)
        self.assertEqual(batch.sum(), 1.0)

    def test_grid_line_matching_is_one_to_one(self) -> None:
        theoretical = np.array([0.0, 0.15, 1.0, 2.0])
        observed = np.array([0.05, 1.0, 2.0])
        matches, _ = match_lines_grid(
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
        coherent_flux = np.array([[[6 + 8j, 1 + 0j]]])
        flux = np.array([[[4.0, -1.0], [25.0, 2.0]]])

        visamp, flag = normalize_visibility(coherent_flux, flux, {0: (0, 1)})

        self.assertEqual(visamp[0, 0, 0], 1.0)
        self.assertFalse(flag[0, 0, 0])
        self.assertTrue(np.isnan(visamp[0, 0, 1]))
        self.assertTrue(flag[0, 0, 1])

    def test_p2vm_inverts_known_full_rank_instrument_response(self):
        p2vm = compute_p2vm(calibration_spectra(n_wave=15), phase_correction=False)
        matrix = build_v2pm_matrix(
            p2vm.transmission,
            p2vm.coherence_amplitude,
            p2vm.coherence_phase_rad,
        )
        for wave in range(len(p2vm.wavelength_um)):
            np.testing.assert_allclose(
                p2vm.p2vm[:, :, wave] @ matrix[:, :, wave],
                np.eye(N_DATA),
                atol=1e-12,
            )
