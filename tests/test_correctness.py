from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from planetes.instrument import (
    IMAG_SLICE,
    N_BASELINES,
    N_OUTPUTS,
    N_TELESCOPES,
    REAL_SLICE,
)
from planetes.recipes.flat import build_flat_product
from planetes.recipes.p2vm import _build_v2pm_matrix
from planetes.recipes.preproc import extract_spec_sparse
from planetes.recipes.reduction import normalize_visibility
from planetes.recipes.wavelength import load_wavelength_lines, match_lines_grid


class CorrectnessTests(unittest.TestCase):
    def test_v2pm_terms_expand_to_cartesian_matrix(self) -> None:
        transmission = np.arange(
            N_OUTPUTS * N_TELESCOPES * 2, dtype=float
        ).reshape(N_OUTPUTS, N_TELESCOPES, 2)
        amplitude = np.full((N_OUTPUTS, N_BASELINES, 2), 2.0)
        phase = np.full((N_OUTPUTS, N_BASELINES, 2), np.pi / 3)

        matrix = _build_v2pm_matrix(transmission, amplitude, phase)

        np.testing.assert_array_equal(
            matrix[:, :N_TELESCOPES], transmission / N_OUTPUTS
        )
        np.testing.assert_array_equal(
            matrix[:, REAL_SLICE], amplitude * np.cos(phase) / N_OUTPUTS
        )
        np.testing.assert_array_equal(
            matrix[:, IMAG_SLICE], amplitude * np.sin(phase) / N_OUTPUTS
        )

    def test_sparse_profile_excludes_bad_pixels(self) -> None:
        dark_map = np.zeros((8, 128), dtype=np.float32)
        dark_std = np.zeros_like(dark_map)
        flat_map = np.zeros_like(dark_map)
        flat_map[4, :] = 100.0
        bad_map = np.zeros((8, 128), dtype=np.int16)
        bad_map[4, 10] = 1
        trace = np.full(128, 4.0)
        trace_error = np.zeros(128)

        with (
            patch(
                "planetes.recipes.flat.compute_bad_map_from_stats",
                return_value=bad_map,
            ),
            patch(
                "planetes.recipes.flat.trace_spectrum",
                return_value=(trace, trace_error),
            ),
        ):
            product = build_flat_product(
                dark_map,
                dark_std,
                [flat_map],
                n_outputs=1,
                profile_width=3,
            )

        selected = set(
            zip(
                product.profile_y_pixels[0],
                product.profile_x_pixels[0],
                strict=True,
            )
        )
        self.assertNotIn((4, 10), selected)
        self.assertIn((-1, -1), selected)

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

    def test_wavelength_reference_is_sorted_and_versioned(self) -> None:
        lines = load_wavelength_lines()
        self.assertEqual(len(lines), 30)
        self.assertTrue(np.all(np.diff(lines) > 0))
        self.assertAlmostEqual(lines[0], 0.965780)
        self.assertAlmostEqual(lines[-1], 1.409370)

    def test_visibility_normalization_flags_invalid_flux_products(self) -> None:
        coherent_flux = np.array([[[6 + 8j, 1 + 0j]]])
        flux = np.array([[[4.0, -1.0], [25.0, 2.0]]])

        visamp, flag = normalize_visibility(coherent_flux, flux, {0: (0, 1)})

        self.assertEqual(visamp[0, 0, 0], 1.0)
        self.assertFalse(flag[0, 0, 0])
        self.assertTrue(np.isnan(visamp[0, 0, 1]))
        self.assertTrue(flag[0, 0, 1])


if __name__ == "__main__":
    unittest.main()
