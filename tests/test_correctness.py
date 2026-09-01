from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from planetes.config import FlatParameters
from planetes.recipes.flat import build_flat_product
from planetes.recipes.preproc import extract_spec_sparse, order_abcd_regions
from planetes.recipes.reduction import normalize_visibility
from planetes.recipes.wavelength import load_wavelength_lines, match_lines_grid
from rtp2vm import extract_spec_sparse_single


class CorrectnessTests(unittest.TestCase):
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
            product, _ = build_flat_product(
                dark_map,
                dark_std,
                [flat_map],
                n_outputs=1,
                parameters=FlatParameters(profile_width=3),
            )

        selected = set(
            zip(product.profile_ys[0], product.profile_xs[0], strict=True)
        )
        self.assertNotIn((4, 10), selected)
        self.assertIn((-1, -1), selected)
        self.assertEqual(product.profile_map[0, 4, 10], 0)

    def test_sparse_extractors_skip_invalid_sentinels(self) -> None:
        cube = np.ones((1, 2, 3), dtype=np.float32)
        cube[0, -1, -1] = 1000.0
        ys = np.array([[0, -1]])
        xs = np.array([[0, -1]])

        batch = extract_spec_sparse(cube, ys, xs)
        realtime = extract_spec_sparse_single(cube[0], ys, xs)

        self.assertEqual(batch[0, 0, 0], 1.0)
        self.assertEqual(batch.sum(), 1.0)
        self.assertEqual(realtime[0, 0], 1.0)
        self.assertEqual(realtime.sum(), 1.0)

    def test_abcd_regions_follow_configured_detector_order(self) -> None:
        ordered = order_abcd_regions(
            [15, 12, 14, 13], detector_order=["A", "B", "C", "D"]
        )
        self.assertEqual(ordered, [12, 14, 13, 15])

    def test_abcd_order_rejects_invalid_labels(self) -> None:
        with self.assertRaisesRegex(ValueError, "abcd_order"):
            order_abcd_regions([0, 1, 2, 3], ["A", "B", "C", "C"])

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

        visamp, flag = normalize_visibility(
            coherent_flux, flux, {0: (0, 1)}
        )

        self.assertEqual(visamp[0, 0, 0], 1.0)
        self.assertFalse(flag[0, 0, 0])
        self.assertTrue(np.isnan(visamp[0, 0, 1]))
        self.assertTrue(flag[0, 0, 1])


if __name__ == "__main__":
    unittest.main()
