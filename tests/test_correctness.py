from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from recipes.flat import run_flat
from recipes.preproc import extract_spec_sparse, order_abcd_regions, resolve_object_entry
from recipes.wave import match_lines_grid
from rtp2vm import extract_spec_sparse_single


class _FlatContext:
    n_reg = 1
    conf = {
        "calib": {"dark": "dark.fits", "flat": ["flat.fits"]},
        "plot_to_pdf": False,
    }

    def __init__(self) -> None:
        self.active_parameters: dict[str, dict] = {}
        self.saved: dict[str, np.ndarray] = {}
        self.dark_cube = np.zeros((4, 8, 128), dtype=np.float32)
        self.flat_cube = np.zeros_like(self.dark_cube)
        self.flat_cube[:, 4, :] = 100.0

    def load_fits(self, filename: str) -> np.ndarray:
        return self.dark_cube if filename == "dark.fits" else self.flat_cube

    def mean_fits(self, filename: str) -> np.ndarray:
        return self.load_fits(filename).mean(axis=0)

    def mean_std_fits(self, filename: str) -> tuple[np.ndarray, np.ndarray]:
        cube = self.load_fits(filename)
        return cube.mean(axis=0), cube.std(axis=0)

    def save_product(self, product, **values: np.ndarray) -> None:
        self.saved = values


class CorrectnessTests(unittest.TestCase):
    def test_sparse_profile_excludes_bad_pixels(self) -> None:
        ctx = _FlatContext()
        bad_map = np.zeros((8, 128), dtype=np.int16)
        bad_map[4, 10] = 1
        trace = np.full(128, 4.0)
        trace_error = np.zeros(128)

        with (
            patch("recipes.flat.compute_bad_map_from_stats", return_value=bad_map),
            patch("recipes.flat.trace_spectrum", return_value=(trace, trace_error)),
        ):
            run_flat(ctx, profile_width=3)

        selected = set(
            zip(ctx.saved["profile_ys"][0], ctx.saved["profile_xs"][0], strict=True)
        )
        self.assertNotIn((4, 10), selected)
        self.assertIn((-1, -1), selected)
        self.assertEqual(ctx.saved["profile_map"][0, 4, 10], 0)

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

    def test_object_entry_supports_legacy_and_explicit_dark(self) -> None:
        self.assertEqual(resolve_object_entry("science.fits"), ("science.fits", None))
        self.assertEqual(
            resolve_object_entry(
                {"data": Path("science.fits"), "dark": Path("dark.fits")}
            ),
            (Path("science.fits"), Path("dark.fits")),
        )

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


if __name__ == "__main__":
    unittest.main()
