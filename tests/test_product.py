from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from planetes.product import P2VMProduct, WaveProduct, read_product, write_product


class ProductTests(unittest.TestCase):
    def test_array_product_round_trip(self) -> None:
        product = WaveProduct(wavelength_map_um=np.arange(96).reshape(24, 4))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "wave.npz"
            write_product(path, product)
            loaded = read_product(path, WaveProduct)

            np.testing.assert_array_equal(
                loaded.wavelength_map_um, product.wavelength_map_um
            )

    def test_mapping_fields_are_unpacked(self) -> None:
        baseline_regions = {
            baseline: list(range(4 * baseline, 4 * baseline + 4))
            for baseline in range(6)
        }
        baseline_telescopes = dict(
            enumerate(((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)))
        )
        product = P2VMProduct(
            p2vm=np.ones(1),
            wavelength_um=np.array([1.2, 1.3]),
            baseline_regions=baseline_regions,
            baseline_telescopes=baseline_telescopes,
            transmission=np.ones(1),
            coherence_amplitude=np.ones(1),
            coherence_phase_rad=np.ones(1),
            opd_um=np.ones(1),
            group_delay_um=np.ones(1),
            ellipse_phase_rad=np.ones(1),
            ellipse_amplitude=np.ones(1),
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "p2vm.npz"
            write_product(path, product)
            loaded = read_product(path, P2VMProduct)

            self.assertEqual(loaded.baseline_regions, baseline_regions)
            self.assertEqual(loaded.baseline_telescopes, baseline_telescopes)

    def test_schema_mismatch_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "wave.npz"
            np.savez(path, typo=np.ones(1))
            with self.assertRaisesRegex(ValueError, "missing keys.*unexpected keys"):
                read_product(path, WaveProduct)


if __name__ == "__main__":
    unittest.main()
