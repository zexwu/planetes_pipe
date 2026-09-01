from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from planetes.product import (
    P2VMPreprocProduct,
    WaveProduct,
    read_product,
    write_product,
)


class ProductTests(unittest.TestCase):
    def test_array_product_round_trip(self) -> None:
        product = WaveProduct(wave_map=np.arange(96).reshape(24, 4))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "wave.npz"
            write_product(path, product)
            loaded = read_product(path, WaveProduct)

            np.testing.assert_array_equal(loaded.wave_map, product.wave_map)
            self.assertFalse(list(Path(tmp).glob("*.tmp")))

    def test_mapping_fields_are_unpacked(self) -> None:
        baseline_regions = {
            baseline: list(range(4 * baseline, 4 * baseline + 4))
            for baseline in range(6)
        }
        baseline_telescopes = dict(
            enumerate(((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)))
        )
        product = P2VMPreprocProduct(
            spec_tel=np.ones((4, 24, 1, 2)),
            spec_bsl=np.ones((6, 24, 1, 2)),
            spec_wavesc=np.ones((24, 1, 2)),
            spec_flat=np.ones((24, 1, 2)),
            tel_regs={0: [0, 1]},
            bsl_regs={0: [0, 1, 2, 3]},
            bsl_to_reg=baseline_regions,
            bsl_to_tel=baseline_telescopes,
            wl_grid=np.array([1.2, 1.3]),
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "p2vm_preproc.npz"
            write_product(path, product)
            loaded = read_product(path, P2VMPreprocProduct)

            self.assertEqual(loaded.bsl_to_tel, baseline_telescopes)

    def test_schema_mismatch_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "wave.npz"
            np.savez(path, typo=np.ones(1))
            with self.assertRaisesRegex(ValueError, "missing keys.*unexpected keys"):
                read_product(path, WaveProduct)

    def test_wrong_planetes_axis_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "24 outputs"):
                write_product(
                    Path(tmp) / "wave.npz",
                    WaveProduct(wave_map=np.ones((23, 4))),
                )


if __name__ == "__main__":
    unittest.main()
