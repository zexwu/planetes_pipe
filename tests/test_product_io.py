from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from planetes.product_io import read_product, write_product
from planetes.products import P2VMPreprocProduct, WaveProduct


class ProductIOTests(unittest.TestCase):
    def test_array_product_round_trip(self) -> None:
        product = WaveProduct(wave_map=np.arange(12).reshape(3, 4))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "wave.npz"
            write_product(path, product)
            loaded = read_product(path, WaveProduct)

            np.testing.assert_array_equal(loaded.wave_map, product.wave_map)
            self.assertFalse(list(Path(tmp).glob("*.tmp")))

    def test_mapping_fields_are_unpacked(self) -> None:
        array = np.ones((1, 1, 1))
        product = P2VMPreprocProduct(
            spec_tel=array,
            spec_bsl=array,
            spec_wavesc=array,
            spec_flat=array,
            tel_regs={0: [0, 1]},
            bsl_regs={0: [0, 1, 2, 3]},
            bsl_to_reg={0: [0, 2, 1, 3]},
            bsl_to_tel={0: (0, 1)},
            wl_grid=np.array([1.2]),
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "p2vm_preproc.npz"
            write_product(path, product)
            loaded = read_product(path, P2VMPreprocProduct)

            self.assertEqual(loaded.bsl_to_tel, {0: (0, 1)})

    def test_schema_mismatch_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "wave.npz"
            np.savez(path, typo=np.ones(1))
            with self.assertRaisesRegex(ValueError, "missing keys.*unexpected keys"):
                read_product(path, WaveProduct)


if __name__ == "__main__":
    unittest.main()
