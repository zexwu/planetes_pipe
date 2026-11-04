"""science.py: visibility normalization and P2VMRED storage precision."""


import tempfile
import unittest
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
from fixtures import calibration_arrays, calibration_spectra

from planetes_pipe.flat import Flat, read_flat
from planetes_pipe.p2vm import P2VM, fit_p2vm, read_p2vm
from planetes_pipe.science import (
    normalize_coherent_flux,
    read_reduced,
    reduce_spectra,
    write_reduced,
)
from planetes_pipe.wave import Wave, read_wave

READERS = {Flat: read_flat, Wave: read_wave, P2VM: read_p2vm}


class ReducedProductTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spectra = calibration_spectra(n_wave=15)
        p2vm = fit_p2vm(spectra)
        science = calibration_arrays(n_wave=15)[2]
        science[:, 0] = 0
        cls.product = reduce_spectra(science, p2vm)

    def test_complex_flux_and_flags_preserve_storage_precision(self):
        coherent = self.product.coherent_flux_adu.copy()
        coherent[0, 0, 0] = complex(-0.0, -0.0)
        coherent[0, 0, 1] = complex(np.nan, np.nan)
        product = replace(self.product, coherent_flux_adu=coherent)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "science.fits"
            write_reduced(path, product)
            loaded = read_reduced(path)
            for field in fields(product):
                before, after = (
                    getattr(product, field.name),
                    getattr(loaded, field.name),
                )
                if isinstance(before, np.ndarray):
                    expected = before
                    if before.dtype.kind in "fc":
                        stored = (
                            np.complex64 if before.dtype.kind == "c" else np.float32
                        )
                        expected = before.astype(stored).astype(before.dtype)
                    np.testing.assert_array_equal(expected, after)
                else:
                    self.assertEqual(before, after)
            np.testing.assert_array_equal(
                coherent.astype(np.complex64).astype(np.complex128).view(np.uint64),
                loaded.coherent_flux_adu.view(np.uint64),
            )
            self.assertTrue(loaded.flags[0].all())
            self.assertEqual(loaded.flags.dtype, bool)


class ReduceTests(unittest.TestCase):
    def test_visibility_normalization_flags_invalid_flux_products(self) -> None:
        coherent_flux = np.tile(np.array([[[6 + 8j, 1 + 0j]]]), (1, 6, 1))
        flux = np.array([[[4.0, -1.0], [25.0, 2.0], [25.0, 2.0], [25.0, 2.0]]])

        visamp, flag = normalize_coherent_flux(coherent_flux, flux)

        self.assertEqual(visamp[0, 0, 0], 1.0)
        self.assertFalse(flag[0, 0, 0])
        self.assertTrue(np.isnan(visamp[0, 0, 1]))
        self.assertTrue(flag[0, 0, 1])


if __name__ == "__main__":
    unittest.main()
