"""Science FITS storage must preserve masks, complex values, and their units."""

from dataclasses import fields, replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from astropy.io import fits

from planetes.algorithms.p2vm import compute_p2vm
from planetes.algorithms.reduction import compute_reduce
from planetes.io.products import append_product, read_product, write_product
from planetes.io.reference import load_wavelength_lines
from planetes.product import CalibrationSpectra, ReducedProduct
from p2vm_fixture import calibration_arrays


class ScienceProductTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        values = calibration_arrays(n_wave=15)
        spectra = CalibrationSpectra(*values[:6])
        p2vm = compute_p2vm(spectra, phase_correction=False)
        science = spectra.phase_reference.copy()
        science[:, 0] = 0
        cls.product = compute_reduce(science, p2vm)

    def test_complex_flux_and_invalid_flags_round_trip_exactly(self):
        coherent = self.product.coherent_flux_adu.copy()
        coherent[0, 0, 0] = complex(-0.0, -0.0)
        coherent[0, 0, 1] = complex(np.nan, np.nan)
        product = replace(self.product, coherent_flux_adu=coherent)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "science.fits"
            write_product(path, product)
            loaded = read_product(path, ReducedProduct)
            for field in fields(product):
                before, after = (
                    getattr(product, field.name),
                    getattr(loaded, field.name),
                )
                if isinstance(before, np.ndarray):
                    np.testing.assert_array_equal(before, after)
                else:
                    self.assertEqual(before, after)
            np.testing.assert_array_equal(
                coherent.view(np.uint64), loaded.coherent_flux_adu.view(np.uint64)
            )
            self.assertTrue(loaded.flags[0].all())
            self.assertEqual(loaded.flags.dtype, bool)

    def test_corrupt_boolean_encoding_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "science.fits"
            write_product(path, self.product)
            with fits.open(path, mode="update") as hdus:
                hdus["FLAGS"].data[0, 0, 0] = 2
                hdus["FLAGS"].add_checksum()
            with self.assertRaisesRegex(ValueError, "boolean visibility flags"):
                read_product(path, ReducedProduct)

    def test_append_cannot_modify_a_raw_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "raw.fits"
            fits.PrimaryHDU(np.ones((3, 4))).writeto(path)
            original = path.read_bytes()
            with self.assertRaisesRegex(ValueError, "new empty CPL"):
                append_product(path, self.product)
            self.assertEqual(path.read_bytes(), original)

    def test_static_reference_retains_original_values(self):
        lines = load_wavelength_lines()
        self.assertEqual(len(lines), 30)
        self.assertAlmostEqual(lines[0], 0.965780)
        self.assertAlmostEqual(lines[-1], 1.409370)

    def test_diagnostic_plotting_consumes_a_product_without_configuration(self):
        import matplotlib

        matplotlib.use("Agg", force=True)
        from planetes.plotting import plot_reduced

        with (
            tempfile.TemporaryDirectory() as temporary,
            patch("planetes.plotting.PdfPages") as writer,
        ):
            pages = writer.return_value.__enter__.return_value
            pages.savefig.side_effect = lambda figure: figure.canvas.draw()
            plot_reduced(self.product, Path(temporary) / "diagnostic.pdf")
            self.assertGreater(pages.savefig.call_count, 0)


if __name__ == "__main__":
    unittest.main()
