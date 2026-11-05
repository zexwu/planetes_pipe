"""plots/: one portrait report per product category, the P2VM quality checks, no overwriting."""
import re
import tempfile
import unittest
from pathlib import Path

import numpy as np
from astropy.io import fits
from fixtures import calibration_arrays, calibration_spectra, write_product

from planetes_pipe import plots
from planetes_pipe.io.keywords import QC_SATURATED
from planetes_pipe.plots import p2vm as p2vm_pages
from planetes_pipe.steps.dark import Bad, Dark
from planetes_pipe.steps.p2vm import fit_p2vm
from planetes_pipe.steps.science import reduce_spectra


def page_count(path: Path) -> int:
    return len(re.findall(rb"/Type\s*/Page[^s]", path.read_bytes()))


class PlotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        values = calibration_arrays(n_wave=15)
        flat, wave = values[-2:]
        cls.p2vm = fit_p2vm(calibration_spectra(n_wave=15), keep_diagnostics=True)
        flags = np.zeros((24, 15), dtype=np.uint8)
        flags[3, 4], flags[5, 6], flags[7, 8] = 1, 2, 4
        cls.products = {
            "dark": Dark(np.full((24, 15), 7.0), np.ones((24, 15)), 2),
            "bad": Bad(flags),
            "flat": flat,
            "wave": wave,
            "p2vm": cls.p2vm,
            "singlescip2vmred": reduce_spectra(values[2], cls.p2vm),
        }

    def test_every_product_gets_its_report_and_none_is_overwritten(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            for name, product in self.products.items():
                path = Path(tmp) / f"x_{name}.fits"
                write_product(path, product)
                with self.subTest(name):
                    pdf = plots.report(path)
                    self.assertEqual(pdf, path.with_suffix(".pdf"))
                    self.assertGreater(page_count(pdf), 0)
                    with self.assertRaises(FileExistsError):
                        plots.report(path)
            n_sections = len(p2vm_pages.sections(self.p2vm))
            self.assertGreaterEqual(n_sections, 15)  # quality ... phase residual, as GRAVITY + scans
            self.assertLess(page_count(Path(tmp) / "x_p2vm.pdf"), n_sections)  # sections share pages

    def test_a_file_without_a_known_category_is_refused(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            path = Path(tmp) / "raw.fits"
            fits.PrimaryHDU(np.zeros((2, 2))).writeto(path)
            with self.assertRaisesRegex(ValueError, "no report for PRO.CATG"):
                plots.report(path)

    def test_quality_follows_gravity_rules_and_flags_saturation(self):
        clean = p2vm_pages.quality(self.p2vm)
        self.assertEqual(clean["checks"]["Saturated raw samples"], 0)
        self.assertEqual(len(clean["visibility"]), 6)
        header = fits.Header()
        QC_SATURATED.at("SC43").set(header, 39725)
        result = p2vm_pages.quality(self.p2vm, header)
        self.assertEqual(result["saturated"], {"SC43": 39725})
        self.assertEqual(result["checks"]["Saturated raw samples"], 1)
        self.assertNotEqual(result["verdict"], "GOOD")

    def test_calibration_set_in_one_pdf(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            out = plots.save_calibration_plots(
                self.products["dark"], self.products["bad"], self.products["flat"],
                self.products["wave"], self.p2vm, Path(tmp) / "calibration_report.pdf")
            self.assertGreater(page_count(out), 5)


if __name__ == "__main__":
    unittest.main()
