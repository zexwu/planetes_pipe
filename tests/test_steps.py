"""Python step functions: products, QC and provenance without CPL."""
import tempfile
import unittest
from inspect import signature
from pathlib import Path

import numpy as np
from astropy.io import fits
from p2vm_fixture import metadata, write_raw

from planetes_pipe import steps
from planetes_pipe.io import dark as dark_io
from planetes_pipe.processing.p2vm import make_p2vm
from planetes_pipe.processing.reduce import reduce_science


class StepTests(unittest.TestCase):
    def test_dark_step_writes_products_with_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "PLANETES.2026-02-05T200000.fits"
            write_raw(source, np.arange(30, dtype=float).reshape(2, 3, 5),
                      metadata("DARK_RAW", n_wave=5, height=3))
            written = steps.dark(source, output_dir=root / "out", bad_pixel_threshold=4.0)
            self.assertEqual({k: p.name for k, p in written.items()}, {
                "DARK": "PLANETES.2026-02-05T200000_dark.fits",
                "BAD": "PLANETES.2026-02-05T200000_bad.fits"})
            np.testing.assert_allclose(dark_io.read(written["DARK"]).mean_adu,
                                       np.arange(15).reshape(3, 5) + 7.5)
            header = fits.getheader(written["DARK"])
            self.assertEqual(header["ESO PRO REC1 ID"], "planetes_dark")
            self.assertEqual(header["ESO PRO REC1 RAW1 NAME"], source.name)
            self.assertEqual(header["ESO PRO REC1 RAW1 CATG"], "DARK_RAW")
            self.assertEqual(header["ESO PRO DATANCOM"], 1)
            self.assertEqual(header["ESO PRO REC1 PARAM1 NAME"], "bad-dark-threshold")
            self.assertEqual(header["ESO PRO REC1 PARAM1 VALUE"], "4")
            self.assertEqual(header["PIPEFILE"], written["DARK"].name)
            self.assertNotIn("ESO DPR TYPE", header)
            self.assertIn("ESO QC MEDIANDARK SC", header)
            with self.assertRaises(OSError):  # Never overwrite.
                steps.dark(source, output_dir=root / "out")

    def test_step_defaults_match_the_scientific_functions(self):
        for step, function in [(steps.p2vm, make_p2vm), (steps.reduce, reduce_science)]:
            for name in ("gd_search_min_m", "gd_search_max_m"):
                self.assertEqual(signature(step).parameters[name].default,
                                 signature(function).parameters[name].default)
        for step in (steps.dark, steps.flat, steps.wave, steps.p2vm, steps.reduce):
            options = [name for name, item in signature(step).parameters.items()
                       if item.kind is item.KEYWORD_ONLY and name != "output_dir"
                       and not name.startswith("phase_")]
            self.assertTrue(set(options) <= steps.PARAMETERS.keys(), step.__name__)


if __name__ == "__main__":
    unittest.main()
