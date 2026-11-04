"""Product headers are valid FITS with complete provenance; QC values are measured as defined."""


import tempfile
import unittest
import warnings
from inspect import signature
from pathlib import Path

import numpy as np
from astropy.io import fits
from fixtures import write_recipe_inputs

from planetes_pipe.steps import dark as dark_step
from planetes_pipe.steps import p2vm, science
from planetes_pipe.steps import wave as wave_step
from planetes_pipe.steps.dark import Bad, Dark
from planetes_pipe.steps.wave import Wave

REQUIRED_PRO = ("ESO PRO CATG", "ESO PRO TYPE", "ESO PRO TECH", "ESO PRO DID",
                "ESO PRO REC1 ID", "ESO PRO REC1 PIPE ID", "ESO PRO DATANCOM", "PIPEFILE")


class HeaderTests(unittest.TestCase):
    def check(self, path: Path):
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # Truncated cards raise VerifyWarning.
            with fits.open(path) as hdus:
                hdus.verify("exception")
                header = hdus[0].header
                for card in header.cards:
                    self.assertLessEqual(len(card.image), 80)
                    if isinstance(card.value, str):
                        self.assertLessEqual(len(card.value), 68, card.keyword)
                    key = card.keyword
                    self.assertFalse(key.startswith("ESO DPR "), key)
                for key in REQUIRED_PRO:
                    self.assertIn(key, header, f"{path.name}: missing {key}")

    def test_p2vm_and_science_products(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temporary:
            root = Path(temporary)
            inputs = write_recipe_inputs(root / "inputs")
            calibration = p2vm.run(inputs["DARK"], inputs["FLAT"], inputs["WAVE"],
                                   inputs["FLAT_RAW"], inputs["P2VM_RAW"],
                                   output_dir=root / "out",
                                   phase_file=inputs["PHASE_REFERENCE"],
                                   phase_dark_file=inputs["PHASE_DARK_CALIB"],
                                   keep_diagnostics=True)
            reduced = science.run(inputs["PHASE_REFERENCE"], inputs["DARK"], inputs["FLAT"],
                                 inputs["WAVE"], calibration["P2VM"], output_dir=root / "out")
            for path in (*calibration.values(), *reduced.values()):
                with self.subTest(path=path.name):
                    self.check(path)
            self.assertIn("ESO QC P2VM_COHERENCE_SC43", fits.getheader(calibration["P2VM"]))


class QCTests(unittest.TestCase):
    def test_qc_definitions_and_step_statistics_preserve_units_and_missing_values(self):
        dark = Dark(np.array([[1., 3.]]))
        values = {keyword.name: value for keyword, value in dark_step.qc(dark).items()}
        self.assertEqual(values, {"QC.MEDIANDARK.SC": 2., "QC.DARKRMS.SC": None})
        bad = Bad(np.array([[0, 1, 2, 4, 7]], dtype=np.int16))
        values = {keyword.name: value for keyword, value in dark_step.qc(bad).items()}
        self.assertEqual(values, {"QC.BADPIX.SC": 4, "QC.BADPIX_DARK.SC": 3,
                                  "QC.BADPIX_RMS.SC": 2})
        wave = Wave(np.array([[1., 2., 3.], [1.5, 2., 2.5]]) * 1e-6,
                    fit_residual_m=np.array([-1., 1.]) * 1e-9)
        quality = {keyword.name: value for keyword, value in wave_step.qc(wave).items()}
        self.assertEqual(quality["QC.MINWAVE.SC"], 1.5e-6)
        np.testing.assert_allclose(quality["QC.MAXWAVE.SC"], 2.5e-6, rtol=1e-15, atol=0)
        self.assertAlmostEqual(quality["QC.RMSWAVE.SC"], np.sqrt(2) * 1e-9)
        self.assertEqual(quality["QC.NWAVE.SC"], 2)

    def test_run_defaults_match_the_scientific_functions(self):
        for module, function in [(p2vm, p2vm.fit_p2vm), (science, science.reduce_science)]:
            for name in ("gd_search_min_m", "gd_search_max_m"):
                self.assertEqual(signature(module.run).parameters[name].default,
                                 signature(function).parameters[name].default)


if __name__ == "__main__":
    unittest.main()
