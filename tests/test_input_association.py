"""Header association must not depend on names or mix ambiguous exposures."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
from astropy.io import fits
from p2vm_fixture import write_raw
from planetes_pipe.model.instrument import layout, DetectorSetup
from planetes_pipe.model.raw import RawMetadata
from planetes_pipe.io.association import find_calibration_inputs

class CalibrationSelectionTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.count = 0
        self.dark = self.raw("DARK_RAW", "", 0.1)
        self.lamp_dark = self.raw("DARK_RAW", "", 1.0)
        self.lamp = self.raw("WAVELAMP_RAW", "1234", 1.0)
        self.flats = {t: self.raw("FLAT_RAW", t, 0.1) for t in layout.telescopes}
        self.scans = {b: self.raw("P2VM_RAW", b, 0.1) for b in layout.baselines}

    def raw(self, category, opened, exposure, width=3):
        self.count += 1
        path = self.root / f"arbitrary-{100-self.count}.fits"
        setup = DetectorSetup(exposure_s=exposure, width=width, height=2,
                              origin_x=0, origin_y=0, binning_x=1, binning_y=1, detector="synthetic")
        write_raw(path, np.zeros((2, 2, width)), RawMetadata(
            category, tuple(str(i) in opened for i in range(1, 5)), setup))
        return path

    def test_uses_headers_and_matches_two_dits(self):
        self.raw("SINGLE_SCI_RAW", "1234", 0.1)
        self.raw("DARK_RAW", "", 0.1, width=4)
        self.assertEqual(find_calibration_inputs(self.root),
                         (self.dark, self.lamp_dark, self.lamp, self.flats, self.scans))

    def test_duplicate_flat_is_not_silently_overwritten(self):
        self.raw("FLAT_RAW", "4", 0.1)
        with self.assertRaisesRegex(ValueError, "flat S4.*found 2"):
            find_calibration_inputs(self.root)

    def test_duplicate_matching_dark_requires_selection(self):
        self.raw("DARK_RAW", "", 0.1)
        with self.assertRaisesRegex(ValueError, "dark for 0.1 s.*found 2"):
            find_calibration_inputs(self.root)

    def test_missing_shutter_does_not_use_filename(self):
        with fits.open(self.flats["4"], mode="update") as hdus:
            del hdus[0].header["PLANETES INS SHUT4 ST"]
        with self.assertRaisesRegex(ValueError, "Missing shutter"):
            find_calibration_inputs(self.root)

    def test_missing_lamp_dark_does_not_reuse_main_dark(self):
        self.lamp_dark.unlink()
        with self.assertRaisesRegex(ValueError, "dark for 1 s.*found 0"):
            find_calibration_inputs(self.root)

    def test_missing_flat_exposure_is_not_assumed(self):
        with fits.open(self.flats["4"], mode="update") as hdus:
            del hdus[0].header["ESO DET WIN1 DIT1"]
        with self.assertRaisesRegex(ValueError, "without exposure"):
            find_calibration_inputs(self.root)


if __name__ == "__main__":
    unittest.main()
