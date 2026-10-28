"""Optional PyEsoRex P2VM-to-science smoke test."""

import subprocess
import sys
import tempfile
import unittest
from dataclasses import fields, replace
from importlib.util import find_spec
from pathlib import Path

import numpy as np
from astropy.io import fits
from p2vm_fixture import input_rows, metadata, write_recipe_inputs

import planetes
from planetes.io.fits import read_product, write_product
from planetes.io.raw import write_raw
from planetes.products import (
    Flat,
    P2VM,
    P2VMRed,
    Wave,
)
from planetes.reduction import reduce_science


@unittest.skipUnless(
    find_spec("pyesorex") is not None, "PyEsoRex runtime not installed"
)
class ESOChainTests(unittest.TestCase):
    def test_science_recipe_preserves_dark_subtraction_and_masks(self):
        recipes = Path(planetes.__file__).parent / "recipes"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = write_recipe_inputs(root / "inputs")
            calibration = root / "p2vm"
            calibration.mkdir()
            sof = root / "p2vm.sof"
            sof.write_text(
                "".join(f"{path} {tag}\n" for tag, path in reversed(input_rows(inputs)))
            )
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pyesorex",
                    f"--recipe-dir={recipes}",
                    "planetes_p2vm",
                    str(sof),
                ],
                cwd=calibration,
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            with fits.open(inputs["PHASE_REFERENCE"]) as hdus:
                signal = np.array(hdus[1].data, copy=True)
            signal[0] = 0
            dark = np.full_like(signal, 7)
            science_file = root / "science.fits"
            write_raw(science_file, signal + dark, metadata("SINGLE_SCI_RAW", "1234"))
            dark_calib = root / "dark_calibration.fits"
            original = read_product(inputs["DARK"])
            write_product(
                dark_calib,
                replace(
                    original, mean_adu=np.full(signal.shape[1:], 7, dtype=np.float32)
                ),
            )
            science_inputs = {
                "SINGLE_SCI_RAW": science_file,
                "DARK": dark_calib,
                "FLAT": inputs["FLAT"],
                "WAVE": inputs["WAVE"],
                "P2VM": calibration / "planetes_p2vm.fits",
            }
            sof = root / "science.sof"
            sof.write_text(
                "".join(f"{path} {tag}\n" for tag, path in science_inputs.items())
            )
            output = root / "science"
            output.mkdir()
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pyesorex",
                    f"--recipe-dir={recipes}",
                    "planetes_reduce",
                    "--gd-search-min-m=-12e-6",
                    "--gd-search-max-m=12e-6",
                    str(sof),
                ],
                cwd=output,
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            actual = read_product(output / "planetes_reduced.fits", P2VMRed)
            flat = read_product(inputs["FLAT"], Flat)
            wave = read_product(inputs["WAVE"], Wave)
            p2vm = read_product(science_inputs["P2VM"], P2VM)
            expected = reduce_science(
                science_file,
                read_product(dark_calib),
                flat,
                wave,
                p2vm,
                gd_search_min_m=-12e-6,
                gd_search_max_m=12e-6,
            )
            for field in fields(expected):
                before, after = (
                    getattr(expected, field.name),
                    getattr(actual, field.name),
                )
                if isinstance(before, np.ndarray):
                    expected_array = before
                    np.testing.assert_array_equal(expected_array, after)
                else:
                    self.assertEqual(before, after)
            self.assertTrue(actual.flags[0].all())
            with fits.open(output / "planetes_reduced.fits") as hdus:
                self.assertEqual(hdus[0].header["ESO PRO CATG"], "SINGLE_SCI_P2VMRED")
                self.assertEqual(
                    float(hdus[0].header["ESO PRO REC1 PARAM1 VALUE"]), -12e-6
                )
                self.assertEqual(
                    float(hdus[0].header["ESO PRO REC1 PARAM2 VALUE"]), 12e-6
                )
                self.assertNotIn("ESO QC REJECTED RATIO", hdus[0].header)
