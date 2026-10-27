"""Run the installed recipe executor with tagged synthetic calibration and science FITS."""

from importlib.util import find_spec
from pathlib import Path
from dataclasses import fields
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from astropy.io import fits

from planetes.algorithms.reduction import compute_reduce
from planetes.io.products import read_product
from planetes.processing import extract_target_spectra
from planetes.product import FlatProduct, P2VMProduct, ReducedProduct, WaveProduct
from p2vm_fixture import write_recipe_inputs


@unittest.skipUnless(
    find_spec("pyesorex") is not None, "PyEsoRex runtime not installed"
)
class ESOChainTests(unittest.TestCase):
    def test_all_four_recipes_are_discoverable(self):
        recipes = Path(__file__).resolve().parents[1] / "planetes/eso"
        result = subprocess.run(
            [sys.executable, "-m", "pyesorex", f"--recipe-dir={recipes}", "--recipes"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        for name in [
            "planetes_flat",
            "planetes_wave",
            "planetes_p2vm",
            "planetes_reduce",
        ]:
            self.assertIn(name, result.stdout)

    def test_science_recipe_preserves_dark_subtraction_and_masks(self):
        recipes = Path(__file__).resolve().parents[1] / "planetes/eso"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = write_recipe_inputs(root / "inputs")
            calibration = root / "p2vm"
            calibration.mkdir()
            sof = root / "p2vm.sof"
            sof.write_text(
                "".join(f"{path} {tag}\n" for tag, path in reversed(inputs.items()))
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
                header = hdus[0].header.copy()
            signal[0] = 0
            header["HIERARCH ESO DPR CATG"] = "SCIENCE"
            header["HIERARCH ESO DPR TYPE"] = "OBJECT"
            dark = np.full_like(signal, 7)
            science_file, dark_file = root / "science.fits", root / "science_dark.fits"
            fits.HDUList(
                [fits.PrimaryHDU(header=header), fits.ImageHDU(signal + dark)]
            ).writeto(science_file)
            fits.HDUList([fits.PrimaryHDU(header=header), fits.ImageHDU(dark)]).writeto(
                dark_file
            )
            science_inputs = {
                "SCIENCE_RAW": science_file,
                "SCIENCE_DARK": dark_file,
                "PLANETES_FLAT": inputs["PLANETES_FLAT"],
                "PLANETES_WAVE": inputs["PLANETES_WAVE"],
                "PLANETES_P2VM": calibration / "planetes_p2vm.fits",
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
                    str(sof),
                ],
                cwd=output,
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            actual = read_product(output / "planetes_reduced.fits", ReducedProduct)
            flat = read_product(inputs["PLANETES_FLAT"], FlatProduct)
            wave = read_product(inputs["PLANETES_WAVE"], WaveProduct)
            p2vm = read_product(science_inputs["PLANETES_P2VM"], P2VMProduct)
            expected = compute_reduce(
                extract_target_spectra(science_file, flat, wave, dark=dark_file), p2vm
            )
            for field in fields(expected):
                before, after = (
                    getattr(expected, field.name),
                    getattr(actual, field.name),
                )
                if isinstance(before, np.ndarray):
                    np.testing.assert_array_equal(before, after)
                else:
                    self.assertEqual(before, after)
            self.assertTrue(actual.flags[0].all())
            with fits.open(output / "planetes_reduced.fits") as hdus:
                self.assertEqual(hdus[0].header["PLTLEVEL"], "INSTRUMENTAL")
                self.assertIn("ESO QC VIS FLAG FRACTION", hdus[0].header)


if __name__ == "__main__":
    unittest.main()
