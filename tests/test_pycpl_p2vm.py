"""Exercise the real PyCPL adapter when its optional runtime is installed."""

from contextlib import chdir
from dataclasses import fields
from importlib.util import find_spec
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import planetes
import numpy as np
from astropy.io import fits

from planetes.p2vm import compute_p2vm
from planetes.instrument import BASELINES, TELESCOPES
from planetes.fits import read_product
from planetes.models import (
    FlatCalibration,
    P2VMCalibration,
    WavelengthCalibration,
    wavelength_grid,
)
from planetes.extraction import extract_calibration_files
from p2vm_fixture import write_recipe_inputs


@unittest.skipUnless(find_spec("cpl") is not None, "PyCPL runtime not installed")
class PyCPLP2VMTests(unittest.TestCase):
    def setUp(self):
        from cpl import ui
        from planetes.eso.p2vm import PlanetesP2VM

        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.paths = write_recipe_inputs(self.root / "inputs")
        self.frames = ui.FrameSet(
            [
                ui.Frame(file=str(path), tag=tag)
                for tag, path in reversed(self.paths.items())
            ]
        )
        self.recipe = PlanetesP2VM()

    def test_shuffled_tagged_inputs_match_direct_scientific_calculation(self):
        flat = read_product(self.paths["PLANETES_FLAT"], FlatCalibration)
        wave = read_product(self.paths["PLANETES_WAVE"], WavelengthCalibration)
        spectra = extract_calibration_files(
            flat,
            wave,
            read_product(self.paths["DARK_CALIB"]),
            wavelength_um=wavelength_grid(),
            telescope_flats={
                name: self.paths[f"TEL_FLAT_{name}"] for name in TELESCOPES
            },
            baseline_cubes={name: self.paths[f"P2VM_RAW_{name}"] for name in BASELINES},
            phase_reference=self.paths["PHASE_REFERENCE"],
            phase_dark=read_product(self.paths["PHASE_DARK_CALIB"]),
        )
        expected = compute_p2vm(spectra, phase_correction=True)
        with chdir(self.root):
            products = self.recipe.run(self.frames, {})
        self.assertEqual(len(products), 1)
        output = self.root / products[0].file
        actual = read_product(output, P2VMCalibration)
        for field in fields(expected):
            if field.name == "provenance":
                continue
            before, after = getattr(expected, field.name), getattr(actual, field.name)
            if isinstance(before, np.ndarray):
                np.testing.assert_array_equal(before, after)
            else:
                self.assertEqual(before, after)
        with fits.open(output) as hdus:
            header = hdus[0].header
            self.assertEqual(header["ESO PRO REC1 ID"], "planetes_p2vm")
            self.assertEqual(header["ESO PRO REC1 PARAM1 VALUE"], "true")
            self.assertTrue(header["ESO DRS PHASE REF APPLIED"])
            self.assertTrue(header["DATAMD5"])
            self.assertFalse(any(key.startswith("ESO DPR ") for key in header))
            recorded = [value for key, value in header.items() if key.endswith(" NAME")]
            for path in self.paths.values():
                self.assertIn(path.name, recorded)

    def test_duplicate_and_missing_tags_are_rejected(self):
        from cpl import ui

        with self.assertRaisesRegex(ValueError, "Missing required"):
            self.recipe.run(ui.FrameSet(), {})
        duplicate = ui.FrameSet([self.frames[0], self.frames[0]])
        with self.assertRaisesRegex(ValueError, "Duplicate input tag"):
            self.recipe.run(duplicate, {})

    @unittest.skipUnless(
        find_spec("pyesorex") is not None, "PyEsoRex runtime not installed"
    )
    def test_pyesorex_discovers_and_runs_recipe_without_yaml(self):
        recipe_directory = Path(planetes.__file__).parent / "eso"
        sof = self.root / "p2vm.sof"
        sof.write_text("".join(f"{path} {tag}\n" for tag, path in self.paths.items()))
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pyesorex",
                f"--recipe-dir={recipe_directory}",
                "planetes_p2vm",
                "--phase-correction=false",
                str(sof),
            ],
            cwd=self.root,
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        path = self.root / "planetes_p2vm.fits"
        product = read_product(path, P2VMCalibration)
        self.assertEqual(product.p2vm.shape, (16, 24, 151))
        self.assertFalse(list(self.root.glob("*.yaml")))
        with fits.open(path) as hdus:
            self.assertEqual(hdus[0].header["ESO PRO REC1 PARAM1 VALUE"], "false")
            self.assertFalse(hdus[0].header["ESO DRS PHASE REF APPLIED"])


if __name__ == "__main__":
    unittest.main()
