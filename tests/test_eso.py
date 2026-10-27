"""Optional real PyCPL/PyEsoRex execution, parameters and DFS failures."""

import subprocess
import sys
import tempfile
import unittest
from contextlib import chdir
from dataclasses import fields
from importlib.util import find_spec
from pathlib import Path
from unittest.mock import patch

import numpy as np
from astropy.io import fits
from p2vm_fixture import write_recipe_inputs

import planetes
from planetes.extraction import extract_calibration_files
from planetes.fits import read_product, write_product
from planetes.instrument import BASELINES, TELESCOPES
from planetes.models import (
    DarkCalibration,
    DarkImage,
    FlatCalibration,
    P2VMCalibration,
    ReducedObservables,
    WavelengthCalibration,
    wavelength_grid,
)
from planetes.p2vm import compute_p2vm
from planetes.reduction import reduce_file


@unittest.skipUnless(
    find_spec("pyesorex") is not None, "PyEsoRex runtime not installed"
)
class ESOChainTests(unittest.TestCase):
    def test_all_five_recipes_are_discoverable(self):
        recipes = Path(planetes.__file__).parent / "eso"
        result = subprocess.run(
            [sys.executable, "-m", "pyesorex", f"--recipe-dir={recipes}", "--recipes"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        for name in [
            "planetes_dark",
            "planetes_flat",
            "planetes_wave",
            "planetes_p2vm",
            "planetes_reduce",
        ]:
            self.assertIn(name, result.stdout)

    def test_science_recipe_preserves_dark_subtraction_and_masks(self):
        recipes = Path(planetes.__file__).parent / "eso"
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
            dark_calib = root / "dark_calibration.fits"
            write_product(
                dark_calib, DarkImage(np.full(signal.shape[1:], 7, dtype=np.float32))
            )
            science_inputs = {
                "SCIENCE_RAW": science_file,
                "DARK_CALIB": dark_calib,
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
            actual = read_product(output / "planetes_reduced.fits", ReducedObservables)
            flat = read_product(inputs["PLANETES_FLAT"], FlatCalibration)
            wave = read_product(inputs["PLANETES_WAVE"], WavelengthCalibration)
            p2vm = read_product(science_inputs["PLANETES_P2VM"], P2VMCalibration)
            expected = reduce_file(
                science_file, read_product(dark_calib), flat, wave, p2vm
            )
            for field in fields(expected):
                if field.name == "provenance":
                    continue
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


@unittest.skipUnless(find_spec("cpl") is not None, "PyCPL runtime not installed")
class P2VMRecipeTests(unittest.TestCase):
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
    def test_executor_applies_phase_correction_parameter(self):
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
        with fits.open(path) as hdus:
            self.assertEqual(hdus[0].header["ESO PRO REC1 PARAM1 VALUE"], "false")
            self.assertFalse(hdus[0].header["ESO DRS PHASE REF APPLIED"])


@unittest.skipUnless(find_spec("cpl") is not None, "PyCPL runtime not installed")
class DFSFailureTests(unittest.TestCase):
    def test_failure_after_dfs_header_is_cleaned_and_retry_succeeds(self):
        from cpl import ui

        from planetes.eso.common import save_product
        from planetes.eso.dark import PlanetesDark

        product = DarkCalibration(
            np.ones((2, 3), dtype=np.float32),
            np.ones((2, 3), dtype=np.float32),
            np.zeros((2, 3), dtype=np.int16),
            2,
        )
        with tempfile.TemporaryDirectory() as temporary, chdir(temporary):
            raw = Path("raw.fits").absolute()
            fits.PrimaryHDU(header=fits.Header({"INSTRUME": "PLANETES"})).writeto(raw)
            used = ui.FrameSet(
                [ui.Frame(file=str(raw), tag="DARK_RAW", group=ui.Frame.FrameGroup.RAW)]
            )
            with patch(
                "planetes.eso.common.fill_dfs_product",
                side_effect=ValueError("injected payload failure"),
            ):
                with self.assertRaisesRegex(ValueError, "injected"):
                    save_product(Path("dark.fits"), product, used, PlanetesDark())
            self.assertFalse(Path("dark.fits").exists())
            self.assertEqual(sorted(p.name for p in Path(".").iterdir()), ["raw.fits"])
            save_product(Path("dark.fits"), product, used, PlanetesDark())
            self.assertTrue(Path("dark.fits").is_file())
            before = Path("dark.fits").read_bytes()
            with self.assertRaises(FileExistsError):
                save_product(Path("dark.fits"), product, used, PlanetesDark())
            self.assertEqual(before, Path("dark.fits").read_bytes())

    def test_nondefault_dark_and_flat_parameters_use_public_task_defaults(self):
        from cpl import ui

        from planetes.eso.dark import PlanetesDark
        from planetes.eso.flat import PlanetesFlat
        from planetes.fits import read_product

        with tempfile.TemporaryDirectory() as temporary, chdir(temporary):
            ny, nx = 576, 128
            dark = np.full((3, ny, nx), 10, dtype=np.float32)
            header = fits.Header({"INSTRUME": "PLANETES"})
            fits.HDUList([fits.PrimaryHDU(header=header), fits.ImageHDU(dark)]).writeto(
                "raw_dark.fits"
            )
            dark_frames = ui.FrameSet(
                [ui.Frame(file=str(Path("raw_dark.fits").absolute()), tag="DARK_RAW")]
            )
            PlanetesDark().run(dark_frames, {"planetes.dark.bad_threshold": 6.5})
            measured = read_product(Path("planetes_dark.fits"))
            self.assertEqual(measured.provenance.parameters["bad_threshold"], 6.5)
            rows = np.arange(ny)
            profile = sum(
                np.exp(-0.5 * ((rows - center) / 2.0) ** 2)
                for center in 12 + 24 * np.arange(24)
            )
            image = np.tile((1000 * profile)[:, None], (1, nx)).astype(np.float32)
            frames = ui.FrameSet(
                [
                    ui.Frame(
                        file=str(Path("planetes_dark.fits").absolute()),
                        tag="DARK_CALIB",
                    )
                ]
            )
            for name in ("4", "3", "2", "1"):
                path = Path(f"tel{name}.fits").absolute()
                fits.HDUList(
                    [fits.PrimaryHDU(header=header), fits.ImageHDU(image[None] + dark)]
                ).writeto(path)
                frames.append(ui.Frame(file=str(path), tag=f"TEL_FLAT_{name}"))
            PlanetesFlat().run(frames, {"planetes.flat.profile_width": 3.5})
            flat = read_product(Path("planetes_flat.fits"))
            self.assertEqual(flat.provenance.parameters["profile_width"], 3.5)
            self.assertEqual(flat.profile_x_pixels.shape[0], 24)
