"""Optional PyEsoRex P2VM-to-science smoke test."""
import logging
import subprocess
import sys
import tempfile
import unittest
from dataclasses import fields, replace
from importlib.util import find_spec
from pathlib import Path

import numpy as np
from astropy.io import fits
from fixtures import input_rows, metadata, write_raw, write_recipe_inputs

import planetes_pipe
from planetes_pipe.dark import read_bad, read_dark, write_dark
from planetes_pipe.flat import read_flat
from planetes_pipe.p2vm import read_p2vm
from planetes_pipe.raw import read_raw
from planetes_pipe.science import read_reduced, reduce_science
from planetes_pipe.wave import read_wave


@unittest.skipUnless(
    find_spec("pyesorex") is not None, "PyEsoRex runtime not installed"
)
class ESOChainTests(unittest.TestCase):
    def test_science_recipe_preserves_dark_subtraction_and_masks(self):
        recipes = Path(planetes_pipe.__file__).parent / "eso"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = write_recipe_inputs(root / "inputs")
            # Give each scan a distinct timestamp/unknown-time identity, then
            # reverse the SOF below to check that baseline 43 remains the anchor.
            for index, old_path in enumerate(inputs["P2VM_RAW"], 6):
                path = old_path.with_name(
                    f"PLANETES.2026-02-05TUNKNOWN.{index:04d}.fits"
                )
                old_path.rename(path)
                # Header inheritance must use the same physical reference as
                # naming, even when the literal filename has an UNKNOWN marker.
                with fits.open(path, mode="update") as hdus:
                    hdus[0].header["DATE-OBS"] = f"2026-02-05T20:00:{index:02d}"
                    hdus[0].add_checksum()
                inputs["P2VM_RAW"][index - 6] = path
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
                    "--msg-level=error",
                    "--log-level=debug",
                    "planetes_p2vm",
                    str(sof),
                ],
                cwd=calibration,
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertNotIn("Fitting baseline", result.stdout + result.stderr)
            logfile = (calibration / "pyesorex.log").read_text()
            self.assertIn("Fitting baseline 43", logfile)
            self.assertIn("Fitting ellipses for phase extraction", logfile)
            header = fits.getheader(
                calibration / "PLANETES.2026-02-05TUNKNOWN.0006_p2vm.fits"
            )
            self.assertEqual(header["DATE-OBS"], "2026-02-05T20:00:06")
            self.assertEqual(
                header["ESO PRO REC1 RAW1 NAME"], inputs["P2VM_RAW"][0].name
            )
            with fits.open(inputs["PHASE_REFERENCE"]) as hdus:
                signal = np.array(hdus[1].data, copy=True)
            signal[0] = 0
            dark = np.full_like(signal, 7)
            science_file = root / "PLANETES.2026-02-05T210609.fits"
            write_raw(science_file, signal + dark, metadata("SINGLE_SCI_RAW", "1234"))
            dark_calib = root / "dark_calibration.fits"
            original = read_dark(inputs["DARK"])
            write_dark(
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
                "P2VM": calibration / "PLANETES.2026-02-05TUNKNOWN.0006_p2vm.fits",
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
            actual = read_reduced(
                output / "PLANETES.2026-02-05T210609_singlescip2vmred.fits"
            )
            flat = read_flat(inputs["FLAT"])
            wave = read_wave(inputs["WAVE"])
            p2vm = read_p2vm(science_inputs["P2VM"])
            expected = reduce_science(
                read_raw(science_file, "SINGLE_SCI_RAW"),
                read_dark(dark_calib),
                flat,
                wave,
                p2vm,
                gd_search_min_m=-1.2e-05,
                gd_search_max_m=1.2e-05,
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
            with fits.open(
                output / "PLANETES.2026-02-05T210609_singlescip2vmred.fits"
            ) as hdus:
                self.assertEqual(hdus[0].header["ESO PRO CATG"], "SINGLE_SCI_P2VMRED")
                self.assertEqual(
                    float(hdus[0].header["ESO PRO REC1 PARAM1 VALUE"]), -12e-6
                )
                self.assertEqual(
                    float(hdus[0].header["ESO PRO REC1 PARAM2 VALUE"]), 12e-6
                )
                self.assertNotIn("ESO QC REJECTED RATIO", hdus[0].header)

    def test_dark_recipe_names_products_after_the_raw_input(self):
        recipes = Path(planetes_pipe.__file__).parent / "eso"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "PLANETES.2026-02-05TUNKNOWN.0001.fits"
            write_raw(
                source,
                np.arange(30, dtype=float).reshape(2, 3, 5),
                metadata("DARK_RAW", n_wave=5, height=3),
            )
            sof = root / "dark.sof"
            sof.write_text(f"{source} DARK_RAW\n")
            output = root / "out"
            output.mkdir()
            result = subprocess.run(
                [sys.executable, "-m", "pyesorex", f"--recipe-dir={recipes}",
                 "planetes_dark", str(sof)],
                cwd=output, capture_output=True, text=True, timeout=60,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            for category, read in (("dark", read_dark), ("bad", read_bad)):
                path = output / f"{source.stem}_{category}.fits"
                read(path)  # Includes the checksum check after signing.
                header = fits.getheader(path)
                self.assertEqual(header["ESO PRO CATG"], category.upper())
                self.assertEqual(header["ESO PRO REC1 RAW1 NAME"], source.name)
                self.assertEqual(len(header["DATAMD5"]), 32)

    def test_p2vm_recipe_signs_products_and_records_phase_inputs(self):
        recipes = Path(planetes_pipe.__file__).parent / "eso"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = write_recipe_inputs(root / "inputs")
            for path in inputs["P2VM_RAW"]:
                with fits.open(path, mode="update") as hdus:
                    hdus[0].header["HIERARCH ESO DPR TECH"] = "INTERFEROMETRY"
                    hdus[0].add_checksum()
            sof = root / "p2vm.sof"
            sof.write_text("".join(f"{p} {tag}\n" for tag, p in input_rows(inputs)))
            output = root / "out"
            output.mkdir()
            result = subprocess.run(
                [sys.executable, "-m", "pyesorex", f"--recipe-dir={recipes}",
                 "planetes_p2vm", str(sof)],
                cwd=output, capture_output=True, text=True, timeout=60,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            p2vm_raw = next(p for p in inputs["P2VM_RAW"] if p.stem.endswith("43"))
            path = output / f"{p2vm_raw.stem}_p2vm.fits"
            read_p2vm(path)  # Checksums still valid after signing.
            header = fits.getheader(path)
            self.assertEqual(len(header["DATAMD5"]), 32)
            self.assertEqual(header["PIPEFILE"], path.name)
            self.assertEqual(header["ESO PRO TECH"], "INTERFEROMETRY")
            self.assertEqual(header["ESO PRO DATANCOM"], 11)  # 4 flats, 6 scans, reference.
            calibrations = [header[f"ESO PRO REC1 CAL{n} CATG"] for n in range(1, 5)]
            self.assertEqual(calibrations, ["DARK", "FLAT", "WAVE", "PHASE_DARK_CALIB"])

    def test_recipe_logging_restores_python_configuration_after_failure(self):
        from planetes_pipe.eso.recipes import recipe_logging

        logger = planetes_pipe.log
        original = logger.handlers, logger.level, logger.propagate
        handler = logging.NullHandler()
        try:
            logger.handlers = [handler]
            logger.setLevel(logging.WARNING)
            logger.propagate = True
            for _ in range(2):
                with self.assertRaisesRegex(ValueError, "recipe failed"):
                    with recipe_logging():
                        self.assertEqual(len(logger.handlers), 1)
                        self.assertNotIn(handler, logger.handlers)
                        self.assertEqual(logger.level, logging.DEBUG)
                        raise ValueError("recipe failed")
                self.assertEqual(logger.handlers, [handler])
                self.assertEqual(logger.level, logging.WARNING)
                self.assertTrue(logger.propagate)
        finally:
            logger.handlers, level, logger.propagate = original
            logger.setLevel(level)
            handler.close()
