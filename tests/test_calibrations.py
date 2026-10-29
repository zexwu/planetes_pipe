"""Bench P2VM inputs and the independently selected phase-reference dark."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
from astropy.io import fits
from p2vm_fixture import write_recipe_inputs

from planetes.extraction import extract_spectra_from_file, prepare_extraction
from planetes.io.fits import read_product
from planetes.instrument import BASELINES, TELESCOPES
from planetes.p2vm import make_p2vm
from planetes.products import Dark


class CalibrationTests(unittest.TestCase):
    def test_p2vm_file_interface_accepts_explicit_bench_identities(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = write_recipe_inputs(root / "versioned")
            flat, wave = read_product(paths["FLAT"]), read_product(paths["WAVE"])
            dark = read_product(paths["DARK"])
            with self.assertRaisesRegex(ValueError, "outside calibration"):
                prepare_extraction(flat, wave, np.linspace(1.14e-6, 1.30e-6, 15))
            options = dict(wavelength_channels=15)
            expected = make_p2vm(
                flat,
                wave,
                dark,
                telescope_flats=dict(zip(TELESCOPES, paths["FLAT_RAW"], strict=True)),
                baseline_cubes=dict(zip(BASELINES, paths["P2VM_RAW"], strict=True)),
                **options,
            )

            def bench_files(role, identities):
                result = {}
                for identity, path in zip(identities, paths[role], strict=True):
                    target = root / f"{role}_{identity}.fits"
                    with fits.open(path) as hdus:
                        fits.PrimaryHDU(hdus[1].data).writeto(target)
                    result[identity] = target
                return result

            actual = make_p2vm(
                flat,
                wave,
                dark,
                telescope_flats=bench_files("FLAT_RAW", ("4", "3", "2", "1")),
                baseline_cubes=bench_files(
                    "P2VM_RAW", ("43", "42", "41", "32", "31", "21")
                ),
                **options,
            )
            np.testing.assert_array_equal(actual.matrix, expected.matrix)
            self.assertEqual(actual.baseline_outputs, expected.baseline_outputs)

    def test_phase_dark_is_independent_from_main_dark(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = write_recipe_inputs(root / "inputs")
            reference = inputs["PHASE_REFERENCE"]
            with fits.open(reference, mode="update") as hdus:
                hdus[1].data += 7
                hdus[1].add_checksum()
            flat, wave = (read_product(inputs["FLAT"]), read_product(inputs["WAVE"]))
            main = read_product(inputs["DARK"])
            phase = Dark(
                np.full(main.mean_adu.shape, 7, dtype=np.float32), setup=main.setup
            )
            raw = np.array(fits.getdata(reference), dtype=np.float32)
            corrected = raw - 7
            prepared = prepare_extraction(
                flat, wave, np.linspace(1.15e-6, 1.30e-6, 151)
            )
            reference_spectra = extract_spectra_from_file(reference, prepared, phase)
            expected = prepared.extract(corrected, Dark(np.zeros_like(phase.mean_adu)))
            np.testing.assert_array_equal(reference_spectra, expected)
            self.assertFalse(
                np.array_equal(reference_spectra, prepared.extract(raw, main))
            )
