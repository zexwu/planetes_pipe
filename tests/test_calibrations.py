"""Bench P2VM inputs and the independently selected phase-reference dark."""

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
from astropy.io import fits
from p2vm_fixture import write_recipe_inputs

from planetes.extraction import prepare_extraction
from planetes.io.fits import read_product
from planetes.p2vm import (
    make_p2vm,
    extract_p2vm_spectra,
)
from planetes.products import DarkImage


class CalibrationTests(unittest.TestCase):
    def test_p2vm_file_interface_accepts_explicit_bench_identities(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = write_recipe_inputs(root / "versioned")
            flat, wave = read_product(paths["FLAT"]), read_product(paths["WAVE"])
            dark = read_product(paths["DARK"])
            options = dict(wavelength_channels=15, phase_correction=False)
            expected = make_p2vm(
                flat,
                wave,
                dark,
                telescope_flats=paths["FLAT_RAW"],
                baseline_cubes=paths["P2VM_RAW"],
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
            self.assertEqual(actual.baseline_telescopes, expected.baseline_telescopes)

    def test_phase_dark_is_independent_from_main_dark(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = write_recipe_inputs(root / "inputs")
            reference = inputs["PHASE_REFERENCE"]
            with fits.open(reference, mode="update") as hdus:
                hdus[1].data += 7
                hdus[1].add_checksum()
            flat, wave = (
                read_product(inputs["FLAT"]),
                read_product(inputs["WAVE"]),
            )
            main = read_product(inputs["DARK"])
            phase = replace(
                main, mean_adu=np.full(main.mean_adu.shape, 7, dtype=np.float32)
            )
            spectra = extract_p2vm_spectra(
                flat,
                wave,
                main,
                telescope_flats=inputs["FLAT_RAW"],
                baseline_cubes=inputs["P2VM_RAW"],
                wavelength_m=np.linspace(1.15e-6, 1.30e-6, 151),
                phase_reference=reference,
                phase_dark=phase,
            )
            raw = np.array(fits.getdata(reference), dtype=np.float32)
            corrected = raw - 7
            prepared = prepare_extraction(
                flat, wave, np.linspace(1.15e-6, 1.30e-6, 151)
            )
            expected = prepared.extract(
                corrected, DarkImage(np.zeros_like(phase.mean_adu))
            )
            np.testing.assert_array_equal(spectra.phase_reference, expected)
