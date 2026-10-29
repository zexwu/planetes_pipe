"""Raw calibration recovery, physical input checks and phase-reference darks."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
from astropy.io import fits
from p2vm_fixture import write_recipe_inputs

from planetes_pipe.dark import make_dark_products
from planetes_pipe.extraction import extract_spectra_from_file, prepare_extraction
from planetes_pipe.flat import make_flat
from planetes_pipe.instrument import BASELINES, TELESCOPES
from planetes_pipe.io.fits import read_product, write_product
from planetes_pipe.p2vm import make_p2vm
from planetes_pipe.products import Bad, Dark, WavelengthReference
from planetes_pipe.wavelength import make_wave


class CalibrationTests(unittest.TestCase):
    def test_flat_rejects_dark_shape_without_setup(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = {}
            for telescope in TELESCOPES:
                path = Path(directory) / f"tel{telescope}.fits"
                fits.PrimaryHDU(np.ones((2, 3, 5))).writeto(path)
                paths[telescope] = path
            for shape in ((1, 5), (3, 1), (3, 4)):
                with self.subTest(shape=shape):
                    with self.assertRaisesRegex(ValueError, "DARK detector shapes"):
                        make_flat(
                            paths,
                            Dark(np.zeros(shape)),
                            Bad(np.zeros((3, 5), dtype=np.int16)),
                            hdu=0,
                        )

    def test_raw_flat_and_lamp_recover_known_traces_and_wavelengths(self):
        # Independent detector model: 24 Gaussian traces and a known dispersion.
        height, width = 768, 256
        y, x = np.indices((height, width))
        centers = np.arange(16, height, 32)
        profiles = np.exp(-0.5 * ((y[:, :1] - centers[:, None, None]) / 3) ** 2)
        flat_image = 1000 * profiles.sum(axis=0) * np.ones((1, width))
        dark_image = 20 + 0.01 * x + 0.002 * y
        dark_image[16, 100] += 100  # Hot pixel inside the first extraction trace.
        # A separate structured lamp dark makes using the main dark detectable
        # in the recovered dispersion, rather than just changing a continuum.
        lamp_dark_image = dark_image + 200 + 100 * np.sin(x / 9)
        line_pixels = np.array([12, 29, 48, 66, 87, 108, 132, 159, 181, 205, 230, 246])
        reference = WavelengthReference(
            1.05e-6 + line_pixels * 1e-9, ("synthetic",) * len(line_pixels)
        )
        shifts = (centers - centers.mean()) / 2000
        lamp_image = np.zeros((height, width))
        for profile, shift in zip(profiles, shifts, strict=True):
            spectrum = 10 + sum(
                3000 * np.exp(-0.5 * ((np.arange(width) - pixel - shift) / 0.8) ** 2)
                for pixel in line_pixels
            )
            lamp_image += profile * spectrum

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, image in (("dark", dark_image), ("lamp_dark", lamp_dark_image)):
                fits.PrimaryHDU(np.stack([image - 1, image + 1])).writeto(
                    root / f"{name}.fits"
                )
            dark, bad = make_dark_products(root / "dark.fits", hdu=0)
            lamp_dark, _ = make_dark_products(root / "lamp_dark.fits", hdu=0)
            self.assertEqual(np.count_nonzero(bad.bad_pixel_flags), 1)
            self.assertEqual(bad.bad_pixel_flags[16, 100], 1)
            paths = {}
            for telescope, scale in zip(TELESCOPES, (0.8, 0.9, 1.1, 1.2), strict=True):
                path = root / f"tel{telescope}.fits"
                frame = scale * flat_image + dark_image
                fits.PrimaryHDU(np.stack([frame, frame])).writeto(path)
                paths[telescope] = path
            flat = make_flat(paths, dark, bad, hdu=0)
            np.testing.assert_allclose(flat.flat_map_adu, flat_image, atol=1e-10)
            np.testing.assert_allclose(
                flat.trace_y_pixels,
                np.broadcast_to(centers[:, None], (24, width)),
                rtol=0,
                atol=0.01,
            )
            self.assertFalse(
                ((flat.profile_x_pixels == 100) & (flat.profile_y_pixels == 16)).any()
            )
            # Exercise the same calibration-file boundary used between recipes.
            write_product(root / "flat.fits", flat)
            flat = read_product(root / "flat.fits")
            frame = lamp_image + lamp_dark_image
            fits.PrimaryHDU(np.stack([frame, frame])).writeto(root / "lamp.fits")
            wave = make_wave(
                root / "lamp.fits", lamp_dark, flat, reference=reference, hdu=0
            )
            truth_m = 1.05e-6 + (np.arange(width)[None] - shifts[:, None]) * 1e-9
            # 0.002 pixel at 1 nm/pixel allows centroid/fit rounding, not a
            # scientific wavelength-accuracy requirement for real observations.
            np.testing.assert_allclose(
                wave.wavelength_map_m, truth_m, rtol=0, atol=2e-12
            )
            self.assertGreaterEqual(len(wave.fit_residual_m), 24 * 6)
            write_product(root / "wave.fits", wave)
            loaded = read_product(root / "wave.fits")
            np.testing.assert_allclose(
                loaded.wavelength_map_m, truth_m, rtol=0, atol=2e-12
            )

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
                hdu=0,
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
