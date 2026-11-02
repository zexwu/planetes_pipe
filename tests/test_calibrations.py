"""Raw calibration recovery, physical input checks and phase-reference darks."""
from planetes_pipe.io.common import write_shutters
from planetes_pipe.io import dark as dark_io
from planetes_pipe.io import flat as flat_io
from planetes_pipe.io import wave as wave_io
from contextlib import ExitStack
from planetes_pipe.io import raw as raw_io

import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
from astropy.io import fits
from p2vm_fixture import write_recipe_inputs

from planetes_pipe.processing.dark import make_dark_products
from planetes_pipe.processing.extraction import extract_spectra, prepare_extraction
from planetes_pipe.processing.flat import build_flat, make_flat
from planetes_pipe.model.instrument import layout
from planetes_pipe.processing.p2vm import make_p2vm
from planetes_pipe.model.products import Bad, Dark
from planetes_pipe.processing.wave import make_wave


class CalibrationTests(unittest.TestCase):
    def test_flat_traces_follow_separate_off_center_illuminated_bands(self):
        # Known curved traces, with each output illuminated away from detector
        # centre. Dark columns must not determine the fitted trace geometry.
        rng = np.random.default_rng(314)
        height, width = 128, 512
        y, x = np.indices((height, width))
        image = rng.normal(0, 0.2, (height, width))
        truth = []
        band_centres = (120, 160, 190, 220)
        for centre_y, centre_x in zip((16, 48, 80, 112), band_centres):
            dx = np.arange(width) - centre_x
            trace = centre_y + 0.02 * dx + 0.0001 * dx**2
            envelope = 1000 * np.exp(-0.5 * ((x - centre_x) / 18)**2)
            image += envelope * np.exp(-0.5 * ((y - trace[None]) / 2)**2)
            truth.append(trace)
        bad = Bad(np.zeros(image.shape, dtype=np.int16))
        flat = build_flat([image] * 4, bad, n_outputs=4)
        for output, centre_x in enumerate(band_centres):
            region = slice(centre_x - 30, centre_x + 31)
            np.testing.assert_allclose(
                flat.trace_y_pixels[output, region], truth[output][region],
                rtol=0, atol=0.03,
            )

    def test_flat_rejects_dark_shape_without_setup(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = {}
            for telescope in layout.telescopes:
                path = Path(directory) / f"tel{telescope}.fits"
                header = fits.Header()
                write_shutters(header, tuple(str(t) == telescope for t in range(1, 5)))
                fits.PrimaryHDU(np.ones((2, 3, 5)), header=header).writeto(path)
                paths[telescope] = path
            for shape in ((1, 5), (3, 1), (3, 4)):
                with self.subTest(shape=shape):
                    with self.assertRaisesRegex(ValueError, "DARK detector shapes"):
                        with ExitStack() as raw_files:
                            flats_raw = [raw_files.enter_context(raw_io.read(path, category='FLAT_RAW', hdu=0)) for key, path in paths.items()]
                            make_flat(
                                flats_raw,
                                Dark(np.zeros(shape)),
                                Bad(np.zeros((3, 5), dtype=np.int16)),
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
        reference = 1.05e-6 + line_pixels * 1e-9
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
            with raw_io.read(root / 'dark.fits', category="DARK_RAW", hdu=0) as raw:
                dark, bad = make_dark_products(raw)
            with raw_io.read(root / 'lamp_dark.fits', category="DARK_RAW", hdu=0) as raw:
                lamp_dark, _ = make_dark_products(raw)
            self.assertEqual(np.count_nonzero(bad.bad_pixel_flags), 1)
            self.assertEqual(bad.bad_pixel_flags[16, 100], 1)
            paths = {}
            for telescope, scale in zip(layout.telescopes, (0.8, 0.9, 1.1, 1.2), strict=True):
                path = root / f"tel{telescope}.fits"
                frame = scale * flat_image + dark_image
                header = fits.Header()
                write_shutters(header, tuple(str(t) == telescope for t in range(1, 5)))
                fits.PrimaryHDU(np.stack([frame, frame]), header=header).writeto(path)
                paths[telescope] = path
            with ExitStack() as raw_files:
                flats_raw = [raw_files.enter_context(raw_io.read(path, category='FLAT_RAW', hdu=0)) for key, path in paths.items()]
                flat = make_flat(flats_raw, dark, bad)
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
            flat_io.write(root / "flat.fits", flat)
            flat = flat_io.read(root / "flat.fits")
            frame = lamp_image + lamp_dark_image
            fits.PrimaryHDU(np.stack([frame, frame])).writeto(root / "lamp.fits")
            with raw_io.read(
                root / "lamp.fits", category="WAVELAMP_RAW", hdu=0
            ) as raw_input, patch("planetes_pipe.processing.wave.REFERENCE_WAVELENGTH_M", reference):
                wave = make_wave(raw_input, lamp_dark, flat)
            truth_m = 1.05e-6 + (np.arange(width)[None] - shifts[:, None]) * 1e-9
            # 0.002 pixel at 1 nm/pixel allows centroid/fit rounding, not a
            # scientific wavelength-accuracy requirement for real observations.
            np.testing.assert_allclose(
                wave.wavelength_map_m, truth_m, rtol=0, atol=2e-12
            )
            self.assertGreaterEqual(len(wave.fit_residual_m), 24 * 6)
            wave_io.write(root / "wave.fits", wave)
            loaded = wave_io.read(root / "wave.fits")
            np.testing.assert_allclose(
                loaded.wavelength_map_m, truth_m, rtol=0, atol=2e-12
            )

    def test_p2vm_uses_shutters_with_reversed_lists_and_arbitrary_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = write_recipe_inputs(root / "versioned")
            flat, wave = flat_io.read(paths["FLAT"]), wave_io.read(paths["WAVE"])
            dark = dark_io.read(paths["DARK"])
            with self.assertRaisesRegex(ValueError, "outside calibration"):
                prepare_extraction(flat, wave, np.linspace(1.14e-6, 1.30e-6, 15))
            with ExitStack() as raw_files:
                telescope_flats_raw = [raw_files.enter_context(raw_io.read(path, category='FLAT_RAW')) for key, path in dict(zip(layout.telescopes, paths['FLAT_RAW'], strict=True)).items()]
                baseline_cubes_raw = [raw_files.enter_context(raw_io.read(path, category='P2VM_RAW')) for key, path in dict(zip(layout.baselines, paths['P2VM_RAW'], strict=True)).items()]
                expected = make_p2vm(
                    flat,
                    wave,
                    dark,
                    telescope_flats=telescope_flats_raw,
                    baseline_cubes=baseline_cubes_raw,
                )

            def bench_files(role, identities):
                result = {}
                for identity, path in zip(identities, paths[role], strict=True):
                    target = root / f"unrelated-{len(result)}-{role}.fits"
                    with fits.open(path) as hdus:
                        fits.PrimaryHDU(hdus[1].data, header=hdus[0].header).writeto(target)
                    result[identity] = target
                return result

            with ExitStack() as raw_files:
                telescope_flats_raw = [raw_files.enter_context(raw_io.read(path, category='FLAT_RAW', hdu=0)) for path in reversed(list(bench_files('FLAT_RAW', ('4', '3', '2', '1')).values()))]
                baseline_cubes_raw = [raw_files.enter_context(raw_io.read(path, category='P2VM_RAW', hdu=0)) for path in reversed(list(bench_files('P2VM_RAW', ('43', '42', '41', '32', '31', '21')).values()))]
                actual = make_p2vm(
                    flat,
                    wave,
                    dark,
                    telescope_flats=telescope_flats_raw,
                    baseline_cubes=baseline_cubes_raw,
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
            flat, wave = (flat_io.read(inputs["FLAT"]), wave_io.read(inputs["WAVE"]))
            main = dark_io.read(inputs["DARK"])
            phase = Dark(
                np.full(main.mean_adu.shape, 7, dtype=np.float32), setup=main.setup
            )
            raw = np.array(fits.getdata(reference), dtype=np.float32)
            corrected = raw - 7
            prepared = prepare_extraction(
                flat, wave, np.linspace(1.15e-6, 1.30e-6, 151)
            )
            with raw_io.read(reference, category="SINGLE_SCI_RAW") as raw_input:
                reference_spectra = extract_spectra(raw_input, prepared, phase)
            expected = prepared.extract(corrected, Dark(np.zeros_like(phase.mean_adu)))
            np.testing.assert_array_equal(reference_spectra, expected)
            self.assertFalse(
                np.array_equal(reference_spectra, prepared.extract(raw, main))
            )
