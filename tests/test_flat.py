"""flat.py: trace detection, metrology trace and calibration checks."""


import tempfile
import unittest
from pathlib import Path

import numpy as np
from astropy.io import fits
from fixtures import RawMetadata, cards, write_raw

from planetes_pipe.instrument import layout
from planetes_pipe.io.raw import read_raws, write_shutters
from planetes_pipe.steps.dark import Bad, Dark
from planetes_pipe.steps.flat import build_flat, make_flat, trace_spectrum


class FlatTests(unittest.TestCase):
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
                rtol=0, atol=0.1,  # tracing averages columns: 0.07 px across this steep band edge
            )

    def test_a_tilted_trace_keeps_its_range_after_leaving_the_starting_strip(self):
        # Output 23 of 2026-11-05: 21 px of tilt carried it out of a strip fixed at its
        # peak, and its fitted range stopped there though it stayed bright.
        height, width = 120, 400
        y, x = np.indices((height, width))
        image = np.random.default_rng(1).normal(0, 0.05, (height, width))
        trace = 80 - 0.15 * np.arange(width)
        image += 100 * np.exp(-0.5 * ((y - trace[None]) / 2.0) ** 2)
        image += np.where(x > 300, 100, 0) * np.exp(-0.5 * ((y - trace[None]) / 2.0) ** 2)
        flat = build_flat([image] * 4, Bad(np.zeros(image.shape, dtype=np.int16)), n_outputs=1)
        self.assertLess(flat.trace_x_range[0, 0], 50)

    def test_faint_telescope_traces_are_found_beside_bright_ones(self):
        # 2026-11-05 17:02: S4 about 19 times S1, so the traces lit only by the faint
        # telescopes lay below the profile's 75th percentile, once the detection cut.
        height, width = 25 * 34, 120
        y, x = np.indices((height, width))
        image = np.random.default_rng(3).normal(0, 0.05, (height, width))
        centres = 17 + 34 * np.arange(25)
        for n, centre in enumerate(centres):
            image += (1000 if n >= 9 else 40) * np.exp(-0.5 * ((y - centre) / 4.0) ** 2)
        flat = build_flat([image] * 4, Bad(np.zeros(image.shape, dtype=np.int16)),
                          n_outputs=25)
        np.testing.assert_allclose(flat.trace_y_pixels[:, 60], centres, atol=0.05)

    def test_trace_is_followed_from_a_start_three_pixels_off(self):
        # A lopsided profile's peak, which gives the start, can sit 2-3 pixels from
        # its Gaussian centre; tracking must still accept the columns (2026-11-05).
        height, width = 40, 200
        y, x = np.indices((height, width))
        image = 50 * np.exp(-0.5 * ((y - 20.0) / 2.5) ** 2)
        bad = np.zeros(image.shape, dtype=np.int16)
        centers, errors, sigma = trace_spectrum(image, bad, ref_x=100, ref_y=17, range_y=21)
        self.assertTrue(np.isfinite(sigma).all())
        np.testing.assert_allclose(centers, 20.0, atol=0.01)

    def test_flat_rejects_dark_shape_without_det_cards(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
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
                        flats = read_raws(paths.values(), "FLAT_RAW", layout.telescopes, hdu=0)
                        make_flat(flats, Dark(np.zeros(shape)), Bad(np.zeros((3, 5), dtype=np.int16)))

    def test_metrology_selection_follows_trace_order_after_translation(self):
        height, width = 1200, 64
        y, x = np.indices((height, width))
        detector = cards(DIT=0.1, WIN_NX=width, WIN_NY=height, WIN_STRX=1, WIN_STRY=1,
                         WIN_BINX=1, WIN_BINY=1, DET_ID="synthetic")
        dark = Dark(np.zeros((height, width)), detector=detector)
        bad = Bad(np.zeros((height, width), dtype=np.int16))
        for shift in (0, 7):
            centers = 40 + 44 * np.arange(25) + shift
            image = sum(1000 * np.exp(-0.5 * ((y - center) / 2)**2)
                        * np.exp(-0.5 * ((x - 32) / 12)**2) for center in centers)
            with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
                paths = {}
                for telescope in layout.telescopes:
                    paths[telescope] = Path(directory) / f"{telescope}.fits"
                    physical = RawMetadata("FLAT_RAW", tuple(str(t) == telescope for t in range(1, 5)), detector)
                    write_raw(paths[telescope], image[None], physical)
                flats = read_raws(paths.values(), "FLAT_RAW", layout.telescopes)
                flat = make_flat(flats, dark, bad, metrology_trace=True)
            expected = np.delete(centers, 12)
            np.testing.assert_allclose(flat.trace_y_pixels[:, 32], expected, atol=0.03)
            self.assertEqual(flat.profile_x_pixels.shape[0], 24)
            self.assertEqual(flat.profile_y_pixels.shape[0], 24)
            self.assertEqual(flat.profile_sigma_pixels.shape[0], 24)


if __name__ == "__main__":
    unittest.main()
