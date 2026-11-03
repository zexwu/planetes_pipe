"""flat.py: trace detection, metrology trace and calibration checks."""


import tempfile
import unittest
from pathlib import Path

import numpy as np
from astropy.io import fits
from fixtures import RawMetadata, write_raw

from planetes_pipe.dark import Bad, Dark
from planetes_pipe.files import read_raw, write_shutters
from planetes_pipe.flat import build_flat, make_flat
from planetes_pipe.instrument import DetectorSetup, layout


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
                        make_flat(
                            [read_raw(path, "FLAT_RAW", hdu=0) for path in paths.values()],
                            Dark(np.zeros(shape)),
                            Bad(np.zeros((3, 5), dtype=np.int16)),
                        )

    def test_metrology_selection_follows_trace_order_after_translation(self):
        height, width = 1200, 64
        y, x = np.indices((height, width))
        setup = DetectorSetup(exposure_s=0.1, width=width, height=height,
                              origin_x=0, origin_y=0, binning_x=1, binning_y=1,
                              detector="synthetic")
        dark = Dark(np.zeros((height, width)), setup=setup)
        bad = Bad(np.zeros((height, width), dtype=np.int16))
        for shift in (0, 7):
            centers = 40 + 44 * np.arange(25) + shift
            image = sum(1000 * np.exp(-0.5 * ((y - center) / 2)**2)
                        * np.exp(-0.5 * ((x - 32) / 12)**2) for center in centers)
            with tempfile.TemporaryDirectory() as directory:
                paths = {}
                for telescope in layout.telescopes:
                    paths[telescope] = Path(directory) / f"{telescope}.fits"
                    physical = RawMetadata("FLAT_RAW", tuple(str(t) == telescope for t in range(1, 5)), setup)
                    write_raw(paths[telescope], image[None], physical)
                flats_raw = [read_raw(path, "FLAT_RAW") for path in paths.values()]
                flat = make_flat(flats_raw, dark, bad, metrology_trace=True)
            expected = np.delete(centers, 12)
            np.testing.assert_allclose(flat.trace_y_pixels[:, 32], expected, atol=0.03)
            self.assertEqual(flat.profile_x_pixels.shape[0], 24)
            self.assertEqual(flat.profile_y_pixels.shape[0], 24)
            self.assertEqual(flat.profile_sigma_pixels.shape[0], 24)


if __name__ == "__main__":
    unittest.main()
