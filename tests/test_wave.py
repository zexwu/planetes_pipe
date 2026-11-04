"""wave.py: wavelength solution from raw flat and lamp, line matching."""


import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from astropy.io import fits

from planetes_pipe.instrument import layout
from planetes_pipe.io.raw import read_raw, read_raws, write_shutters
from planetes_pipe.steps.dark import make_dark_products
from planetes_pipe.steps.flat import make_flat, read_flat, write_flat
from planetes_pipe.steps.wave import make_wave, match_peaks_grid, read_wave, write_wave


class WaveTests(unittest.TestCase):
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

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            root = Path(directory)
            for name, image in (("dark", dark_image), ("lamp_dark", lamp_dark_image)):
                fits.PrimaryHDU(np.stack([image - 1, image + 1])).writeto(
                    root / f"{name}.fits"
                )
            dark, bad = make_dark_products(read_raw(root / "dark.fits", "DARK_RAW", hdu=0))
            lamp_dark, _ = make_dark_products(read_raw(root / "lamp_dark.fits", "DARK_RAW", hdu=0))
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
            flats = read_raws(paths.values(), "FLAT_RAW", layout.telescopes, hdu=0)
            flat = make_flat(flats, dark, bad)
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
            write_flat(root / "flat.fits", flat)
            flat = read_flat(root / "flat.fits")
            frame = lamp_image + lamp_dark_image
            fits.PrimaryHDU(np.stack([frame, frame])).writeto(root / "lamp.fits")
            with patch("planetes_pipe.steps.wave.REFERENCE_WAVELENGTH_M", reference):
                wave = make_wave(read_raw(root / "lamp.fits", "WAVELAMP_RAW", hdu=0),
                                 lamp_dark, flat)
            truth_m = 1.05e-6 + (np.arange(width)[None] - shifts[:, None]) * 1e-9
            # 0.002 pixel at 1 nm/pixel allows centroid/fit rounding, not a
            # scientific wavelength-accuracy requirement for real observations.
            np.testing.assert_allclose(
                wave.wavelength_map_m, truth_m, rtol=0, atol=2e-12
            )
            self.assertGreaterEqual(len(wave.fit_residual_m), 24 * 6)
            write_wave(root / "wave.fits", wave)
            loaded = read_wave(root / "wave.fits")
            np.testing.assert_allclose(
                loaded.wavelength_map_m, truth_m, rtol=0, atol=2e-12
            )

    def test_grid_line_matching_is_one_to_one(self) -> None:
        theoretical = np.array([0.0, 0.15, 1.0, 2.0])
        observed = np.array([0.05, 1.0, 2.0])
        matches, _ = match_peaks_grid(
            theoretical,
            observed,
            tol=0.2,
            min_inliers=3,
            offset_limit=0.5,
            slope_bounds=(0.8, 1.2),
            curve_limit=0.1,
            steps_per_unit=2,
        )
        self.assertEqual(len(matches), len(np.unique(matches[:, 1])))


if __name__ == "__main__":
    unittest.main()
