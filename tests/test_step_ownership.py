"""Pipeline-owned metrology selection, OPD-guided phase and step QC."""
from contextlib import ExitStack
from planetes_pipe.io import raw as raw_io
from pathlib import Path
import tempfile
import unittest

import numpy as np
from p2vm_fixture import write_raw

from planetes_pipe.processing.dark import measure_quality as dark_quality
from planetes_pipe.processing.flat import make_flat
from planetes_pipe.model.instrument import layout, DetectorSetup
from planetes_pipe.model.raw import RawMetadata
from planetes_pipe.processing.p2vm import unwrap_scan_phase
from planetes_pipe.model.products import Bad, Dark, Wave
from planetes_pipe.model.quality import DarkQC, BadQC, WaveQC
from planetes_pipe.processing.wave import measure_quality as wave_quality


class StepOwnershipTests(unittest.TestCase):
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
                with ExitStack() as raw_files:
                    flats_raw = {
                        key: raw_files.enter_context(
                            raw_io.read(path, category="FLAT_RAW")
                        )
                        for key, path in (paths).items()
                    }
                    flat = make_flat(flats_raw, dark, bad, metrology_trace=True)
            expected = np.delete(centers, 12)
            np.testing.assert_allclose(flat.trace_y_pixels[:, 32], expected, atol=0.03)
            self.assertEqual(flat.profile_x_pixels.shape[0], 24)
            self.assertEqual(flat.profile_y_pixels.shape[0], 24)
            self.assertEqual(flat.profile_sigma_pixels.shape[0], 24)

    def test_dl_guide_recovers_sparse_round_trip_without_changing_wrapped_phase(self):
        wavelength = np.linspace(1.22e-6, 1.28e-6, 31)
        position = np.interp(np.arange(33), [0, 8, 16, 24, 32], [0, 30e-6, 0, -30e-6, 0])
        for gain in (-1, 1):
            opd = gain * position
            truth = 2 * np.pi * opd[:, None] / wavelength + 0.3
            measured = np.unwrap(np.angle(np.exp(1j * truth)), axis=0)
            recovered = unwrap_scan_phase(measured, opd, wavelength)
            anchor = len(opd) // 2
            np.testing.assert_allclose(recovered - recovered[anchor], truth - truth[anchor], atol=1e-12)
            np.testing.assert_allclose(np.exp(1j * recovered), np.exp(1j * measured), atol=1e-12)

    def test_qc_definitions_and_step_statistics_preserve_units_and_missing_values(self):
        dark = Dark(np.array([[1., 3.]]))
        self.assertEqual(dark_quality(dark), DarkQC(median_adu=2.))
        bad = Bad(np.array([[0, 1, 2, 4, 7]], dtype=np.int16))
        quality = dark_quality(bad)
        self.assertEqual(quality, BadQC(4, 3, 2))
        wave = Wave(np.array([[1., 2., 3.], [1.5, 2., 2.5]]) * 1e-6,
                    fit_residual_m=np.array([-1., 1.]) * 1e-9)
        quality = wave_quality(wave)
        self.assertEqual(quality.min_m, 1.5e-6)
        np.testing.assert_allclose(quality.max_m, 2.5e-6, rtol=1e-15, atol=0)
        self.assertAlmostEqual(quality.residual_std_m, np.sqrt(2) * 1e-9)
        self.assertIsInstance(quality, WaveQC)
        self.assertEqual(quality.residual_count, 2)


if __name__ == "__main__":
    unittest.main()
