"""Automatic common-band sampling and its persisted trace limits."""
import unittest

import numpy as np

from p2vm_fixture import calibration_arrays
from planetes_pipe import io
from planetes_pipe.processing.p2vm import wavelength_grid


class WavelengthGridTests(unittest.TestCase):
    def test_intersection_at_native_sampling_and_fits_roundtrip(self):
        *_, flat, wave = calibration_arrays(n_wave=11)
        flat.trace_x_range[:] = [1, 9]
        flat.trace_x_range[0] = [3, 7]
        wave.wavelength_map_m[:] = np.arange(11) * 1e-9 + 1e-6
        wave.wavelength_map_m[0] += 0.2e-9
        with io.flat.encode(flat) as hdus:
            np.testing.assert_array_equal(hdus['PROFILE_PARAMS'].data['XRANGE'][0], [4, 8])
            restored = io.flat.decode(hdus)
        np.testing.assert_array_equal(restored.trace_x_range, flat.trace_x_range)
        expected = np.arange(4, 8) * 1e-9 + 1e-6
        np.testing.assert_allclose(wavelength_grid(restored, wave), expected)
        wave.wavelength_map_m[:] = wave.wavelength_map_m[:, ::-1]
        np.testing.assert_allclose(wavelength_grid(restored, wave), expected)

    def test_missing_limits_do_not_use_extrapolated_band(self):
        *_, flat, wave = calibration_arrays(n_wave=11)
        flat.trace_x_range = None
        with self.assertRaisesRegex(ValueError, 'regenerate'):
            wavelength_grid(flat, wave)

    def test_nonoverlapping_bands(self):
        *_, flat, wave = calibration_arrays(n_wave=11)
        flat.trace_x_range[0] = [0, 2]
        flat.trace_x_range[1] = [8, 10]
        with self.assertRaisesRegex(ValueError, 'no common'):
            wavelength_grid(flat, wave)
