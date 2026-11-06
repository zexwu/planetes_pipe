"""science.py: visibility normalization and exact P2VMRED round trips."""


import tempfile
import unittest
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
from astropy.io import fits
from fixtures import calibration_arrays, calibration_spectra

from planetes_pipe.instrument import layout
from planetes_pipe.steps.extraction import stacked_group_delay
from planetes_pipe.steps.flat import Flat, read_flat
from planetes_pipe.steps.p2vm import P2VM, fit_p2vm, read_p2vm
from planetes_pipe.steps.science import (
    normalize_coherent_flux,
    read_reduced,
    reduce_spectra,
    write_reduced,
)
from planetes_pipe.steps.wave import Wave, read_wave

READERS = {Flat: read_flat, Wave: read_wave, P2VM: read_p2vm}


class StackedGroupDelayTests(unittest.TestCase):
    WAVELENGTH = np.linspace(1.2e-6, 1.27e-6, 200)

    def frames(self, delay_m, amplitude):
        rng = np.random.default_rng(2)
        phases = rng.uniform(0, 2 * np.pi, (32, 1))  # the fringe moves between frames
        noise = rng.normal(size=(32, 200)) + 1j * rng.normal(size=(32, 200))
        return amplitude * np.exp(1j * phases + 2j * np.pi * delay_m / self.WAVELENGTH) + noise

    def test_finds_a_delay_through_fringe_motion_and_noise(self):
        delay, peak = stacked_group_delay(self.frames(37.2e-6, 3.0), self.WAVELENGTH)
        self.assertAlmostEqual(delay * 1e6, 37.2, delta=0.1)
        self.assertGreater(peak, 20)

    def test_no_fringes_give_no_clear_peak(self):
        _, peak = stacked_group_delay(self.frames(37.2e-6, 0.0), self.WAVELENGTH)
        self.assertLess(peak, 5)


class ReducedProductTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spectra = calibration_spectra(n_wave=15)
        p2vm = fit_p2vm(spectra)
        science = calibration_arrays(n_wave=15)[2]
        science[:, 0] = 0
        cls.product = reduce_spectra(science, p2vm)

    def test_tables_are_named_as_gravity_with_estimated_times(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temporary:
            path = Path(temporary) / "science.fits"
            write_reduced(path, self.product, start_mjd=60000.0, dit_s=0.02)
            with fits.open(path) as hdus:
                self.assertEqual(hdus["OI_VIS"].header["INSNAME"], "PLANETES_SC")
                self.assertIn("FLUX", hdus["OI_FLUX"].columns.names)
                self.assertIn("EFF_WAVE", hdus["OI_WAVELENGTH"].columns.names)
                vis = hdus["OI_VIS"].data
                n_baselines = len(layout.baselines)
                np.testing.assert_allclose(vis["TIME"][n_baselines], 0.02e6)
                np.testing.assert_allclose(vis["MJD"][n_baselines], 60000.0 + 0.02 / 86400)
                np.testing.assert_allclose(vis["INT_TIME"], 0.02)

    def test_times_are_nan_when_the_raw_header_lacks_them(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temporary:
            path = Path(temporary) / "science.fits"
            write_reduced(path, self.product)
            with fits.open(path) as hdus:
                self.assertTrue(np.isnan(hdus["OI_FLUX"].data["MJD"]).all())

    def test_products_with_the_old_table_names_are_still_read(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temporary:
            new, old = Path(temporary) / "new.fits", Path(temporary) / "old.fits"
            write_reduced(new, self.product)
            with fits.open(new) as hdus:
                hdus["OI_WAVELENGTH"].columns.change_name("EFF_WAVE", "WAVELENGTH")
                hdus["OI_WAVELENGTH"].name = "PLANETES_WAVELENGTH"
                hdus["OI_FLUX"].columns.change_name("FLUX", "FLUXDATA")
                hdus["OI_FLUX"].name = "PLANETES_FLUX"
                hdus["OI_VIS"].name = "PLANETES_VIS"
                hdus.writeto(old)
            np.testing.assert_array_equal(read_reduced(old).photometric_flux_adu,
                                          self.product.photometric_flux_adu)

    def test_complex_flux_and_flags_round_trip_exactly(self):
        coherent = self.product.coherent_flux_adu.copy()
        coherent[0, 0, 0] = complex(-0.0, -0.0)
        coherent[0, 0, 1] = complex(np.nan, np.nan)
        product = replace(self.product, coherent_flux_adu=coherent)
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temporary:
            path = Path(temporary) / "science.fits"
            write_reduced(path, product)
            loaded = read_reduced(path)
            for field in fields(product):
                before, after = (
                    getattr(product, field.name),
                    getattr(loaded, field.name),
                )
                if isinstance(before, np.ndarray):
                    np.testing.assert_array_equal(before, after)  # Stored as computed.
                else:
                    self.assertEqual(before, after)
            np.testing.assert_array_equal(
                coherent.view(np.uint64),  # Bit-exact, including -0.0 and NaN.
                loaded.coherent_flux_adu.view(np.uint64),
            )
            self.assertTrue(loaded.flags[0].all())
            self.assertEqual(loaded.flags.dtype, bool)


class ReduceTests(unittest.TestCase):
    def test_visibility_normalization_flags_invalid_flux_products(self) -> None:
        coherent_flux = np.tile(np.array([[[6 + 8j, 1 + 0j]]]), (1, 6, 1))
        flux = np.array([[[4.0, -1.0], [25.0, 2.0], [25.0, 2.0], [25.0, 2.0]]])

        visamp, flag = normalize_coherent_flux(coherent_flux, flux)

        self.assertEqual(visamp[0, 0, 0], 1.0)
        self.assertFalse(flag[0, 0, 0])
        self.assertTrue(np.isnan(visamp[0, 0, 1]))
        self.assertTrue(flag[0, 0, 1])


if __name__ == "__main__":
    unittest.main()
