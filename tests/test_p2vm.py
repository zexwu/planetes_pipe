"""p2vm.py: wavelength grid, response fit, phase reference and DL scan OPD."""


import tempfile
import unittest
from pathlib import Path

import numpy as np
from astropy.io import fits
from fixtures import calibration_arrays, calibration_spectra, write_recipe_inputs

from planetes_pipe.instrument import layout
from planetes_pipe.io.raw import Raw, read_raw, read_raws, write_shutters
from planetes_pipe.steps.dark import Dark, read_dark
from planetes_pipe.steps.extraction import extract_spectra, prepare_extraction
from planetes_pipe.steps.flat import read_flat, write_flat
from planetes_pipe.steps.p2vm import (
    build_v2pm,
    extract_p2vm_spectra,
    fit_abcd_ellipse,
    fit_p2vm,
    read_scan_opd,
    unwrap_scan_phase,
    wavelength_grid,
)
from planetes_pipe.steps.science import reduce_spectra
from planetes_pipe.steps.wave import read_wave


class WavelengthGridTests(unittest.TestCase):
    def test_intersection_at_native_sampling_and_fits_roundtrip(self):
        *_, flat, wave = calibration_arrays(n_wave=11)
        flat.trace_x_range[:] = [1, 9]
        flat.trace_x_range[0] = [3, 7]
        wave.wavelength_map_m[:] = np.arange(11) * 1e-9 + 1e-6
        wave.wavelength_map_m[0] += 0.2e-9
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "flat.fits"
            write_flat(path, flat)
            xrange = fits.getdata(path, "PROFILE_PARAMS")["XRANGE"][0]
            np.testing.assert_array_equal(xrange, [4, 8])  # One-based on disk.
            restored = read_flat(path)
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


class P2VMTests(unittest.TestCase):
    def test_p2vm_uses_shutters_with_reversed_lists_and_arbitrary_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = write_recipe_inputs(root / "versioned")
            flat, wave = read_flat(paths["FLAT"]), read_wave(paths["WAVE"])
            dark = read_dark(paths["DARK"])
            with self.assertRaisesRegex(ValueError, "outside calibration"):
                prepare_extraction(flat, wave, np.linspace(1.14e-6, 1.30e-6, 15))
            extractor = prepare_extraction(flat, wave, wavelength_grid(flat, wave))

            def fit(flat_files, scan_files, hdu):
                flats = read_raws(flat_files, "FLAT_RAW", layout.telescopes, hdu=hdu)
                scans = read_raws(scan_files, "P2VM_RAW", layout.baselines, hdu=hdu)
                return fit_p2vm(extract_p2vm_spectra(flats, scans, extractor, dark, flat))

            expected = fit(paths["FLAT_RAW"], paths["P2VM_RAW"], "IMAGING_DATA_SC")

            def bench_files(role, identities):
                result = {}
                for identity, path in zip(identities, paths[role], strict=True):
                    target = root / f"unrelated-{len(result)}-{role}.fits"
                    with fits.open(path) as hdus:
                        fits.PrimaryHDU(hdus[1].data, header=hdus[0].header).writeto(target)
                    result[identity] = target
                return result

            flats = bench_files("FLAT_RAW", ("4", "3", "2", "1")).values()
            scans = bench_files("P2VM_RAW", ("43", "42", "41", "32", "31", "21")).values()
            actual = fit(reversed(list(flats)), reversed(list(scans)), 0)
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
            flat, wave = (read_flat(inputs["FLAT"]), read_wave(inputs["WAVE"]))
            main = read_dark(inputs["DARK"])
            phase = Dark(
                np.full(main.mean_adu.shape, 7, dtype=np.float32), detector=main.detector
            )
            raw = np.array(fits.getdata(reference), dtype=np.float32)
            corrected = raw - 7
            prepared = prepare_extraction(
                flat, wave, np.linspace(1.15e-6, 1.30e-6, 151)
            )
            reference_spectra = extract_spectra(
                read_raw(reference, "SINGLE_SCI_RAW"), prepared, phase)
            expected = prepared.extract(corrected, Dark(np.zeros_like(phase.mean_adu)))
            np.testing.assert_array_equal(reference_spectra, expected)
            self.assertFalse(
                np.array_equal(reference_spectra, prepared.extract(raw, main))
            )

    def test_ellipse_radius_is_normalized_independently_of_fringe_contrast(self):
        phase_rad = np.linspace(0, 2 * np.pi, 64, endpoint=False)[:, None]
        phase_rad = phase_rad + np.array([0.0, 0.25, 0.5])
        abcd_phase_rad = np.array([0.0, np.pi, np.pi / 2, 3 * np.pi / 2])
        expected_phasor = np.exp(-1j * (phase_rad - phase_rad[0]))
        for contrast_adu in (50.0, 400.0):
            spectra = 1000 + contrast_adu * np.cos(
                phase_rad[None, :, :] + abcd_phase_rad[:, None, None]
            )
            ellipse_phase_rad, ellipse_radius = fit_abcd_ellipse(spectra)
            np.testing.assert_allclose(ellipse_radius, 1.0, rtol=0, atol=1e-12)
            # The retained A,C,B,D differences give phase decreasing with input OPD.
            np.testing.assert_allclose(
                np.exp(1j * (ellipse_phase_rad - ellipse_phase_rad[0])),
                expected_phasor,
                rtol=0,
                atol=1e-12,
            )

    def test_p2vm_inverts_known_full_rank_instrument_response(self):
        spectra = calibration_spectra(n_wave=15)
        p2vm = fit_p2vm(spectra)
        diagnostic = fit_p2vm(spectra, keep_diagnostics=True)
        self.assertIsNone(p2vm.opd_m)
        self.assertIsNotNone(diagnostic.opd_m)
        self.assertEqual(p2vm.matrix.tobytes(), diagnostic.matrix.tobytes())
        matrix = build_v2pm(
            p2vm.transmission, p2vm.coherence_amplitude, p2vm.coherence_phase_rad
        )
        for wave in range(len(p2vm.wavelength_m)):
            np.testing.assert_allclose(
                p2vm.matrix[:, :, wave] @ matrix[:, :, wave], np.eye(layout.n_data), atol=1e-12
            )
        reference = calibration_arrays(n_wave=15)[2]
        with self.assertRaisesRegex(ValueError, "dimensions"):
            reduce_spectra(reference[..., :1], p2vm)

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

    def test_explicit_scan_conversion_preserves_frame_order_units_and_sign(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = {}
            gains = {baseline: (-1 if i % 2 else 1) for i, baseline in enumerate(layout.baselines)}
            for baseline in layout.baselines:
                table = fits.BinTableHDU.from_columns([
                    fits.Column(name="FIRST_FRAME", format="K", array=[0, 1, 3]),
                    fits.Column(name="NFRAMES", format="K", array=[1, 2, 1]),
                    fits.Column(name="BEFORE_MM", format="D", array=[0., .02, -.04]),
                    fits.Column(name="AFTER_MM", format="D", array=[0., .04, -.02]),
                ], name="DL_SCAN")
                table.header["MOVINGDL"] = int(baseline[1])
                path = Path(directory) / f"{baseline}.fits"
                header = fits.Header()
                write_shutters(header, tuple(str(t) in baseline for t in range(1, 5)))
                fits.HDUList([fits.PrimaryHDU(header=header), table]).writeto(path)
                paths[baseline] = path
            scans = {b: Raw(np.zeros((1, 1, 1)), "P2VM_RAW", fits.Header(), path)
                     for b, path in paths.items()}
            actual = read_scan_opd(scans, gains)
            expected = np.array([gains[b] * np.array([0., 30e-6, 30e-6, -30e-6])
                                 for b in layout.baselines])
            np.testing.assert_allclose(actual, expected, rtol=1e-15, atol=0)


if __name__ == "__main__":
    unittest.main()
