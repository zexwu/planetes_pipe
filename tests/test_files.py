"""FITS layer: raw cubes, product round trips, names and no-overwrite."""


import tempfile
import unittest
from dataclasses import fields
from pathlib import Path
from unittest.mock import patch

import numpy as np
from astropy.io import fits
from fixtures import (
    RawMetadata,
    calibration_arrays,
    calibration_spectra,
    metadata,
    write_product,
    write_raw,
)

from planetes_pipe import files
from planetes_pipe.dark import Dark, detect_bad_pixels, measure_dark, qc, write_dark
from planetes_pipe.files import as_compute_array, find_raw, product_filename, read_raw
from planetes_pipe.flat import Flat, read_flat
from planetes_pipe.instrument import DetectorSetup
from planetes_pipe.p2vm import P2VM, fit_p2vm, read_p2vm
from planetes_pipe.wave import Wave, read_wave, write_wave

READERS = {Flat: read_flat, Wave: read_wave, P2VM: read_p2vm}


class CalibrationFitsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        arrays = calibration_arrays(n_wave=15)
        cls.spectra = calibration_spectra(n_wave=15)
        cls.flat, cls.wave = arrays[-2:]
        cls.p2vm = fit_p2vm(cls.spectra)
        cls.p2vm_diagnostics = fit_p2vm(cls.spectra, keep_diagnostics=True)

    def test_fits_round_trip_preserves_storage_precision_and_mappings(self):
        for product in [self.flat, self.wave, self.p2vm, self.p2vm_diagnostics]:
            with (
                self.subTest(product=type(product).__name__),
                tempfile.TemporaryDirectory() as tmp,
            ):
                path = Path(tmp) / "calibration.fits"
                write_product(path, product)
                with fits.open(path, mode="update") as hdus:
                    self.assertNotIn("PLANETES_META", hdus)
                    self.assertNotIn("PLANETES FORMAT", hdus[0].header)
                    # Product layout, not a private version number, determines reading.
                    hdus[0].header["HIERARCH PLANETES FORMAT"] = -1
                    hdus[0].add_checksum()
                loaded = READERS[type(product)](path)
                for field in fields(product):
                    before, after = (
                        getattr(product, field.name),
                        getattr(loaded, field.name),
                    )
                    if isinstance(before, np.ndarray):
                        expected = before
                        if before.dtype.kind in "fc":
                            stored = (
                                np.complex64 if before.dtype.kind == "c" else np.float32
                            )
                            expected = before.astype(stored).astype(before.dtype)
                        np.testing.assert_array_equal(expected, after)
                        self.assertEqual(after.dtype, before.dtype)
                        self.assertTrue(after.dtype.isnative)
                    else:
                        self.assertEqual(before, after)

    def test_fits_writer_preserves_existing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "wave.fits"
            write_wave(path, self.wave)
            header = fits.getheader(path)
            self.assertEqual(header["ESO QC MINWAVE SC"],
                             float(self.wave.wavelength_map_m.min(axis=1).max()))
            original = path.read_bytes()
            with self.assertRaises(OSError):
                write_wave(path, self.wave)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(Path(tmp).glob(".planetes-*")), [])


class RawTests(unittest.TestCase):
    def test_bench_cubes_keep_missing_metadata_and_all_frames(self):
        cube = np.arange(17 * 3 * 5, dtype=np.int16).reshape(17, 3, 5)
        for primary in (True, False):
            with self.subTest(primary=primary), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "bench.fits"
                hdus = (
                    [fits.PrimaryHDU(cube)]
                    if primary
                    else [fits.PrimaryHDU(), fits.ImageHDU(cube)]
                )
                fits.HDUList(hdus).writeto(path)
                original = path.read_bytes()
                raw = read_raw(path, "DARK_RAW", hdu=0 if primary else 1)
                self.assertEqual(raw.shutters, (None,) * 4)
                self.assertIsNone(raw.setup.exposure_s)
                self.assertEqual((raw.setup.height, raw.setup.width), (3, 5))
                self.assertNotIn("ESO DET WIN1 DIT1", fits.getheader(path))
                chunks = list(raw.chunks())
                self.assertEqual([len(chunk) for chunk in chunks], [16, 1])
                self.assertTrue(all(chunk.dtype == np.float64 for chunk in chunks))
                np.testing.assert_array_equal(np.concatenate(chunks), cube)
                self.assertEqual(path.read_bytes(), original)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "raw.fits"
            physical = metadata("DARK_RAW", n_wave=5, height=3)
            write_raw(path, cube, physical)
            raw = read_raw(path, "DARK_RAW")
            self.assertEqual(RawMetadata(raw.category, raw.shutters, raw.setup), physical)
            with fits.open(path) as hdus:
                self.assertEqual(
                    [hdu.name for hdu in hdus], ["PRIMARY", "IMAGING_DATA_SC"]
                )
                self.assertEqual(hdus[0].header["ESO DET ID"], "synthetic")
                np.testing.assert_array_equal(hdus[1].data, cube)
                for hdu in hdus:
                    self.assertEqual(hdu.verify_checksum(), 1)
                    self.assertEqual(hdu.verify_datasum(), 1)
            original = path.read_bytes()
            with self.assertRaises(OSError):
                write_raw(path, cube, physical)
            self.assertEqual(path.read_bytes(), original)
        single = measure_dark([cube[:1]])
        self.assertEqual(single.n_frames, 1)
        self.assertIsNone(single.temporal_rms_adu)
        samples = np.array([1e12 - 1, 1e12 + 1]).reshape(2, 1, 1)
        for chunks in ([samples], [samples[:1], samples[1:]]):
            measured = measure_dark(chunks)
            self.assertEqual(measured.mean_adu.item(), 1e12)
            self.assertEqual(measured.temporal_rms_adu.item(), 1.0)
            self.assertEqual(measured.mean_adu.dtype, np.float64)
        for dark, count in ((Dark(as_compute_array(cube[0])), None), (single, 1)):
            with tempfile.TemporaryDirectory() as tmp:
                np.testing.assert_array_equal(dark.mean_adu, cube[0])
                self.assertEqual(dark.n_frames, count)
                self.assertIsNone(dark.temporal_rms_adu)
                quality = qc(dark)
                self.assertEqual(quality["ESO QC MEDIANDARK SC"][0], np.median(cube[0]))
                self.assertIsNone(quality["ESO QC DARKRMS SC"][0])
                with self.assertRaisesRegex(ValueError, "temporal RMS"):
                    detect_bad_pixels(dark)
                with self.assertRaisesRegex(ValueError, "Saved DARK"):
                    write_dark(Path(tmp) / "product.fits", dark)

    def test_find_raw_selects_by_header_type(self):
        metadata = RawMetadata("FLAT_RAW", (False, False, False, True),
                               DetectorSetup(width=3, height=2))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("arbitrary-a.fits", "arbitrary-b.fits"):
                write_raw(root / name, np.zeros((35, 2, 3)), metadata)
            self.assertEqual(len(find_raw(root, "FLAT_RAW")), 2)
            self.assertEqual(find_raw(root, "P2VM_RAW"), [])

    def test_read_raw_closes_file_and_keeps_cube_readable(self):
        metadata = RawMetadata("DARK_RAW", (False,) * 4, DetectorSetup(width=2, height=2))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.fits"
            write_raw(path, np.arange(140.).reshape(35, 2, 2), metadata)
            handle = fits.open(path, memmap=True)
            with patch.object(files.afits, "open", return_value=handle):
                source = read_raw(path, "DARK_RAW")
            self.assertTrue(handle._file.closed)
            self.assertEqual(float(source.cube[34, 1, 1]), 139.)
            self.assertEqual(sum(len(c) for c in source.chunks()), 35)

    def test_raw_does_not_parse_scan_table(self):
        cube = np.arange(35 * 2 * 3, dtype=float).reshape(35, 2, 3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.fits"
            # Deliberately unrelated columns: reading pixels does not need a scan.
            fits.HDUList([
                fits.PrimaryHDU(cube),
                fits.BinTableHDU.from_columns([
                    fits.Column(name="UNRELATED", format="K", array=[1]),
                ], name="DL_SCAN"),
            ]).writeto(path)
            chunks = list(read_raw(path, "P2VM_RAW", hdu=0).chunks())
            self.assertEqual([len(c) for c in chunks], [16, 16, 3])
            np.testing.assert_array_equal(np.concatenate(chunks), cube)

    def test_product_filename_preserves_reference_identity(self):
        for stem in (
            "PLANETES.2026-02-05T210609",
            "PLANETES.2026-02-05T21:06:09.123",
            "PLANETES.2026-02-05TUNKNOWN.0006",
        ):
            with self.subTest(stem=stem):
                reference = Path("/unread/raw") / f"{stem}.fits"
                self.assertEqual(
                    product_filename(reference, "SINGLE_SCI_P2VMRED"),
                    f"{stem}_singlescip2vmred.fits",
                )
                self.assertEqual(
                    product_filename(reference, "DARK"),
                    f"{stem}_dark.fits",
                )


if __name__ == "__main__":
    unittest.main()
