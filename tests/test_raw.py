"""raw.py: raw cubes, header decoding, identity by shutters."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from astropy.io import fits
from fixtures import (
    RawMetadata,
    metadata,
    write_raw,
)

from planetes_pipe import raw as raw_module
from planetes_pipe.dark import Dark, detect_bad_pixels, measure_dark, qc, write_dark
from planetes_pipe.instrument import DetectorSetup
from planetes_pipe.raw import find_raw, read_raw


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
        for dark, count in ((Dark(np.asarray(cube[0], dtype=float)), None), (single, 1)):
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
            with patch.object(raw_module.afits, "open", return_value=handle):
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


if __name__ == "__main__":
    unittest.main()
