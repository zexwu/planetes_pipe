"""dark.py: dark statistics, bad pixels, DARK/BAD files and run()."""


import tempfile
import unittest
from pathlib import Path

import numpy as np
from astropy.io import fits
from fixtures import RawMetadata, metadata, write_raw

from planetes_pipe import dark
from planetes_pipe.dark import (
    make_dark_products,
    read_bad,
    read_dark,
    write_bad,
    write_dark,
)
from planetes_pipe.files import read_raw
from planetes_pipe.instrument import DetectorSetup


class DarkTests(unittest.TestCase):
    def test_chunks_statistics_roundtrip_and_raw_immutability(self):
        cube = 1e6 + np.random.default_rng(22).normal(size=(35, 4, 5))
        metadata = RawMetadata("DARK_RAW", (False,) * 4, DetectorSetup(
            exposure_s=.1, width=5, height=4, detector="test",
            origin_x=0, origin_y=0, binning_x=1, binning_y=1))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.fits"
            write_raw(path, cube, metadata)
            original = path.read_bytes()
            source = read_raw(path, "DARK_RAW")
            chunks = list(source.chunks())
            self.assertEqual([len(c) for c in chunks], [16, 16, 3])
            self.assertTrue(all(c.dtype == np.float64 for c in chunks))
            result, flags = make_dark_products(source)
            np.testing.assert_allclose(result.mean_adu, cube.mean(axis=0), rtol=0, atol=5e-10)
            np.testing.assert_allclose(result.temporal_rms_adu, cube.std(axis=0), rtol=0, atol=5e-10)
            self.assertEqual(result.n_frames, 35)
            self.assertEqual(result.setup, metadata.setup)
            for write, read, product, field in (
                (write_dark, read_dark, result, "mean_adu"),
                (write_bad, read_bad, flags, "bad_pixel_flags"),
            ):
                output = Path(directory) / (type(product).__name__ + ".fits")
                write(output, product)
                expected = getattr(product, field)
                if field == "mean_adu":
                    expected = expected.astype(np.float32).astype(float)
                np.testing.assert_array_equal(getattr(read(output), field), expected)
                with self.assertRaises(OSError):
                    write(output, product)
            self.assertEqual(path.read_bytes(), original)

    def test_dark_run_writes_products_with_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "PLANETES.2026-02-05T200000.fits"
            write_raw(source, np.arange(30, dtype=float).reshape(2, 3, 5),
                      metadata("DARK_RAW", n_wave=5, height=3))
            written = dark.run(source, output_dir=root / "out", bad_pixel_threshold=4.0)
            self.assertEqual({k: p.name for k, p in written.items()}, {
                "DARK": "PLANETES.2026-02-05T200000_dark.fits",
                "BAD": "PLANETES.2026-02-05T200000_bad.fits"})
            np.testing.assert_allclose(dark.read_dark(written["DARK"]).mean_adu,
                                       np.arange(15).reshape(3, 5) + 7.5)
            header = fits.getheader(written["DARK"])
            self.assertEqual(header["ESO PRO REC1 ID"], "planetes_dark")
            self.assertEqual(header["ESO PRO REC1 RAW1 NAME"], source.name)
            self.assertEqual(header["ESO PRO REC1 RAW1 CATG"], "DARK_RAW")
            self.assertEqual(header["ESO PRO DATANCOM"], 1)
            self.assertEqual(header["ESO PRO REC1 PARAM1 NAME"], "bad-dark-threshold")
            self.assertEqual(header["ESO PRO REC1 PARAM1 VALUE"], "4")
            self.assertEqual(header["PIPEFILE"], written["DARK"].name)
            self.assertNotIn("ESO DPR TYPE", header)
            self.assertIn("ESO QC MEDIANDARK SC", header)
            with self.assertRaises(OSError):  # Never overwrite.
                dark.run(source, output_dir=root / "out")


if __name__ == "__main__":
    unittest.main()
