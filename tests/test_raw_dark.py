"""Raw lifetime, chunk boundaries and the direct DARK/BAD file boundary."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from astropy.io import fits
from p2vm_fixture import write_raw
from planetes_pipe.model.instrument import DetectorSetup
from planetes_pipe.model.raw import RawMetadata
from planetes_pipe.io import raw, dark, bad
from planetes_pipe.processing.dark import make_dark_products, measure_quality


class RawDarkTests(unittest.TestCase):
    def test_find_and_read_many_preserve_chunks_and_close_streams(self):
        cube = np.zeros((35, 2, 3))
        metadata = RawMetadata("FLAT_RAW", (False, False, False, True),
                               DetectorSetup(width=3, height=2))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("arbitrary-a.fits", "arbitrary-b.fits"):
                write_raw(root / name, cube, metadata)
            paths = raw.find(root, category="FLAT_RAW")
            self.assertEqual(len(paths), 2)
            self.assertEqual(raw.find(root, category="P2VM_RAW"), [])
            with raw.read_many(paths, category="FLAT_RAW") as sources:
                self.assertEqual([len(next(s.frames)) for s in sources], [16, 16])
            for source in sources:
                self.assertEqual(list(source.frames), [])

    def test_chunks_statistics_roundtrip_and_raw_immutability(self):
        cube = 1e6 + np.random.default_rng(22).normal(size=(35, 4, 5))
        metadata = RawMetadata("DARK_RAW", (False,) * 4, DetectorSetup(
            exposure_s=.1, width=5, height=4, detector="test",
            origin_x=0, origin_y=0, binning_x=1, binning_y=1))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.fits"
            write_raw(path, cube, metadata)
            original = path.read_bytes()
            with raw.read(path, category="DARK_RAW") as source:
                chunks = list(source.frames)
                self.assertEqual([len(c) for c in chunks], [16, 16, 3])
                self.assertTrue(all(c.dtype == np.float64 for c in chunks))
            with raw.read(path, category="DARK_RAW") as source:
                result, flags = make_dark_products(source)
            np.testing.assert_allclose(result.mean_adu, cube.mean(axis=0), rtol=0, atol=5e-10)
            np.testing.assert_allclose(result.temporal_rms_adu, cube.std(axis=0), rtol=0, atol=5e-10)
            self.assertEqual(result.n_frames, 35)
            self.assertEqual(result.setup, metadata.setup)
            for module, product in ((dark, result), (bad, flags)):
                output = Path(directory) / (type(product).__name__ + ".fits")
                module.write(output, product, quality=measure_quality(product))
                loaded = module.read(output)
                field = "mean_adu" if module is dark else "bad_pixel_flags"
                expected = getattr(product, field)
                if module is dark:
                    expected = expected.astype(np.float32).astype(float)
                np.testing.assert_array_equal(getattr(loaded, field), expected)
                with self.assertRaises(OSError):
                    module.write(output, product)
            self.assertEqual(path.read_bytes(), original)

    def test_early_exit_and_exception_close_file_and_iterator(self):
        metadata = RawMetadata("DARK_RAW", (False,) * 4, DetectorSetup(width=2, height=2, exposure_s=.1,
            origin_x=0, origin_y=0, binning_x=1, binning_y=1))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.fits"
            write_raw(path, np.zeros((35, 2, 2)), metadata)
            for fail in (False, True):
                handle = fits.open(path, memmap=True)
                with patch.object(raw.afits, "open", return_value=handle):
                    try:
                        with raw.read(path, category="DARK_RAW") as source:
                            next(source.frames)
                            if fail:
                                raise RuntimeError("stop")
                    except RuntimeError:
                        pass
                self.assertTrue(handle._file.closed)
                self.assertEqual(list(source.frames), [])
