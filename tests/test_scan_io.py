"""Raw pixels and P2VM scan-table decoding have independent file boundaries."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
from astropy.io import fits
from planetes_pipe.io import raw, p2vm
from planetes_pipe.io.common import write_shutters
from planetes_pipe.model.instrument import layout


class ScanIOTests(unittest.TestCase):
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
            with raw.read(path, category="P2VM_RAW", hdu=0) as source:
                chunks = list(source.frames)
            self.assertEqual([len(c) for c in chunks], [16, 16, 3])
            np.testing.assert_array_equal(np.concatenate(chunks), cube)

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
            actual = p2vm.read_scan_opd(list(reversed(paths.values())), gains)
            expected = np.array([gains[b] * np.array([0., 30e-6, 30e-6, -30e-6])
                                 for b in layout.baselines])
            np.testing.assert_allclose(actual, expected, rtol=1e-15, atol=0)
