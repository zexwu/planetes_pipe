"""Real DFS failures may not publish partial products or modify existing files."""

from contextlib import chdir
from importlib.util import find_spec
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from astropy.io import fits

from planetes.models import DarkCalibration


@unittest.skipUnless(find_spec("cpl") is not None, "PyCPL runtime not installed")
class DFSFailureTests(unittest.TestCase):
    def test_failure_after_dfs_header_is_cleaned_and_retry_succeeds(self):
        from cpl import ui
        from planetes.eso.dark import PlanetesDark
        from planetes.eso.common import save_product

        product = DarkCalibration(
            np.ones((2, 3), dtype=np.float32),
            np.ones((2, 3), dtype=np.float32),
            np.zeros((2, 3), dtype=np.int16),
            2,
        )
        with tempfile.TemporaryDirectory() as temporary, chdir(temporary):
            raw = Path("raw.fits").absolute()
            fits.PrimaryHDU(header=fits.Header({"INSTRUME": "PLANETES"})).writeto(raw)
            used = ui.FrameSet(
                [ui.Frame(file=str(raw), tag="DARK_RAW", group=ui.Frame.FrameGroup.RAW)]
            )
            with patch(
                "planetes.eso.common.fill_dfs_product",
                side_effect=ValueError("injected payload failure"),
            ):
                with self.assertRaisesRegex(ValueError, "injected"):
                    save_product(Path("dark.fits"), product, used, PlanetesDark())
            self.assertFalse(Path("dark.fits").exists())
            self.assertEqual(sorted(p.name for p in Path(".").iterdir()), ["raw.fits"])
            save_product(Path("dark.fits"), product, used, PlanetesDark())
            self.assertTrue(Path("dark.fits").is_file())
            before = Path("dark.fits").read_bytes()
            with self.assertRaises(FileExistsError):
                save_product(Path("dark.fits"), product, used, PlanetesDark())
            self.assertEqual(before, Path("dark.fits").read_bytes())

    def test_nondefault_dark_and_flat_parameters_use_public_task_defaults(self):
        from cpl import ui
        from planetes.eso.dark import PlanetesDark
        from planetes.eso.flat import PlanetesFlat
        from planetes.fits import read_product

        with tempfile.TemporaryDirectory() as temporary, chdir(temporary):
            ny, nx = 576, 128
            dark = np.full((3, ny, nx), 10, dtype=np.float32)
            header = fits.Header({"INSTRUME": "PLANETES"})
            fits.HDUList([fits.PrimaryHDU(header=header), fits.ImageHDU(dark)]).writeto(
                "raw_dark.fits"
            )
            dark_frames = ui.FrameSet(
                [ui.Frame(file=str(Path("raw_dark.fits").absolute()), tag="DARK_RAW")]
            )
            PlanetesDark().run(dark_frames, {"planetes.dark.bad_threshold": 6.5})
            measured = read_product(Path("planetes_dark.fits"))
            self.assertEqual(measured.provenance.parameters["bad_threshold"], 6.5)
            rows = np.arange(ny)
            profile = sum(
                np.exp(-0.5 * ((rows - center) / 2.0) ** 2)
                for center in 12 + 24 * np.arange(24)
            )
            image = np.tile((1000 * profile)[:, None], (1, nx)).astype(np.float32)
            frames = ui.FrameSet(
                [
                    ui.Frame(
                        file=str(Path("planetes_dark.fits").absolute()),
                        tag="DARK_CALIB",
                    )
                ]
            )
            for name in ("4", "3", "2", "1"):
                path = Path(f"tel{name}.fits").absolute()
                fits.HDUList(
                    [fits.PrimaryHDU(header=header), fits.ImageHDU(image[None] + dark)]
                ).writeto(path)
                frames.append(ui.Frame(file=str(path), tag=f"TEL_FLAT_{name}"))
            PlanetesFlat().run(frames, {"planetes.flat.profile_width": 3.5})
            flat = read_product(Path("planetes_flat.fits"))
            self.assertEqual(flat.provenance.parameters["profile_width"], 3.5)
            self.assertEqual(flat.profile_x_pixels.shape[0], 24)


if __name__ == "__main__":
    unittest.main()
