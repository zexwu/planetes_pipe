"""Raw-cube reading and scientific-core isolation in the ESO interface version."""

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from astropy.io import fits

from planetes.fits import iter_raw_cube as iter_cube
from planetes.dark import frame_statistics


class InputTests(unittest.TestCase):
    def test_fits_cube_iteration_and_statistics_use_the_full_cube(self) -> None:
        cube = np.arange(17 * 3 * 5, dtype=np.int16).reshape(17, 3, 5)

        with tempfile.TemporaryDirectory() as tmp:
            cube_path = Path(tmp) / "cube.fits"
            fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(cube)]).writeto(cube_path)

            chunks = list(iter_cube(cube_path))
            mean, std, _ = frame_statistics(chunks)
            expected = cube.astype(np.float32)

            self.assertEqual([len(chunk) for chunk in chunks], [16, 1])
            np.testing.assert_array_equal(np.concatenate(chunks), expected)
            np.testing.assert_allclose(mean, expected.mean(axis=0), rtol=1e-6)
            np.testing.assert_allclose(std, expected.std(axis=0), rtol=1e-6)

    def test_fits_input_must_be_a_3d_extension(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "image.fits"
            fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(np.ones((3, 5)))]).writeto(
                path
            )

            with self.assertRaisesRegex(
                ValueError, "three-dimensional raw detector cube"
            ):
                list(iter_cube(path))

    def test_scientific_core_imports_without_runner_or_project_configuration(self):
        command = "import sys; from planetes import flat, wavelength, p2vm, reduction; assert 'cpl' not in sys.modules; assert 'planetes.config' not in sys.modules"
        result = subprocess.run(
            [sys.executable, "-c", command], capture_output=True, text=True, timeout=30
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
