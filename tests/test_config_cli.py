from __future__ import annotations

import tempfile
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path

import numpy as np
import yaml
from astropy.io import fits

from main import build_parser
from planetes.config import PipelineConfig
from planetes.cube import iter_cube
from planetes.numerics import frame_mean_std


def _valid_config() -> dict:
    return {
        "schema_version": 1,
        "data_dir": "data",
        "output_dir": "products",
        "phase_correction": True,
        "calibration": {
            "dark": "dark.fits",
            "flats": {
                "4": "tel4.fits",
                "3": "tel3.fits",
                "2": "tel2.fits",
                "1": "tel1.fits",
            },
            "wavelength": {"data": "wave.fits", "dark": "wave_dark.fits"},
            "p2vm": {
                "43": "bsl43.fits",
                "42": "bsl42.fits",
                "41": "bsl41.fits",
                "32": "bsl32.fits",
                "31": "bsl31.fits",
                "21": "bsl21.fits",
            },
            "phase_reference": {
                "data": "phase.fits",
                "dark": "phase_dark.fits",
            },
        },
        "targets": {"science": {"data": "science.fits", "dark": "science_dark.fits"}},
    }


class ConfigTests(unittest.TestCase):
    def _write_config(self, config: dict, directory: str) -> Path:
        path = Path(directory) / "config.yaml"
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        return path

    def test_paths_and_named_inputs_are_resolved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_config(_valid_config(), tmp)
            config = PipelineConfig.from_yaml(path)
            data_dir = (Path(tmp) / "data").resolve()

            self.assertEqual(config.data_dir, data_dir)
            self.assertEqual(config.output_dir, (Path(tmp) / "products").resolve())
            self.assertEqual(config.calibration.flats["4"], data_dir / "tel4.fits")
            self.assertEqual(config.calibration.p2vm["43"], data_dir / "bsl43.fits")
            self.assertEqual(
                config.calibration.wavelength.dark, data_dir / "wave_dark.fits"
            )
            self.assertEqual(
                config.target("science").dark, data_dir / "science_dark.fits"
            )
            self.assertTrue(config.phase_correction)

            with self.assertRaisesRegex(ValueError, "Unknown target"):
                config.target("missing")

    def test_phase_correction_is_not_coerced_from_text(self) -> None:
        raw = _valid_config()
        raw["phase_correction"] = "false"
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_config(raw, tmp)
            with self.assertRaisesRegex(ValueError, "true or false"):
                PipelineConfig.from_yaml(path)

    def test_schema_version_is_explicit(self) -> None:
        raw = _valid_config()
        raw["schema_version"] = 2
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_config(raw, tmp)
            with self.assertRaisesRegex(ValueError, "schema_version"):
                PipelineConfig.from_yaml(path)

    def test_used_config_is_an_exact_copy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_config(_valid_config(), tmp)
            config = PipelineConfig.from_yaml(path)
            used_path = Path(tmp) / "products" / "config_used.yaml"

            config.copy_used_config(used_path)

            self.assertEqual(
                used_path.read_text(encoding="utf-8"),
                path.read_text(encoding="utf-8"),
            )

    def test_fits_cube_iteration_and_statistics_use_the_full_cube(self) -> None:
        cube = np.arange(17 * 3 * 5, dtype=np.int16).reshape(17, 3, 5)

        with tempfile.TemporaryDirectory() as tmp:
            cube_path = Path(tmp) / "cube.fits"
            fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(cube)]).writeto(cube_path)

            chunks = list(iter_cube(cube_path))
            mean, std = frame_mean_std(chunks)
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

            with self.assertRaisesRegex(ValueError, "3D FITS cube"):
                list(iter_cube(path))

    def test_cli_exposes_only_complete_recipes(self) -> None:
        parser = build_parser()
        plot = parser.parse_args(["plot", "science", "--full"])
        reduce = parser.parse_args(["reduce", "science"])

        self.assertEqual(
            vars(plot),
            {
                "conf": "conf.yaml",
                "command": "plot",
                "target": "science",
                "full": True,
            },
        )
        self.assertEqual(
            vars(reduce),
            {"conf": "conf.yaml", "command": "reduce", "target": "science"},
        )
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["flat"])


if __name__ == "__main__":
    unittest.main()
