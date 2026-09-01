from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml
from astropy.io import fits

from planetes.config import PipelineConfig
from planetes.fits import CubeSelection, iter_cube
from planetes.numerics import frame_mean_std
from main import build_parser


def _valid_config() -> dict:
    return {
        "schema_version": 1,
        "data_dir": ".",
        "output_dir": "output",
        "chunk_frames": 2,
        "automatic_plots": False,
        "crop": {"x": [0, 4], "y": [0, 4]},
        "instrument": {
            "telescopes": ["2", "1"],
            "baselines": ["21"],
            "abcd_order": ["A", "C", "B", "D"],
        },
        "calibration": {
            "frames": [0, 2],
            "dark": "dark.fits",
            "flats": {"2": "tel2.fits", "1": "tel1.fits"},
            "wavelength": {"data": "wave.fits", "dark": "wave_dark.fits"},
            "p2vm": {"21": "bsl21.fits"},
        },
        "targets": {
            "target": {"data": "science.fits", "dark": "dark.fits"}
        },
        "parameters": {},
    }


class ConfigTests(unittest.TestCase):
    def _write_config(self, config: dict, directory: str) -> Path:
        path = Path(directory) / "config.yaml"
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        return path

    def test_config_validates_baseline_pairs(self) -> None:
        config = _valid_config()
        config["instrument"]["baselines"] = ["12"]
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_config(config, tmp)
            with self.assertRaisesRegex(ValueError, "telescope pair"):
                PipelineConfig.from_yaml(path)

    def test_named_inputs_and_paths_are_resolved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_config(_valid_config(), tmp)
            config = PipelineConfig.from_yaml(path)

            self.assertEqual(config.instrument.telescopes, ("2", "1"))
            self.assertEqual(
                config.calibration.flats["2"], (Path(tmp) / "tel2.fits").resolve()
            )
            self.assertEqual(config.targets["target"].frames.slice, slice(0, 2))
            with self.assertRaises(TypeError):
                config.targets["new"] = config.targets["target"]  # type: ignore[index]

    def test_unknown_parameter_is_rejected(self) -> None:
        config = _valid_config()
        config["parameters"] = {"flat": {"bad_thresh_typo": 5}}
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_config(config, tmp)
            with self.assertRaisesRegex(ValueError, "Unknown keys"):
                PipelineConfig.from_yaml(path)

    def test_boolean_parameter_is_not_coerced_from_text(self) -> None:
        config = _valid_config()
        config["parameters"] = {"p2vm": {"phase_correction": "false"}}
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_config(config, tmp)
            with self.assertRaisesRegex(ValueError, "true or false"):
                PipelineConfig.from_yaml(path)

    def test_resolved_config_can_be_saved_and_reloaded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_config(_valid_config(), tmp)
            config = PipelineConfig.from_yaml(path)
            used_path = Path(tmp) / "output" / "config_used.yaml"

            config.save(used_path)
            reloaded = PipelineConfig.from_yaml(used_path)

            self.assertEqual(reloaded.as_mapping(), config.as_mapping())

    def test_fits_cube_iteration_and_statistics_are_chunked(self) -> None:
        cube = np.arange(6 * 3 * 5, dtype=np.int16).reshape(6, 3, 5)

        with tempfile.TemporaryDirectory() as tmp:
            cube_path = Path(tmp) / "cube.fits"
            fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(cube)]).writeto(cube_path)
            selection = CubeSelection(
                frames=slice(1, 5), y=slice(0, 2), x=slice(1, 4)
            )

            chunks = list(iter_cube(cube_path, selection, chunk_frames=2))
            selected = cube[1:5, 0:2, 1:4].astype(np.float32)
            mean, std = frame_mean_std(chunks)

            self.assertEqual([len(chunk) for chunk in chunks], [2, 2])
            np.testing.assert_array_equal(np.concatenate(chunks), selected)
            np.testing.assert_allclose(mean, selected.mean(axis=0), rtol=1e-6)
            np.testing.assert_allclose(std, selected.std(axis=0), rtol=1e-6)

    def test_cli_only_selects_recipe_and_target(self) -> None:
        args = build_parser().parse_args(["reduce", "target"])
        self.assertEqual(
            vars(args),
            {"conf": "conf.yaml", "command": "reduce", "target": "target"},
        )


if __name__ == "__main__":
    unittest.main()
