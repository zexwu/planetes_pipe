from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml
from astropy.io import fits

from planetes import COMMANDS, PipelineContext, arg, command
from planetes.fits import CubeSelection, iter_cube
from planetes.numerics import frame_mean_std
from planetes.products import ProductSpec


def _valid_config() -> dict:
    return {
        "frames": [0, 2],
        "xrange": [0, 4],
        "yrange": [0, 4],
        "data_dir": ".",
        "output_dir": "output",
        "telescopes": ["2", "1"],
        "baselines": ["21"],
        "abcd_order": ["A", "C", "B", "D"],
        "calib": {
            "dark": "dark.fits",
            "wave": "wave.fits",
            "wave_dark": "wave_dark.fits",
            "flat": ["tel2.fits", "tel1.fits"],
            "p2vm": ["bsl21.fits"],
        },
        "object": {"target": {"data": "science.fits", "dark": "dark.fits"}},
        "products": {},
        "parameters": {},
    }


class ConfigAndCommandTests(unittest.TestCase):
    def _write_config(self, config: dict, directory: str) -> Path:
        path = Path(directory) / "config.yaml"
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        return path

    def test_pipeline_context_validates_baseline_pairs(self) -> None:
        config = _valid_config()
        config["baselines"] = ["12"]
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_config(config, tmp)
            with self.assertRaisesRegex(ValueError, "telescope pair"):
                PipelineContext(path)

    def test_recipe_parameters_use_config_then_cli_override(self) -> None:
        name = "_parameter_precedence_test"

        @command(name, "test")
        @arg("--value", type=int, default=1)
        def sample(ctx, **kwargs):
            return kwargs["value"]

        class Context:
            conf = {"parameters": {name: {"value": 2}}}
            active_parameters = {}

        try:
            self.assertEqual(sample(Context()), 2)
            self.assertEqual(sample(Context(), value=3), 3)
        finally:
            COMMANDS.pop(name, None)

    def test_recipe_rejects_unknown_configured_parameter(self) -> None:
        name = "_unknown_parameter_test"

        @command(name, "test")
        @arg("--value", type=int, default=1)
        def sample(ctx, **kwargs):
            return kwargs["value"]

        class Context:
            conf = {"parameters": {name: {"typo": 2}}}
            active_parameters = {}

        try:
            with self.assertRaisesRegex(ValueError, "Unknown parameters"):
                sample(Context())
        finally:
            COMMANDS.pop(name, None)

    def test_product_manifest_detects_parameter_changes(self) -> None:
        config = _valid_config()
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_config(config, tmp)
            ctx = PipelineContext(path)
            product = ProductSpec("flat", ("array",))
            ctx.save_product(product, array=np.arange(4))

            product_path = ctx.output_dir / "flat.npz"
            manifest_path = ctx.output_dir / "flat.npz.manifest.json"
            self.assertTrue(product_path.exists())
            self.assertTrue(manifest_path.exists())
            self.assertTrue(ctx.product_exists(product))
            self.assertFalse(list(ctx.output_dir.glob("*.tmp")))

            ctx.conf.setdefault("parameters", {})["flat"] = {"profile_width": 99}
            self.assertFalse(ctx.product_exists(product))

    def test_fits_cube_iteration_and_statistics_are_chunked(self) -> None:
        config = _valid_config()
        config["frames"] = [1, 5]
        config["xrange"] = [1, 4]
        config["yrange"] = [0, 2]
        config["io_chunk_frames"] = 2
        cube = np.arange(6 * 3 * 5, dtype=np.int16).reshape(6, 3, 5)

        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_config(config, tmp)
            cube_path = Path(tmp) / "cube.fits"
            fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(cube)]).writeto(cube_path)
            ctx = PipelineContext(path)

            chunks = list(ctx.iter_fits("cube.fits"))
            selected = cube[1:5, 0:2, 1:4].astype(np.float32)
            self.assertEqual([len(chunk) for chunk in chunks], [2, 2])
            np.testing.assert_array_equal(np.concatenate(chunks), selected)

            mean, std = ctx.mean_std_fits("cube.fits")
            np.testing.assert_allclose(mean, selected.mean(axis=0), rtol=1e-6)
            np.testing.assert_allclose(std, selected.std(axis=0), rtol=1e-6)

            selection = CubeSelection(
                frames=slice(1, 5), y=slice(0, 2), x=slice(1, 4)
            )
            direct_chunks = list(iter_cube(cube_path, selection, chunk_frames=2))
            direct_mean, direct_std = frame_mean_std(direct_chunks)
            np.testing.assert_array_equal(np.concatenate(direct_chunks), selected)
            np.testing.assert_allclose(direct_mean, mean, rtol=1e-6)
            np.testing.assert_allclose(direct_std, std, rtol=1e-6)

    def test_dynamic_object_is_normalized_in_product_snapshot(self) -> None:
        config = _valid_config()
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_config(config, tmp)
            ctx = PipelineContext(path)
            ctx.active_parameters["preproc"] = {"object": "target"}
            ctx.active_parameters["reduce"] = {"object": "target"}
            active_snapshot = ctx.manifests.snapshot(("reduced", "target"))

            ctx.active_parameters.clear()
            fresh_snapshot = ctx.manifests.snapshot(("reduced", "target"))
            self.assertEqual(active_snapshot, fresh_snapshot)


if __name__ == "__main__":
    unittest.main()
