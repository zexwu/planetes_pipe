from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from recipes import COMMANDS, PipelineContext, arg, command
from recipes.products import ProductSpec


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


if __name__ == "__main__":
    unittest.main()
