"""PyCPL entry point for detector calibration and spectral tracing."""

from typing import Any

from cpl import ui

from planetes import __version__
from planetes.algorithms.flat import build_flat_product
from planetes.cube import iter_cube
from planetes.instrument import TELESCOPES
from planetes.numerics import frame_mean, frame_mean_std
from planetes.eso.common import new_output, save_product, select_inputs


class PlanetesFlat(ui.PyRecipe):
    """Build dark, flat, bad-pixel maps and extraction traces from full raw cubes."""

    _name = "planetes_flat"
    _version = __version__
    _author = "PLANETES contributors"
    _email = ""
    _copyright = "PLANETES contributors; ESO interface under development"
    _synopsis = "Build PLANETES detector maps and extraction traces"
    _description = "Inputs: DARK_RAW and TEL_FLAT_4/3/2/1. Output: planetes_flat.fits. Fixed detector geometry, thresholds, and full-cube reading are preserved."

    def __init__(self) -> None:
        super().__init__()
        self.parameters = ui.ParameterList()

    def run(self, frameset: ui.FrameSet, settings: dict[str, Any]) -> ui.FrameSet:
        """Read named detector cubes, calculate the flat product, and register FITS."""
        if settings:
            raise ValueError(f"Unknown flat parameters: {', '.join(settings)}")
        required = ["DARK_RAW", *[f"TEL_FLAT_{name}" for name in TELESCOPES]]
        inputs, used = select_inputs(frameset, required)
        output = new_output("planetes_flat.fits")
        dark, dark_std = frame_mean_std(iter_cube(inputs["DARK_RAW"]))
        flats = [
            frame_mean(iter_cube(inputs[f"TEL_FLAT_{name}"])) - dark
            for name in TELESCOPES
        ]
        return save_product(
            output, build_flat_product(dark, dark_std, flats), used, self
        )
