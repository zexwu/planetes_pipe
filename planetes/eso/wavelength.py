"""PyCPL entry point for wavelength calibration with a traceable static line list."""

from typing import Any

from cpl import ui

from planetes import __version__
from planetes.algorithms.wavelength import compute_wave
from planetes.cube import iter_cube
from planetes.io.products import read_product
from planetes.io.reference import load_wavelength_lines, wavelength_reference_path
from planetes.numerics import frame_mean
from planetes.product import FlatProduct
from planetes.eso.common import new_output, save_product, select_inputs


class PlanetesWave(ui.PyRecipe):
    """Build a wavelength map from a lamp cube, its dark, and extraction geometry."""

    _name = "planetes_wave"
    _version = __version__
    _author = "PLANETES contributors"
    _email = ""
    _copyright = "PLANETES contributors; ESO interface under development"
    _synopsis = "Calibrate the PLANETES detector wavelength map"
    _description = "Inputs: WAVE_RAW, WAVE_DARK, PLANETES_FLAT; optional PLANETES_LINE_LIST overrides the packaged reference. Output: planetes_wave.fits."

    def __init__(self) -> None:
        super().__init__()
        self.parameters = ui.ParameterList()

    def run(self, frameset: ui.FrameSet, settings: dict[str, Any]) -> ui.FrameSet:
        """Read the lamp and static calibration, fit wavelengths, and register FITS."""
        if settings:
            raise ValueError(f"Unknown wave parameters: {', '.join(settings)}")
        frames = ui.FrameSet(list(frameset))
        if not any(frame.tag == "PLANETES_LINE_LIST" for frame in frames):
            frames.append(
                ui.Frame(
                    file=str(wavelength_reference_path()), tag="PLANETES_LINE_LIST"
                )
            )
        inputs, used = select_inputs(
            frames, ["WAVE_RAW", "WAVE_DARK", "PLANETES_FLAT", "PLANETES_LINE_LIST"]
        )
        output = new_output("planetes_wave.fits")
        flat = read_product(inputs["PLANETES_FLAT"], FlatProduct)
        image = frame_mean(iter_cube(inputs["WAVE_RAW"])) - frame_mean(
            iter_cube(inputs["WAVE_DARK"])
        )
        product = compute_wave(
            image, flat, load_wavelength_lines(inputs["PLANETES_LINE_LIST"])
        )
        return save_product(output, product, used, self)
