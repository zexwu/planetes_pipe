"""PyCPL binding for the ordinary flat scientific task."""

from cpl import ui

from planetes import __version__
from planetes.flat import PARAMETERS, calibrate_flat_file
from planetes.products import BadPixelMap, DarkCalibration
from planetes.recipes.dfs import (
    save_product,
    select_inputs,
)
from planetes.recipes.parameters import make_parameters, read_parameters


class PlanetesFlat(ui.PyRecipe):
    """Execute the same flat function available to Python callers."""

    _name = "planetes_flat"
    _version = __version__
    _author = "PLANETES contributors"
    _email = ""
    _copyright = "PLANETES contributors"
    _synopsis = "PLANETES flat calibration/reduction"
    _description = "Explicit SOF roles and shared scientific defaults; GRAVITY-derived categories and explicit PLANETES instrumental extensions."

    def __init__(self):
        super().__init__()
        self.parameters = make_parameters(PARAMETERS, calibrate_flat_file)

    def run(self, frameset, settings):
        options = read_parameters(self, settings, PARAMETERS)
        paths, cal, used = select_inputs(
            frameset, {"DARK": DarkCalibration, "BAD": BadPixelMap, "FLAT_RAW": None}
        )
        product = calibrate_flat_file(
            paths["FLAT_RAW"], cal["DARK"], cal["BAD"], **options
        )
        return save_product("planetes_flat.fits", product, used, self)
