"""PyCPL binding for the ordinary flat scientific task."""

from cpl import ui
from planetes import __version__
from planetes.flat import calibrate_flat_file
from planetes.models import DarkCalibration, BadPixelMap
from planetes.eso.common import (
    parameter_list,
    scientific_settings,
    select_inputs,
    save_product,
)


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
        self.parameters = parameter_list("flat", calibrate_flat_file, ["profile_width"])

    def run(self, frameset, settings):
        options = scientific_settings(self, settings)
        paths, cal, used = select_inputs(
            frameset, {"DARK": DarkCalibration, "BAD": BadPixelMap, "FLAT_RAW": None}
        )
        product = calibrate_flat_file(
            paths["FLAT_RAW"], cal["DARK"], cal["BAD"], **options
        )
        return save_product("planetes_flat.fits", product, used, self)
