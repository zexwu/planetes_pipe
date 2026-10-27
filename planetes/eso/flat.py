"""PyCPL binding for the ordinary flat scientific task."""

from cpl import ui
from planetes import __version__
from planetes.flat import calibrate_flat_file
from planetes.models import DarkCalibration
from planetes.instrument import TELESCOPES
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
    _description = "Explicit SOF roles and shared scientific defaults; schema-2 instrumental products."

    def __init__(self):
        super().__init__()
        self.parameters = parameter_list("flat", calibrate_flat_file, ["profile_width"])

    def run(self, frameset, settings):
        options = scientific_settings(self, settings)
        roles = {
            "DARK_CALIB": DarkCalibration,
            **{f"TEL_FLAT_{n}": None for n in TELESCOPES},
        }
        paths, cal, used = select_inputs(frameset, roles)
        product = calibrate_flat_file(
            {n: paths[f"TEL_FLAT_{n}"] for n in TELESCOPES},
            cal["DARK_CALIB"],
            **options,
        )
        return save_product("planetes_flat.fits", product, used, self)
