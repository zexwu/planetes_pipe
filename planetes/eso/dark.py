"""PyCPL binding for the ordinary dark scientific task."""

from cpl import ui
from planetes import __version__
from planetes.dark import calibrate_dark_file
from planetes.eso.common import (
    parameter_list,
    scientific_settings,
    select_inputs,
    save_product,
)


class PlanetesDark(ui.PyRecipe):
    """Execute the same dark function available to Python callers."""

    _name = "planetes_dark"
    _version = __version__
    _author = "PLANETES contributors"
    _email = ""
    _copyright = "PLANETES contributors"
    _synopsis = "PLANETES dark calibration/reduction"
    _description = "Explicit SOF roles and shared scientific defaults; schema-2 instrumental products."

    def __init__(self):
        super().__init__()
        self.parameters = parameter_list("dark", calibrate_dark_file, ["bad_threshold"])

    def run(self, frameset, settings):
        options = scientific_settings(self, settings)
        paths, _, used = select_inputs(frameset, {"DARK_RAW": None})
        product = calibrate_dark_file(paths["DARK_RAW"], **options)
        return save_product("planetes_dark.fits", product, used, self)
