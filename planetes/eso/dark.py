"""PyCPL binding for the ordinary dark scientific task."""

from cpl import ui
from planetes import __version__
from planetes.dark import calibrate_dark_file
from planetes.eso.common import (
    parameter_list,
    scientific_settings,
    select_inputs,
    save_products,
)


class PlanetesDark(ui.PyRecipe):
    """Execute the same dark function available to Python callers."""

    _name = "planetes_dark"
    _version = __version__
    _author = "PLANETES contributors"
    _email = ""
    _copyright = "PLANETES contributors"
    _synopsis = "PLANETES dark calibration/reduction"
    _description = "Explicit SOF roles and shared scientific defaults; GRAVITY-derived categories and explicit PLANETES instrumental extensions."

    def __init__(self):
        super().__init__()
        self.parameters = parameter_list("dark", calibrate_dark_file, ["bad_threshold"])

    def run(self, frameset, settings):
        options = scientific_settings(self, settings)
        paths, _, used = select_inputs(frameset, {"DARK_RAW": None})
        dark, bad = calibrate_dark_file(paths["DARK_RAW"], **options)
        return save_products(
            [("planetes_dark.fits", dark), ("planetes_bad.fits", bad)], used, self
        )
