"""PyCPL binding for the ordinary dark scientific task."""

from cpl import ui

from planetes import __version__
from planetes.recipes.dfs import (
    save_products,
    select_inputs,
)
from planetes.recipes.parameters import make_parameters, read_parameters
from planetes.tasks import calibrate_dark_file


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
        self.parameters = make_parameters("dark", calibrate_dark_file)

    def run(self, frameset, settings):
        options = read_parameters(self, settings)
        paths, _, used = select_inputs(frameset, {"DARK_RAW": None})
        dark, bad = calibrate_dark_file(paths["DARK_RAW"], **options)
        return save_products(
            [("planetes_dark.fits", dark), ("planetes_bad.fits", bad)], used, self
        )
