"""SOF, CPL parameters and DFS adapter for the dark step."""

from cpl import ui

from planetes import __version__
from planetes.dark import calibrate_dark_file
from planetes.recipes.dfs import (
    save_products,
    select_inputs,
)
from planetes.recipes.parameters import make_parameters, read_parameters

PARAMETERS = {
    "bad_pixel_threshold": (
        "planetes.calib.bad-dark-threshold",
        "bad-dark-threshold",
        "Positive multiplier in the retained dark bad-pixel tests",
    )
}


class PlanetesDark(ui.PyRecipe):
    """Execute the same dark function available to Python callers."""

    _name = "planetes_dark"
    _version = __version__
    _author = "PLANETES contributors"
    _email = ""
    _copyright = "PLANETES contributors"
    _synopsis = "Measure detector dark statistics and bad pixels"
    _description = "Measure the temporal mean and RMS of a DARK_RAW stack and return DARK and BAD products."

    def __init__(self):
        super().__init__()
        self.parameters = make_parameters(PARAMETERS, calibrate_dark_file)

    def run(self, frameset, settings):
        options = read_parameters(self, settings, PARAMETERS)
        paths, _, used = select_inputs(frameset, {"DARK_RAW": None})
        dark, bad = calibrate_dark_file(paths["DARK_RAW"], **options)
        return save_products(
            [("planetes_dark.fits", dark), ("planetes_bad.fits", bad)], used, self
        )
