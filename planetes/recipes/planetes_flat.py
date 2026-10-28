"""SOF, CPL parameters and DFS adapter for the flat step."""

from cpl import ui

from planetes import __version__
from planetes.flat import make_flat
from planetes.products import Bad, Dark
from planetes.recipes.dfs import (
    save_product,
    load_recipe_inputs,
)
from planetes.recipes.parameters import make_parameters, read_parameters

PARAMETERS = {
    "aperture_width_pixels": (
        "planetes.calib.profile-width",
        "profile-width",
        "Full extraction aperture in detector pixels",
    )
}


class PlanetesFlat(ui.PyRecipe):
    """Execute the same flat function available to Python callers."""

    _name = "planetes_flat"
    _version = __version__
    _author = "PLANETES contributors"
    _email = ""
    _copyright = "PLANETES contributors"
    _synopsis = "Calibrate spectral traces and extraction geometry"
    _description = "Use four FLAT_RAW shutter configurations with DARK and BAD to measure response and extraction geometry."

    def __init__(self):
        super().__init__()
        self.parameters = make_parameters(PARAMETERS, make_flat)

    def run(self, frameset, settings):
        options = read_parameters(self, settings, PARAMETERS)
        paths, cal, used = load_recipe_inputs(
            frameset,
            {"DARK": Dark, "BAD": Bad, "FLAT_RAW": None},
            multiple={"FLAT_RAW": 4},
        )
        product = make_flat(paths["FLAT_RAW"], cal["DARK"], cal["BAD"], **options)
        return save_product("planetes_flat.fits", product, used, self, include_qc=True)
