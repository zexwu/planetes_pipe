"""SOF, CPL parameters and DFS adapter for the flat step."""

from cpl import ui

from planetes import __version__
from planetes.io.fits import read_product
from planetes.flat import make_flat
from planetes.products import Bad, Dark
from planetes.recipes.dfs import read_sof, save_products, read_shutter_id
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

    # fmt: off
    _name      = "planetes_flat"
    _version   = __version__
    _author    = "PLANETES contributors"
    _email     = ""
    _copyright = "PLANETES contributors"
    _synopsis  = "Calibrate spectral traces and extraction geometry"
    # fmt: on
    _description = (
        "Use four FLAT_RAW shutter configurations with DARK and BAD to measure "
        "response and extraction geometry."
    )

    def __init__(self):
        super().__init__()
        self.parameters = make_parameters(PARAMETERS, make_flat)

    def run(self, frameset, settings):
        options = read_parameters(self, settings, PARAMETERS)
        paths, used = read_sof(frameset, {"DARK": 1, "BAD": 1, "FLAT_RAW": 4})
        dark = read_product(paths["DARK"][0], Dark)
        bad = read_product(paths["BAD"][0], Bad)
        flats = {read_shutter_id(path, "FLAT_RAW"): path for path in paths["FLAT_RAW"]}
        product = make_flat(flats, dark, bad, **options)
        return save_products([("planetes_flat.fits", product)], used, self)
