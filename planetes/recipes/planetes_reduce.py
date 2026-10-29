"""SOF, CPL parameters and DFS adapter for the reduce step."""

from cpl import ui

from planetes import __version__
from planetes.qc import measure_quality
from planetes.io.fits import read_product
from planetes.products import Dark, Flat, P2VM, Wave
from planetes.recipes.dfs import read_sof, save_products
from planetes.recipes.logging import recipe_logging
from planetes.recipes.parameters import make_parameters, read_parameters
from planetes.reduction import reduce_science

PARAMETERS = {
    "gd_search_min_m": (
        "planetes.reduce.gd-search-min-m",
        "gd-search-min-m",
        "Coarse science GD lower initialization [m]",
    ),
    "gd_search_max_m": (
        "planetes.reduce.gd-search-max-m",
        "gd-search-max-m",
        "Coarse science GD upper initialization [m]",
    ),
}


class PlanetesReduce(ui.PyRecipe):
    """Execute the same reduce function available to Python callers."""

    # fmt: off
    _name      = "planetes_reduce"
    _version   = __version__
    _author    = "PLANETES contributors"
    _email     = ""
    _copyright = "PLANETES contributors"
    _synopsis  = "Extract per-frame instrumental interferometric observables"
    # fmt: on
    _description = (
        "Use explicit DARK, FLAT, WAVE and P2VM calibrations to extract every "
        "SINGLE_SCI_RAW frame. Return instrumental fluxes, visibility amplitude, "
        "GD and differential phase."
    )

    def __init__(self):
        super().__init__()
        self.parameters = make_parameters(PARAMETERS, reduce_science)

    @recipe_logging()
    def run(self, frameset, settings):
        options = read_parameters(self, settings, PARAMETERS)
        paths, used = read_sof(
            frameset, {"SINGLE_SCI_RAW": 1, "DARK": 1, "FLAT": 1, "WAVE": 1, "P2VM": 1}
        )
        dark = read_product(paths["DARK"][0], Dark)
        flat = read_product(paths["FLAT"][0], Flat)
        wave = read_product(paths["WAVE"][0], Wave)
        p2vm = read_product(paths["P2VM"][0], P2VM)
        product = reduce_science(
            paths["SINGLE_SCI_RAW"][0], dark, flat, wave, p2vm, **options
        )
        return save_products(
            [("planetes_reduced.fits", product, measure_quality(product))],
            used,
            self,
            reference=paths["SINGLE_SCI_RAW"][0],
        )
