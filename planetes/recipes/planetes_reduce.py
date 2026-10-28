"""SOF, CPL parameters and DFS adapter for the reduce step."""

from cpl import ui

from planetes import __version__
from planetes.products import (
    Dark,
    Flat,
    P2VM,
    Wave,
)
from planetes.recipes.dfs import (
    save_product,
    load_recipe_inputs,
)
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

    _name = "planetes_reduce"
    _version = __version__
    _author = "PLANETES contributors"
    _email = ""
    _copyright = "PLANETES contributors"
    _synopsis = "Extract per-frame instrumental interferometric observables"
    _description = "Use explicit DARK, FLAT, WAVE and P2VM calibrations to extract every SINGLE_SCI_RAW frame. Return instrumental fluxes, visibility amplitude, GD and differential phase."

    def __init__(self):
        super().__init__()
        self.parameters = make_parameters(PARAMETERS, reduce_science)

    def run(self, frameset, settings):
        options = read_parameters(self, settings, PARAMETERS)
        paths, cal, used = load_recipe_inputs(
            frameset,
            {
                "SINGLE_SCI_RAW": None,
                "DARK": Dark,
                "FLAT": Flat,
                "WAVE": Wave,
                "P2VM": P2VM,
            },
        )
        product = reduce_science(
            paths["SINGLE_SCI_RAW"],
            cal["DARK"],
            cal["FLAT"],
            cal["WAVE"],
            cal["P2VM"],
            **options,
        )
        return save_product(
            "planetes_reduced.fits", product, used, self, include_qc=True
        )
