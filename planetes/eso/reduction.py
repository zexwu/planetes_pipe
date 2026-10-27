"""PyCPL binding for the ordinary reduce scientific task."""

from cpl import ui
from planetes import __version__
from planetes.reduction import reduce_file
from planetes.models import (
    DarkCalibration,
    FlatCalibration,
    WavelengthCalibration,
    P2VMCalibration,
)
from planetes.eso.common import (
    parameter_list,
    scientific_settings,
    select_inputs,
    save_product,
)


class PlanetesReduce(ui.PyRecipe):
    """Execute the same reduce function available to Python callers."""

    _name = "planetes_reduce"
    _version = __version__
    _author = "PLANETES contributors"
    _email = ""
    _copyright = "PLANETES contributors"
    _synopsis = "PLANETES reduce calibration/reduction"
    _description = "Explicit SOF roles and shared scientific defaults; GRAVITY-derived categories and explicit PLANETES instrumental extensions."

    def __init__(self):
        super().__init__()
        self.parameters = parameter_list(
            "reduce", reduce_file, ["gd_search_min_um", "gd_search_max_um"]
        )

    def run(self, frameset, settings):
        options = scientific_settings(self, settings)
        paths, cal, used = select_inputs(
            frameset,
            {
                "SINGLE_SCI_RAW": None,
                "DARK": DarkCalibration,
                "FLAT": FlatCalibration,
                "WAVE": WavelengthCalibration,
                "P2VM": P2VMCalibration,
            },
        )
        product = reduce_file(
            paths["SINGLE_SCI_RAW"],
            cal["DARK"],
            cal["FLAT"],
            cal["WAVE"],
            cal["P2VM"],
            **options,
        )
        return save_product("planetes_reduced.fits", product, used, self)
