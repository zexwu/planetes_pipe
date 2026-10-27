"""PyCPL binding for the ordinary reduce scientific task."""

from cpl import ui
from planetes import __version__
from planetes.reduction import reduce_file
from planetes.models import (
    DarkCalibration,
    DarkImage,
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
    _description = "Explicit SOF roles and shared scientific defaults; schema-2 instrumental products."

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
                "SCIENCE_RAW": None,
                "DARK_CALIB": (DarkCalibration, DarkImage),
                "PLANETES_FLAT": FlatCalibration,
                "PLANETES_WAVE": WavelengthCalibration,
                "PLANETES_P2VM": P2VMCalibration,
            },
        )
        product = reduce_file(
            paths["SCIENCE_RAW"],
            cal["DARK_CALIB"],
            cal["PLANETES_FLAT"],
            cal["PLANETES_WAVE"],
            cal["PLANETES_P2VM"],
            **options,
        )
        return save_product("planetes_reduced.fits", product, used, self)
