"""SOF, CPL parameters and DFS adapter for the flat step."""
from planetes_pipe.io import flat as flat_io
from planetes_pipe.io import raw as raw_io
from planetes_pipe.io import bad as bad_io
from planetes_pipe.io import dark as dark_io

from cpl import ui

from planetes_pipe import __version__
from planetes_pipe.processing.flat import measure_quality
from planetes_pipe.processing.flat import make_flat
from planetes_pipe.model.products import Bad, Dark
from planetes_pipe.eso.dfs import read_sof, save_products, read_shutter_id
from planetes_pipe.eso.logging import recipe_logging
from planetes_pipe.eso.parameters import make_parameters, read_parameters

PARAMETERS = {
    "aperture_width_pixels": (
        "planetes.calib.profile-width",
        "profile-width",
        "Full extraction aperture in detector pixels",
    ),
    "metrology_trace": (
        "planetes.calib.metrology-trace", "metrology-trace",
        "Detect 25 traces and omit the middle metrology trace",
    ),
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

    @recipe_logging()
    def run(self, frameset, settings):
        options = read_parameters(self, settings, PARAMETERS)
        paths, used = read_sof(frameset, {"DARK": 1, "BAD": 1, "FLAT_RAW": 4})
        dark = dark_io.read(paths['DARK'][0])
        bad = bad_io.read(paths['BAD'][0])
        flats = paths["FLAT_RAW"]
        with raw_io.read_many(flats, category="FLAT_RAW") as flats_raw:
            product = make_flat(flats_raw, dark, bad, **options)
        return save_products(
            [("planetes_flat.fits", product, measure_quality(product), flat_io.encode)],
            used,
            self,
            reference=next(p for p in flats if read_shutter_id(p, "FLAT_RAW") == "4"),
        )
