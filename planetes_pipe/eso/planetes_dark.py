"""SOF, CPL parameters and DFS adapter for the dark step."""
from planetes_pipe.io import bad as bad_io
from planetes_pipe.io import dark as dark_io
from planetes_pipe.io import raw as raw_io

from cpl import ui

from planetes_pipe import __version__
from planetes_pipe.processing.dark import measure_quality
from planetes_pipe.processing.dark import make_dark_products
from planetes_pipe.eso.dfs import read_sof, save_products
from planetes_pipe.eso.logging import recipe_logging
from planetes_pipe.eso.parameters import make_parameters, read_parameters

PARAMETERS = {
    "bad_pixel_threshold": (
        "planetes.calib.bad-dark-threshold",
        "bad-dark-threshold",
        "Positive multiplier in the retained dark bad-pixel tests",
    )
}


class PlanetesDark(ui.PyRecipe):
    """Execute the same dark function available to Python callers."""

    # fmt: off
    _name      = "planetes_dark"
    _version   = __version__
    _author    = "PLANETES contributors"
    _email     = ""
    _copyright = "PLANETES contributors"
    _synopsis  = "Measure detector dark statistics and bad pixels"
    # fmt: on
    _description = (
        "Measure the temporal mean and RMS of a DARK_RAW stack and return DARK "
        "and BAD products."
    )

    def __init__(self):
        super().__init__()
        self.parameters = make_parameters(PARAMETERS, make_dark_products)

    @recipe_logging()
    def run(self, frameset, settings):
        options = read_parameters(self, settings, PARAMETERS)
        paths, used = read_sof(frameset, {"DARK_RAW": 1})
        with raw_io.read(paths['DARK_RAW'][0], category="DARK_RAW") as raw:
            dark, bad = make_dark_products(raw, **options)
        return save_products(
            [
                ("planetes_dark.fits", dark, measure_quality(dark), dark_io.encode),
                ("planetes_bad.fits", bad, measure_quality(bad), bad_io.encode),
            ],
            used,
            self,
            reference=paths["DARK_RAW"][0],
        )
