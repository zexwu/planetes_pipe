"""SOF, CPL parameters and DFS adapter for the dark step."""

from cpl import ui

from planetes import __version__
from planetes.qc import measure_quality
from planetes.dark import make_dark_products
from planetes.recipes.dfs import read_sof, save_products
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

    def run(self, frameset, settings):
        options = read_parameters(self, settings, PARAMETERS)
        paths, used = read_sof(frameset, {"DARK_RAW": 1})
        dark, bad = make_dark_products(paths["DARK_RAW"][0], **options)
        return save_products(
            [
                ("planetes_dark.fits", dark, measure_quality(dark)),
                ("planetes_bad.fits", bad, measure_quality(bad)),
            ],
            used,
            self,
        )
