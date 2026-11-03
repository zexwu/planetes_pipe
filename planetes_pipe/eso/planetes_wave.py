"""SOF, CPL parameters and DFS adapter for the wave step."""
from planetes_pipe.io import wave as wave_io
from planetes_pipe.io import flat as flat_io
from planetes_pipe.io import raw as raw_io
from planetes_pipe.io import dark as dark_io

from cpl import ui

from planetes_pipe import __version__
from planetes_pipe.processing.wave import measure_quality
from planetes_pipe.eso.dfs import read_sof, save_products
from planetes_pipe.eso.logging import recipe_logging
from planetes_pipe.eso.parameters import make_parameters, read_parameters
from planetes_pipe.processing.wave import make_wave

PARAMETERS = {}


class PlanetesWave(ui.PyRecipe):
    """Execute the same wave function available to Python callers."""

    # fmt: off
    _name      = "planetes_wave"
    _version   = __version__
    _author    = "PLANETES contributors"
    _email     = ""
    _copyright = "PLANETES contributors"
    _synopsis  = "Fit the detector wavelength calibration"
    # fmt: on
    _description = (
        "Use WAVELAMP_RAW, its DARK, FLAT geometry and built-in laboratory lines to fit "
        "wavelengths in metres."
    )

    def __init__(self):
        super().__init__()
        self.parameters = make_parameters(PARAMETERS, make_wave)

    @recipe_logging()
    def run(self, frameset, settings):
        options = read_parameters(self, settings, PARAMETERS)
        paths, used = read_sof(
            frameset, {"WAVELAMP_RAW": 1, "DARK": 1, "FLAT": 1}
        )
        dark = dark_io.read(paths['DARK'][0])
        flat = flat_io.read(paths["FLAT"][0])
        with raw_io.read(
            paths["WAVELAMP_RAW"][0], category="WAVELAMP_RAW"
        ) as raw_input:
            product = make_wave(raw_input, dark, flat, **options)
        return save_products(
            [("planetes_wave.fits", product, measure_quality(product), wave_io.encode)],
            used,
            self,
            reference=paths["WAVELAMP_RAW"][0],
        )
