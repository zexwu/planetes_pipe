"""SOF, CPL parameters and DFS adapter for the wave step."""

from cpl import ui

from planetes import __version__
from planetes.products import (
    DarkCalibration,
    FlatCalibration,
    WavelengthReference,
)
from planetes.recipes.dfs import (
    save_product,
    select_inputs,
)
from planetes.recipes.parameters import make_parameters, read_parameters
from planetes.wavelength import (
    calibrate_wavelength_file,
    wavelength_reference_path,
)

PARAMETERS = {}


class PlanetesWave(ui.PyRecipe):
    """Execute the same wave function available to Python callers."""

    _name = "planetes_wave"
    _version = __version__
    _author = "PLANETES contributors"
    _email = ""
    _copyright = "PLANETES contributors"
    _synopsis = "Fit the detector wavelength calibration"
    _description = "Use WAVELAMP_RAW, its DARK, FLAT geometry and LINE_LIST to fit wavelengths in metres."

    def __init__(self):
        super().__init__()
        self.parameters = make_parameters(PARAMETERS, calibrate_wavelength_file)

    def run(self, frameset, settings):
        options = read_parameters(self, settings, PARAMETERS)
        frames = ui.FrameSet(list(frameset))
        if not any(frame.tag == "LINE_LIST" for frame in frames):
            frames.append(
                ui.Frame(file=str(wavelength_reference_path()), tag="LINE_LIST")
            )
        paths, cal, used = select_inputs(
            frames,
            {
                "WAVELAMP_RAW": None,
                "DARK": DarkCalibration,
                "FLAT": FlatCalibration,
                "LINE_LIST": WavelengthReference,
            },
        )
        product = calibrate_wavelength_file(
            paths["WAVELAMP_RAW"],
            cal["DARK"],
            cal["FLAT"],
            reference=cal["LINE_LIST"],
            **options,
        )
        return save_product("planetes_wave.fits", product, used, self, include_qc=True)
