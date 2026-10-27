"""PyCPL binding for the ordinary wave scientific task."""

from cpl import ui
from planetes import __version__
from planetes.wavelength import calibrate_wavelength_file
from planetes.models import (
    DarkCalibration,
    DarkImage,
    FlatCalibration,
    WavelengthReference,
)
from planetes.eso.common import (
    parameter_list,
    scientific_settings,
    select_inputs,
    save_product,
)
from planetes.wavelength import wavelength_reference_path


class PlanetesWave(ui.PyRecipe):
    """Execute the same wave function available to Python callers."""

    _name = "planetes_wave"
    _version = __version__
    _author = "PLANETES contributors"
    _email = ""
    _copyright = "PLANETES contributors"
    _synopsis = "PLANETES wave calibration/reduction"
    _description = "Explicit SOF roles and shared scientific defaults; schema-2 instrumental products."

    def __init__(self):
        super().__init__()
        self.parameters = parameter_list("wave", calibrate_wavelength_file, [])

    def run(self, frameset, settings):
        options = scientific_settings(self, settings)
        frames = ui.FrameSet(list(frameset))
        if not any(frame.tag == "PLANETES_LINE_LIST" for frame in frames):
            frames.append(
                ui.Frame(
                    file=str(wavelength_reference_path()), tag="PLANETES_LINE_LIST"
                )
            )
        paths, cal, used = select_inputs(
            frames,
            {
                "WAVE_RAW": None,
                "DARK_CALIB": (DarkCalibration, DarkImage),
                "PLANETES_FLAT": FlatCalibration,
                "PLANETES_LINE_LIST": WavelengthReference,
            },
        )
        product = calibrate_wavelength_file(
            paths["WAVE_RAW"],
            cal["DARK_CALIB"],
            cal["PLANETES_FLAT"],
            reference=cal["PLANETES_LINE_LIST"],
            **options,
        )
        return save_product("planetes_wave.fits", product, used, self)
