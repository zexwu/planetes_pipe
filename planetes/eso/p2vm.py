"""PyCPL binding for the ordinary p2vm scientific task."""

from cpl import ui
from planetes import __version__
from planetes.p2vm import calibrate_p2vm_file
from planetes.models import (
    DarkCalibration,
    FlatCalibration,
    WavelengthCalibration,
)
from planetes.eso.common import (
    parameter_list,
    scientific_settings,
    select_inputs,
    save_product,
)


class PlanetesP2VM(ui.PyRecipe):
    """Execute the same p2vm function available to Python callers."""

    _name = "planetes_p2vm"
    _version = __version__
    _author = "PLANETES contributors"
    _email = ""
    _copyright = "PLANETES contributors"
    _synopsis = "PLANETES p2vm calibration/reduction"
    _description = "Explicit SOF roles and shared scientific defaults; GRAVITY-derived categories and explicit PLANETES instrumental extensions."

    def __init__(self):
        super().__init__()
        self.parameters = parameter_list(
            "p2vm",
            calibrate_p2vm_file,
            [
                "phase_correction",
                "wavelength_min_um",
                "wavelength_max_um",
                "wavelength_channels",
                "gd_search_min_um",
                "gd_search_max_um",
            ],
        )

    def run(self, frameset, settings):
        options = scientific_settings(self, settings)
        roles = {
            "DARK": DarkCalibration,
            "FLAT": FlatCalibration,
            "WAVE": WavelengthCalibration,
            "FLAT_RAW": None,
            "P2VM_RAW": None,
        }
        paths, cal, used = select_inputs(
            frameset,
            roles,
            {"PHASE_REFERENCE": None, "PHASE_DARK_CALIB": DarkCalibration},
        )
        product = calibrate_p2vm_file(
            cal["FLAT"],
            cal["WAVE"],
            cal["DARK"],
            telescope_flats=paths["FLAT_RAW"],
            baseline_cubes=paths["P2VM_RAW"],
            phase_reference=paths.get("PHASE_REFERENCE"),
            phase_dark=cal.get("PHASE_DARK_CALIB"),
            **options,
        )
        if not product.provenance.parameters["phase_reference_applied"]:
            used = ui.FrameSet(
                [
                    frame
                    for frame in used
                    if frame.tag not in ("PHASE_REFERENCE", "PHASE_DARK_CALIB")
                ]
            )
        return save_product("planetes_p2vm.fits", product, used, self)
