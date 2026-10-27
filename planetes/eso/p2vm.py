"""PyCPL binding for the ordinary p2vm scientific task."""

from cpl import ui
from planetes import __version__
from planetes.p2vm import calibrate_p2vm_file
from planetes.models import (
    DarkCalibration,
    DarkImage,
    FlatCalibration,
    WavelengthCalibration,
)
from planetes.instrument import TELESCOPES, BASELINES
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
    _description = "Explicit SOF roles and shared scientific defaults; schema-2 instrumental products."

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
            "DARK_CALIB": (DarkCalibration, DarkImage),
            "PLANETES_FLAT": FlatCalibration,
            "PLANETES_WAVE": WavelengthCalibration,
            **{f"TEL_FLAT_{n}": None for n in TELESCOPES},
            **{f"P2VM_RAW_{n}": None for n in BASELINES},
        }
        paths, cal, used = select_inputs(
            frameset,
            roles,
            {"PHASE_REFERENCE": None, "PHASE_DARK_CALIB": (DarkCalibration, DarkImage)},
        )
        product = calibrate_p2vm_file(
            cal["PLANETES_FLAT"],
            cal["PLANETES_WAVE"],
            cal["DARK_CALIB"],
            telescope_flats={n: paths[f"TEL_FLAT_{n}"] for n in TELESCOPES},
            baseline_cubes={n: paths[f"P2VM_RAW_{n}"] for n in BASELINES},
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
