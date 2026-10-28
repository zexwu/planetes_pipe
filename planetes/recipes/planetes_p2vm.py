"""SOF, CPL parameters and DFS adapter for the p2vm step."""

from cpl import ui

from planetes import __version__
from planetes.p2vm import calibrate_p2vm_file
from planetes.products import (
    Dark,
    Flat,
    Wave,
)
from planetes.recipes.dfs import (
    save_product,
    select_inputs,
)
from planetes.recipes.parameters import make_parameters, read_parameters

PARAMETERS = {
    "phase_correction": (
        "planetes.calib.phase-correction",
        "phase-correction",
        "Apply supplied phase reference",
    ),
    "wavelength_min_m": (
        "planetes.calib.wavelength-min-m",
        "wavelength-min-m",
        "P2VM target minimum [m]",
    ),
    "wavelength_max_m": (
        "planetes.calib.wavelength-max-m",
        "wavelength-max-m",
        "P2VM target maximum [m]",
    ),
    "wavelength_channels": (
        "planetes.calib.wavelength-channels",
        "wavelength-channels",
        "Integer target grid channels",
    ),
    "gd_search_min_m": (
        "planetes.calib.gd-search-min-m",
        "gd-search-min-m",
        "Coarse GD lower initialization [m]; Newton remains unconstrained",
    ),
    "gd_search_max_m": (
        "planetes.calib.gd-search-max-m",
        "gd-search-max-m",
        "Coarse GD upper initialization [m]; Newton remains unconstrained",
    ),
}


class PlanetesP2VM(ui.PyRecipe):
    """Execute the same p2vm function available to Python callers."""

    _name = "planetes_p2vm"
    _version = __version__
    _author = "PLANETES contributors"
    _email = ""
    _copyright = "PLANETES contributors"
    _synopsis = "Calibrate the instrumental pixel-to-visibility response"
    _description = "Use telescope and baseline sequences with explicit calibrations to fit the full 24-output response and inverse. An optional instrumental phase reference has its own selected DARK."

    def __init__(self):
        super().__init__()
        self.parameters = make_parameters(PARAMETERS, calibrate_p2vm_file)

    def run(self, frameset, settings):
        options = read_parameters(self, settings, PARAMETERS)
        roles = {
            "DARK": Dark,
            "FLAT": Flat,
            "WAVE": Wave,
            "FLAT_RAW": None,
            "P2VM_RAW": None,
        }
        paths, cal, used = select_inputs(
            frameset,
            roles,
            {"PHASE_REFERENCE": None, "PHASE_DARK_CALIB": Dark},
            multiple={"FLAT_RAW": 4, "P2VM_RAW": 6},
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
        if not (options["phase_correction"] and "PHASE_REFERENCE" in paths):
            used = ui.FrameSet(
                [
                    frame
                    for frame in used
                    if frame.tag not in ("PHASE_REFERENCE", "PHASE_DARK_CALIB")
                ]
            )
        return save_product("planetes_p2vm.fits", product, used, self, include_qc=True)
