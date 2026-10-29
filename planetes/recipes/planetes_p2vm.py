"""SOF, CPL parameters and DFS adapter for the p2vm step."""

import numpy as np
from cpl import ui

from planetes import __version__
from planetes.io.fits import read_product
from planetes.io.raw import check_raw_calibrations, read_raw_metadata
from planetes.extraction import extract_spectra_from_file, prepare_extraction
from planetes.p2vm import make_p2vm
from planetes.products import Dark, Flat, Wave
from planetes.recipes.dfs import read_sof, save_products, read_shutter_id
from planetes.recipes.parameters import make_parameters, read_parameters

PARAMETERS = {
    "keep_diagnostics": (
        "planetes.calib.keep-diagnostics",
        "keep-diagnostics",
        "Retain P2VM scan and fit diagnostics in the calibration product",
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

    # fmt: off
    _name      = "planetes_p2vm"
    _version   = __version__
    _author    = "PLANETES contributors"
    _email     = ""
    _copyright = "PLANETES contributors"
    _synopsis  = "Calibrate the instrumental pixel-to-visibility response"
    # fmt: on
    _description = (
        "Use telescope and baseline sequences with explicit calibrations to fit "
        "the full 24-output response and inverse. An optional instrumental phase "
        "reference has its own selected DARK."
    )

    def __init__(self):
        super().__init__()
        self.parameters = make_parameters(PARAMETERS, make_p2vm)
        phase = ui.ParameterValue(
            name="planetes.calib.phase-correction",
            context="planetes.calib",
            description="Apply supplied phase reference",
            default=True,
        )
        phase.cli_alias = "phase-correction"
        self.parameters.append(phase)

    def run(self, frameset, settings):
        options = read_parameters(self, settings, PARAMETERS)
        paths, used = read_sof(
            frameset,
            {"DARK": 1, "FLAT": 1, "WAVE": 1, "FLAT_RAW": 4, "P2VM_RAW": 6},
            optional=("PHASE_REFERENCE", "PHASE_DARK_CALIB"),
        )
        dark = read_product(paths["DARK"][0], Dark)
        flat = read_product(paths["FLAT"][0], Flat)
        wave = read_product(paths["WAVE"][0], Wave)
        reference = None
        if (
            self.parameters["planetes.calib.phase-correction"].value
            and "PHASE_REFERENCE" in paths
        ):
            path = paths["PHASE_REFERENCE"][0]
            selected_dark = (
                read_product(paths["PHASE_DARK_CALIB"][0], Dark)
                if "PHASE_DARK_CALIB" in paths
                else dark
            )
            raw = read_raw_metadata(path, category="SINGLE_SCI_RAW")
            check_raw_calibrations(raw, selected_dark, flat)
            grid = np.linspace(
                options["wavelength_min_m"],
                options["wavelength_max_m"],
                options["wavelength_channels"],
            )
            extractor = prepare_extraction(flat, wave, grid)
            reference = extract_spectra_from_file(path, extractor, selected_dark)
        else:
            used = ui.FrameSet(
                [
                    frame
                    for frame in used
                    if frame.tag not in ("PHASE_REFERENCE", "PHASE_DARK_CALIB")
                ]
            )
        flats = {read_shutter_id(path, "FLAT_RAW"): path for path in paths["FLAT_RAW"]}
        scans = {read_shutter_id(path, "P2VM_RAW"): path for path in paths["P2VM_RAW"]}
        product = make_p2vm(
            flat,
            wave,
            dark,
            telescope_flats=flats,
            baseline_cubes=scans,
            phase_reference=reference,
            **options,
        )
        return save_products([("planetes_p2vm.fits", product)], used, self)
