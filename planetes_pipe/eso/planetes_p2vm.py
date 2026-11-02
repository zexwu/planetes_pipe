"""SOF, CPL parameters and DFS adapter for the p2vm step."""
from planetes_pipe.io import p2vm as p2vm_io
from planetes_pipe.io import flat as flat_io
from planetes_pipe.io import wave as wave_io
from planetes_pipe.io import raw as raw_io
from planetes_pipe.io import dark as dark_io

from cpl import ui

from planetes_pipe import __version__
from planetes_pipe.processing.p2vm import wavelength_grid, measure_quality
from planetes_pipe.model.instrument import check_setup
from planetes_pipe.io.raw import read_raw_metadata
from planetes_pipe.processing.extraction import extract_spectra, prepare_extraction
from planetes_pipe.processing.p2vm import make_p2vm
from planetes_pipe.model.products import Dark, Flat, Wave
from planetes_pipe.eso.dfs import read_sof, save_products, read_shutter_id
from planetes_pipe.eso.logging import recipe_logging
from planetes_pipe.eso.parameters import make_parameters, read_parameters

PARAMETERS = {
    "keep_diagnostics": (
        "planetes.calib.keep-diagnostics",
        "keep-diagnostics",
        "Retain P2VM scan and fit diagnostics in the calibration product",
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

    @recipe_logging()
    def run(self, frameset, settings):
        options = read_parameters(self, settings, PARAMETERS)
        paths, used = read_sof(
            frameset,
            {"DARK": 1, "FLAT": 1, "WAVE": 1, "FLAT_RAW": 4, "P2VM_RAW": 6},
            optional=("PHASE_REFERENCE", "PHASE_DARK_CALIB"),
        )
        dark = dark_io.read(paths['DARK'][0])
        flat = flat_io.read(paths["FLAT"][0])
        wave = wave_io.read(paths["WAVE"][0])
        reference = None
        if (
            self.parameters["planetes.calib.phase-correction"].value
            and "PHASE_REFERENCE" in paths
        ):
            path = paths["PHASE_REFERENCE"][0]
            selected_dark = (
                dark_io.read(paths['PHASE_DARK_CALIB'][0])
                if "PHASE_DARK_CALIB" in paths
                else dark
            )
            raw = read_raw_metadata(path, category="SINGLE_SCI_RAW")
            check_setup(raw.setup, selected_dark.setup, exposure=True, role="DARK")
            check_setup(raw.setup, flat.setup, exposure=False, role="FLAT")
            grid = wavelength_grid(flat, wave)
            extractor = prepare_extraction(flat, wave, grid)
            with raw_io.read(path, category="SINGLE_SCI_RAW") as raw_input:
                reference = extract_spectra(raw_input, extractor, selected_dark)
        else:
            used = ui.FrameSet(
                [
                    frame
                    for frame in used
                    if frame.tag not in ("PHASE_REFERENCE", "PHASE_DARK_CALIB")
                ]
            )
        flats, scans = paths["FLAT_RAW"], paths["P2VM_RAW"]
        with (
            raw_io.read_many(flats, category="FLAT_RAW") as telescope_flats_raw,
            raw_io.read_many(scans, category="P2VM_RAW") as baseline_cubes_raw,
        ):
            product = make_p2vm(
                flat,
                wave,
                dark,
                telescope_flats=telescope_flats_raw,
                baseline_cubes=baseline_cubes_raw,
                phase_reference=reference,
                **options,
            )
        return save_products(
            [("planetes_p2vm.fits", product, measure_quality(product), p2vm_io.encode)],
            used,
            self,
            reference=next(p for p in scans if read_shutter_id(p, "P2VM_RAW") == "43"),
        )
