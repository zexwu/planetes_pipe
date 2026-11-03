"""PyEsoRex recipes: each reads its SOF and options and calls one planetes_pipe.steps function.

PyEsoRex imports every module in the recipe directory and instantiates each
PyRecipe subclass it finds; these five are the PLANETES recipes.
"""

from cpl import ui

from planetes_pipe import __version__, steps
from planetes_pipe.eso.dfs import publish, read_sof
from planetes_pipe.eso.logging import recipe_logging
from planetes_pipe.eso.parameters import make_parameters, read_parameters, step_options

AUTHOR = "PLANETES contributors"
PHASE_CORRECTION = ("planetes.calib.phase-correction", "Apply supplied phase reference", True)


def _gain(text: str) -> dict[str, float] | None:
    """'43:-1,42:-1,...' -> {'43': -1.0, ...}; empty -> None."""
    if not text:
        return None
    return {key.strip(): float(value) for key, value in (i.split(":") for i in text.split(","))}


class PlanetesDark(ui.PyRecipe):
    _name, _version, _author, _email, _copyright = "planetes_dark", __version__, AUTHOR, "", AUTHOR
    _synopsis = "Measure detector dark statistics and bad pixels"
    _description = "Measure the temporal mean and RMS of a DARK_RAW stack; return DARK and BAD."

    def __init__(self):
        super().__init__()
        self.parameters = make_parameters(steps.dark, "calib")

    @recipe_logging()
    def run(self, frameset, settings):
        values = read_parameters(self, settings)
        sof = read_sof(frameset, {"DARK_RAW": 1})
        written = steps.dark(sof["DARK_RAW"][0], output_dir=".",
                             **step_options(steps.dark, values))
        return publish(written, static_name=values["static-name"])


class PlanetesFlat(ui.PyRecipe):
    _name, _version, _author, _email, _copyright = "planetes_flat", __version__, AUTHOR, "", AUTHOR
    _synopsis = "Calibrate spectral traces and extraction geometry"
    _description = ("Use four FLAT_RAW shutter configurations with DARK and BAD to measure "
                    "response and extraction geometry.")

    def __init__(self):
        super().__init__()
        self.parameters = make_parameters(steps.flat, "calib")

    @recipe_logging()
    def run(self, frameset, settings):
        values = read_parameters(self, settings)
        sof = read_sof(frameset, {"DARK": 1, "BAD": 1, "FLAT_RAW": 4})
        written = steps.flat(sof["DARK"][0], sof["BAD"][0], sof["FLAT_RAW"], output_dir=".",
                             **step_options(steps.flat, values))
        return publish(written, static_name=values["static-name"])


class PlanetesWave(ui.PyRecipe):
    _name, _version, _author, _email, _copyright = "planetes_wave", __version__, AUTHOR, "", AUTHOR
    _synopsis = "Fit the detector wavelength calibration"
    _description = ("Use WAVELAMP_RAW, its DARK, FLAT geometry and built-in laboratory lines "
                    "to fit wavelengths in metres.")

    def __init__(self):
        super().__init__()
        self.parameters = make_parameters(steps.wave, "calib")

    @recipe_logging()
    def run(self, frameset, settings):
        values = read_parameters(self, settings)
        sof = read_sof(frameset, {"WAVELAMP_RAW": 1, "DARK": 1, "FLAT": 1})
        written = steps.wave(sof["WAVELAMP_RAW"][0], sof["DARK"][0], sof["FLAT"][0],
                             output_dir=".")
        return publish(written, static_name=values["static-name"])


class PlanetesP2VM(ui.PyRecipe):
    _name, _version, _author, _email, _copyright = "planetes_p2vm", __version__, AUTHOR, "", AUTHOR
    _synopsis = "Calibrate the instrumental pixel-to-visibility response"
    _description = ("Use telescope and baseline sequences with explicit calibrations to fit "
                    "the full 24-output response and inverse. An optional instrumental phase "
                    "reference has its own selected DARK.")

    def __init__(self):
        super().__init__()
        self.parameters = make_parameters(steps.p2vm, "calib", PHASE_CORRECTION)

    @recipe_logging()
    def run(self, frameset, settings):
        values = read_parameters(self, settings)
        sof = read_sof(frameset, {"DARK": 1, "FLAT": 1, "WAVE": 1, "FLAT_RAW": 4, "P2VM_RAW": 6},
                       optional=("PHASE_REFERENCE", "PHASE_DARK_CALIB"))
        phase = values["phase-correction"] and "PHASE_REFERENCE" in sof
        options = step_options(steps.p2vm, values) | {"dl_gain": _gain(values["dl-gain"])}
        written = steps.p2vm(
            sof["DARK"][0], sof["FLAT"][0], sof["WAVE"][0], sof["FLAT_RAW"], sof["P2VM_RAW"],
            output_dir=".",
            phase_reference=sof["PHASE_REFERENCE"][0] if phase else None,
            phase_dark=sof["PHASE_DARK_CALIB"][0] if phase and "PHASE_DARK_CALIB" in sof
            else None,
            **options,
        )
        return publish(written, static_name=values["static-name"])


class PlanetesReduce(ui.PyRecipe):
    _name, _version, _author, _email, _copyright = "planetes_reduce", __version__, AUTHOR, "", AUTHOR
    _synopsis = "Extract per-frame instrumental interferometric observables"
    _description = ("Use explicit DARK, FLAT, WAVE and P2VM calibrations to extract every "
                    "SINGLE_SCI_RAW frame. Return instrumental fluxes, visibility amplitude, "
                    "GD and differential phase.")

    def __init__(self):
        super().__init__()
        self.parameters = make_parameters(steps.reduce, "reduce")

    @recipe_logging()
    def run(self, frameset, settings):
        values = read_parameters(self, settings)
        sof = read_sof(frameset, {"SINGLE_SCI_RAW": 1, "DARK": 1, "FLAT": 1, "WAVE": 1, "P2VM": 1})
        written = steps.reduce(sof["SINGLE_SCI_RAW"][0], sof["DARK"][0], sof["FLAT"][0],
                               sof["WAVE"][0], sof["P2VM"][0], output_dir=".",
                               **step_options(steps.reduce, values))
        return publish(written, static_name=values["static-name"])
