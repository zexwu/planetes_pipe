"""PyEsoRex recipes. Each reads its SOF and options and calls one step's run().

CPL is used only in this file: parameters, logging, product signing (DATAMD5,
PIPEFILE) and registration. PyEsoRex imports every module in the recipe
directory and instantiates each PyRecipe subclass it finds.
"""

import logging
from contextlib import contextmanager
from inspect import signature
from pathlib import Path

from cpl import ui
from cpl.core import Msg

from planetes_pipe import __version__, log
from planetes_pipe.logs import StepFormatter
from planetes_pipe.steps import dark, flat, p2vm, science, wave

AUTHOR = "PLANETES contributors"
TABLE_CATEGORIES = ("WAVE", "P2VM", "SINGLE_SCI_P2VMRED")


# ---------------------------------------------------------------- parameters


def _parameter(name, description, default):
    parameter = ui.ParameterValue(
        name=name, context=name.rsplit(".", 1)[0], description=description, default=default
    )
    parameter.cli_alias = name.rsplit(".", 1)[1]
    return parameter


# run() option -> help text. The CPL parameter is planetes.<context>.<option with dashes>
# and its default is the run() signature default.
HELP = {
    "bad_pixel_threshold": "Positive multiplier in the retained dark bad-pixel tests",
    "aperture_width_pixels": "Full extraction aperture in detector pixels",
    "metrology_trace": "Detect 25 traces and omit the middle metrology trace",
    "keep_diagnostics": "Retain P2VM scan and fit diagnostics in the calibration product",
    "gd_search_min_m": "Lower edge of the GD search [m] (science: where fringes are found)",
    "gd_search_max_m": "Upper edge of the GD search [m] (science: where fringes are found)",
    "wave_min_m": "Shortest wavelength of the P2VM band [m]; 0 keeps the whole common band",
    "wave_max_m": "Longest wavelength of the P2VM band [m]; 1 keeps the whole common band",
}


def make_parameters(run, context: str):
    """CPL parameters for run()'s options, with defaults from its signature."""
    return ui.ParameterList([
        _parameter(f"planetes.{context}.{name.replace('_', '-')}", HELP[name],
                   "" if item.default is None else item.default)
        for name, item in signature(run).parameters.items()
        if name in HELP
    ])


def read_parameters(recipe, settings) -> dict:
    """Apply this invocation's settings to the defaults; return {option name: value}."""
    for parameter in recipe.parameters:
        parameter.value = parameter.default
    for name, value in settings.items():
        recipe.parameters[name].value = value
    return {parameter.name.rsplit(".", 1)[1]: parameter.value for parameter in recipe.parameters}


def run_options(run, values: dict) -> dict:
    """Keyword arguments for run() from {option name: value}."""
    return {name: values[name.replace("_", "-")]
            for name in signature(run).parameters if name in HELP}


# ---------------------------------------------------------------- SOF and products


def read_sof(frameset, required: dict[str, int], optional=()) -> dict[str, list[Path]]:
    """Group SOF paths by tag, checking required counts and at most one optional."""
    paths = {}
    for frame in frameset:
        if frame.tag not in required and frame.tag not in optional:
            raise ValueError(f"Unexpected input tag: {frame.tag}")
        paths.setdefault(frame.tag, []).append(Path(frame.file))
    for tag, count in required.items():
        if len(paths.get(tag, [])) != count:
            raise ValueError(f"Expected {count} {tag} input(s), got {len(paths.get(tag, []))}")
    for tag in optional:
        if len(paths.get(tag, [])) > 1:
            raise ValueError(f"Expected at most one {tag} input")
    return paths


def publish(written: dict[str, Path]):
    """Sign (DATAMD5, PIPEFILE) and register written products as final PRODUCT frames."""
    products = ui.FrameSet()
    for category, path in written.items():
        frame = ui.Frame(
            file=str(path),
            tag=category,
            group=ui.Frame.FrameGroup.PRODUCT,
            level=ui.Frame.FrameLevel.FINAL,
            frameType=ui.Frame.FrameType.TABLE
            if category in TABLE_CATEGORIES
            else ui.Frame.FrameType.IMAGE,
        )
        ui.FrameSet([frame]).sign_products()
        products.append(frame)
    return products


# ---------------------------------------------------------------- logging


class _CPLHandler(logging.Handler):
    """Forward records without applying a second terminal or file verbosity filter.

    CPL prints the component (module.function) and the level; the message keeps the
    step tag, as on the console.
    """

    def __init__(self):
        super().__init__()
        self.setFormatter(StepFormatter("[%(step)s] %(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        if record.levelno >= logging.ERROR:
            send = Msg.error
        elif record.levelno >= logging.WARNING:
            send = Msg.warning
        elif record.levelno >= logging.INFO:
            send = Msg.info
        else:
            send = Msg.debug
        message = self.format(record)  # sets record.where
        send(record.where, message)


@contextmanager
def recipe_logging():
    """Route pipeline logging through CPL for one recipe invocation.

    PyEsoRex owns the terminal and logfile levels, so DEBUG and above are
    forwarded. Python handlers, level and propagation are restored on exit,
    even on failure. Also usable as a decorator on run().
    """
    handlers, level, propagate = log.handlers, log.level, log.propagate
    handler = _CPLHandler()
    log.handlers = [handler]
    log.setLevel(logging.DEBUG)
    log.propagate = False
    try:
        yield
    finally:
        log.handlers = handlers
        log.setLevel(level)
        log.propagate = propagate
        handler.close()


# ---------------------------------------------------------------- recipes


class PlanetesDark(ui.PyRecipe):
    _name, _version, _author, _email, _copyright = "planetes_dark", __version__, AUTHOR, "", AUTHOR
    _synopsis = "Measure detector dark statistics and bad pixels"
    _description = "Measure the temporal mean and RMS of a DARK_RAW stack; return DARK and BAD."

    def __init__(self):
        super().__init__()
        self.parameters = make_parameters(dark.run, "calib")

    @recipe_logging()
    def run(self, frameset, settings):
        values = read_parameters(self, settings)
        sof = read_sof(frameset, {"DARK_RAW": 1})
        options = run_options(dark.run, values)
        return publish(dark.run(sof["DARK_RAW"][0], output_dir=".", **options))


class PlanetesFlat(ui.PyRecipe):
    _name, _version, _author, _email, _copyright = "planetes_flat", __version__, AUTHOR, "", AUTHOR
    _synopsis = "Calibrate spectral traces and extraction geometry"
    _description = ("Use four FLAT_RAW shutter configurations with DARK and BAD to measure "
                    "response and extraction geometry.")

    def __init__(self):
        super().__init__()
        self.parameters = make_parameters(flat.run, "calib")

    @recipe_logging()
    def run(self, frameset, settings):
        values = read_parameters(self, settings)
        sof = read_sof(frameset, {"DARK": 1, "BAD": 1, "FLAT_RAW": 4})
        options = run_options(flat.run, values)
        written = flat.run(sof["DARK"][0], sof["BAD"][0], sof["FLAT_RAW"], output_dir=".",
                           **options)
        return publish(written)


class PlanetesWave(ui.PyRecipe):
    _name, _version, _author, _email, _copyright = "planetes_wave", __version__, AUTHOR, "", AUTHOR
    _synopsis = "Fit the detector wavelength calibration"
    _description = ("Use WAVELAMP_RAW, its DARK, FLAT geometry and built-in laboratory lines "
                    "to fit wavelengths in metres.")

    def __init__(self):
        super().__init__()
        self.parameters = make_parameters(wave.run, "calib")

    @recipe_logging()
    def run(self, frameset, settings):
        sof = read_sof(frameset, {"WAVELAMP_RAW": 1, "DARK": 1, "FLAT": 1})
        written = wave.run(sof["WAVELAMP_RAW"][0], sof["DARK"][0], sof["FLAT"][0], output_dir=".")
        return publish(written)


class PlanetesP2VM(ui.PyRecipe):
    _name, _version, _author, _email, _copyright = "planetes_p2vm", __version__, AUTHOR, "", AUTHOR
    _synopsis = "Calibrate the instrumental pixel-to-visibility response"
    _description = ("Use telescope and baseline sequences with explicit calibrations to fit "
                    "the full 24-output response and inverse. An optional instrumental phase "
                    "reference has its own selected DARK.")

    def __init__(self):
        super().__init__()
        self.parameters = make_parameters(p2vm.run, "calib")

    @recipe_logging()
    def run(self, frameset, settings):
        values = read_parameters(self, settings)
        sof = read_sof(frameset, {"DARK": 1, "FLAT": 1, "WAVE": 1, "FLAT_RAW": 4, "P2VM_RAW": 6},
                       optional=("PHASE_REFERENCE", "PHASE_DARK_CALIB"))
        options = run_options(p2vm.run, values)
        if "PHASE_REFERENCE" in sof:
            options["phase_file"] = sof["PHASE_REFERENCE"][0]
        if "PHASE_DARK_CALIB" in sof:
            options["phase_dark_file"] = sof["PHASE_DARK_CALIB"][0]
        written = p2vm.run(sof["DARK"][0], sof["FLAT"][0], sof["WAVE"][0], sof["FLAT_RAW"],
                           sof["P2VM_RAW"], output_dir=".", **options)
        return publish(written)


class PlanetesReduce(ui.PyRecipe):
    _name, _version, _author, _email, _copyright = "planetes_reduce", __version__, AUTHOR, "", AUTHOR
    _synopsis = "Extract per-frame instrumental interferometric observables"
    _description = ("Use explicit DARK, FLAT, WAVE and P2VM calibrations to extract every "
                    "SINGLE_SCI_RAW frame. Return instrumental fluxes, visibility amplitude, "
                    "GD and differential phase.")

    def __init__(self):
        super().__init__()
        self.parameters = make_parameters(science.run, "reduce")

    @recipe_logging()
    def run(self, frameset, settings):
        values = read_parameters(self, settings)
        sof = read_sof(frameset, {"SINGLE_SCI_RAW": 1, "DARK": 1, "FLAT": 1, "WAVE": 1, "P2VM": 1})
        options = run_options(science.run, values)
        written = science.run(sof["SINGLE_SCI_RAW"][0], sof["DARK"][0], sof["FLAT"][0],
                              sof["WAVE"][0], sof["P2VM"][0], output_dir=".", **options)
        return publish(written)
