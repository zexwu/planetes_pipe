"""Declare a step function's options as CPL parameters and read them back."""

from inspect import signature

from cpl import ui

from planetes_pipe.steps import PARAMETERS

STATIC_NAME = "planetes.dfs.static-name"


def _parameter(name, description, default):
    parameter = ui.ParameterValue(
        name=name, context=name.rsplit(".", 1)[0], description=description, default=default
    )
    parameter.cli_alias = name.rsplit(".", 1)[1]
    return parameter


def make_parameters(step, context: str, *extra):
    """CPL parameters for the step's options, defaults taken from its signature.

    `extra` holds recipe-only (name, description, default) options; the common
    static-name option is appended last.
    """
    parameters = [
        _parameter(f"planetes.{context}.{PARAMETERS[name][0]}", PARAMETERS[name][1],
                   "" if item.default is None else item.default)
        for name, item in signature(step).parameters.items()
        if name in PARAMETERS
    ]
    parameters += [_parameter(*item) for item in extra]
    parameters.append(_parameter(
        STATIC_NAME, "Use fixed recipe output names instead of reference raw filenames", False
    ))
    return ui.ParameterList(parameters)


def read_parameters(recipe, settings) -> dict:
    """Apply this invocation's settings; return {option name: value}.

    Defaults are restored first so repeated calls do not inherit previous values.
    """
    for parameter in recipe.parameters:
        parameter.value = parameter.default
    for name, value in settings.items():
        recipe.parameters[name].value = value
    return {parameter.name.rsplit(".", 1)[1]: parameter.value for parameter in recipe.parameters}


def step_options(step, values: dict) -> dict:
    """Keyword arguments for the step function from {option name: value}."""
    return {name: values[PARAMETERS[name][0]]
            for name in signature(step).parameters if name in PARAMETERS}
