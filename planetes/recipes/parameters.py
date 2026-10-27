"""Translate Python science parameters to the documented CPL names and aliases."""

from inspect import signature

from cpl import ui

from planetes import configure_logging


def make_parameters(definitions, function):
    """Map explicitly declared Python keywords to CPL names, aliases and defaults."""
    defaults = signature(function).parameters
    parameters = []
    for name in definitions:
        canonical, alias, description = definitions[name]
        parameter = ui.ParameterValue(
            name=canonical,
            context=canonical.rsplit(".", 1)[0],
            description=description,
            default=defaults[name].default,
        )
        parameter.cli_alias = alias
        parameters.append(parameter)
    return ui.ParameterList(parameters)


def read_parameters(recipe, settings, definitions):
    """Reset each invocation and translate an explicit registry, not string spelling."""
    configure_logging()
    for parameter in recipe.parameters:
        parameter.value = parameter.default
    for name, value in settings.items():
        recipe.parameters[name].value = value
    return {name: recipe.parameters[definitions[name][0]].value for name in definitions}
