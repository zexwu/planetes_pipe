"""Translate Python science parameters to the documented CPL names and aliases."""

from inspect import signature

from cpl import ui


def make_parameters(definitions, function):
    """Declare CPL parameters using the scientific function's existing defaults.

    Parameters
    ----------
    definitions : mapping
        Python keyword -> (canonical CPL name, CLI alias, help description).
    function : callable
        Scientific step whose signature defines each keyword's default value.

    Returns
    -------
    parameters : cpl.ui.ParameterList
        Executor-facing parameters with explicit aliases and shared Python defaults.
    """
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
    """Reset a recipe's parameters and translate the settings for its scientific step.

    Parameters
    ----------
    recipe : cpl.ui.PyRecipe
        Recipe holding the declared ParameterList.
    settings : mapping
        Canonical CPL parameter name -> value for this invocation.
    definitions : mapping
        Python keyword -> (canonical CPL name, CLI alias, help description).

    Returns
    -------
    options : dict
        Scientific keyword arguments ready for the step call. Recipe-only options
        remain in recipe.parameters and are handled explicitly by the recipe.

    Notes
    -----
    Reset defaults before applying settings so repeated calls do not inherit
    previous values. Logging is configured separately for the whole invocation.
    """
    for parameter in recipe.parameters:
        parameter.value = parameter.default
    for name, value in settings.items():
        recipe.parameters[name].value = value
    return {name: recipe.parameters[definitions[name][0]].value for name in definitions}
