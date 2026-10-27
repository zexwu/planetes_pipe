"""Translate Python science parameters to the documented CPL names and aliases."""

from inspect import signature

from cpl import ui

from planetes import configure_logging

# Canonical names follow gravi_dfs.c. Values/defaults belong to our science functions.
CALIB_PARAMETERS = {
    "bad_pixel_threshold": (
        "planetes.calib.bad-dark-threshold",
        "bad-dark-threshold",
        "Positive multiplier in the retained dark bad-pixel tests",
    ),
    "profile_width": (
        "planetes.calib.profile-width",
        "profile-width",
        "Full extraction aperture in detector pixels",
    ),
    "phase_correction": (
        "planetes.calib.phase-correction",
        "phase-correction",
        "Apply supplied phase reference",
    ),
    "wavelength_min_um": (
        "planetes.calib.wavelength-min-um",
        "wavelength-min-um",
        "P2VM target minimum [um]",
    ),
    "wavelength_max_um": (
        "planetes.calib.wavelength-max-um",
        "wavelength-max-um",
        "P2VM target maximum [um]",
    ),
    "wavelength_channels": (
        "planetes.calib.wavelength-channels",
        "wavelength-channels",
        "Integer target grid channels",
    ),
    "gd_search_min_um": (
        "planetes.calib.gd-search-min-um",
        "gd-search-min-um",
        "Coarse GD lower initialization [um]; Newton remains unconstrained",
    ),
    "gd_search_max_um": (
        "planetes.calib.gd-search-max-um",
        "gd-search-max-um",
        "Coarse GD upper initialization [um]; Newton remains unconstrained",
    ),
}
SCIENCE_PARAMETERS = {
    "gd_search_min_um": (
        "planetes.reduce.gd-search-min-um",
        "gd-search-min-um",
        "Coarse science GD lower initialization [um]",
    ),
    "gd_search_max_um": (
        "planetes.reduce.gd-search-max-um",
        "gd-search-max-um",
        "Coarse science GD upper initialization [um]",
    ),
}
RECIPE_OPTIONS = {
    "dark": ("bad_pixel_threshold",),
    "flat": ("profile_width",),
    "wave": (),
    "p2vm": (
        "phase_correction",
        "wavelength_min_um",
        "wavelength_max_um",
        "wavelength_channels",
        "gd_search_min_um",
        "gd_search_max_um",
    ),
    "reduce": ("gd_search_min_um", "gd_search_max_um"),
}


def make_parameters(stage, function):
    """Map explicitly declared Python keywords to CPL names, aliases and defaults."""
    definitions = SCIENCE_PARAMETERS if stage == "reduce" else CALIB_PARAMETERS
    defaults = signature(function).parameters
    parameters = []
    for name in RECIPE_OPTIONS[stage]:
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


def read_parameters(recipe, settings):
    """Reset each invocation and translate an explicit registry, not string spelling."""
    configure_logging()
    stage = recipe.name.removeprefix("planetes_")
    definitions = SCIENCE_PARAMETERS if stage == "reduce" else CALIB_PARAMETERS
    for parameter in recipe.parameters:
        parameter.value = parameter.default
    for name, value in settings.items():
        recipe.parameters[name].value = value
    return {
        name: recipe.parameters[definitions[name][0]].value
        for name in RECIPE_OPTIONS[stage]
    }
