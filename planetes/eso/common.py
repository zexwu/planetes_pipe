"""CPL binding of explicit categories/parameters and verified product publication."""

from dataclasses import replace
from inspect import signature
from pathlib import Path
from cpl import core, dfs, ui
from planetes import __version__, configure_logging
from planetes.fits import (
    CATEGORIES,
    read_product,
    fill_dfs_product,
    destinations,
    read_raw_metadata,
)
from planetes.models import (
    validate,
    WavelengthReference,
    WavelengthCalibration,
    P2VMCalibration,
    ReducedObservables,
)

# Canonical names follow gravi_dfs.c. Values/defaults belong to our science functions.
CALIB_PARAMETERS = {
    "bad_threshold": (
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
    "dark": ("bad_threshold",),
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
MULTIPLE = {"FLAT_RAW": 4, "P2VM_RAW": 6}


def parameter_list(stage, function, names):
    """Map explicitly declared Python keywords to CPL names, aliases and defaults."""
    definitions = SCIENCE_PARAMETERS if stage == "reduce" else CALIB_PARAMETERS
    defaults = signature(function).parameters
    parameters = []
    for name in names:
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


def scientific_settings(recipe, settings):
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


def frame_type(cls):
    return (
        ui.Frame.FrameType.TABLE
        if cls
        in (
            WavelengthReference,
            WavelengthCalibration,
            P2VMCalibration,
            ReducedObservables,
        )
        else ui.Frame.FrameType.IMAGE
    )


def select_inputs(frameset, required, optional=None):
    """Repeated raw categories are legal; their physical identity is read from headers."""
    optional = {} if optional is None else optional
    roles = {**required, **optional}
    grouped = {}
    for frame in frameset:
        if frame.tag not in roles:
            raise ValueError(f"Unexpected input tag: {frame.tag}")
        grouped.setdefault(frame.tag, []).append(Path(frame.file))
    missing = set(required) - grouped.keys()
    if missing:
        raise ValueError(f"Missing required input tags: {sorted(missing)}")
    paths, objects = {}, {}
    used = ui.FrameSet()
    for role, allowed in roles.items():
        entries = grouped.get(role, [])
        if not entries:
            continue
        count = MULTIPLE.get(role, 1)
        if len(entries) != count:
            raise ValueError(
                f"Duplicate or incomplete input tag {role}: expected {count}, got {len(entries)}"
            )
        paths[role] = entries if role in MULTIPLE else entries[0]
        for path in entries:
            if allowed is None:
                metadata = read_raw_metadata(path)
                if role != "PHASE_REFERENCE" and metadata.category != role:
                    raise ValueError(f"Raw category disagrees with SOF tag {role}")
                if role == "PHASE_REFERENCE" and metadata.category != "SINGLE_SCI_RAW":
                    raise ValueError(
                        "Phase reference requires an explicitly classified all-open interferometric raw"
                    )
            else:
                product = read_product(path, allowed)
                objects[role] = product
            used.append(
                ui.Frame(
                    file=str(path),
                    tag=role,
                    group=ui.Frame.FrameGroup.RAW
                    if allowed is None
                    else ui.Frame.FrameGroup.CALIB,
                    frameType=frame_type(allowed),
                )
            )
    return paths, objects, used


def save_products(outputs, used, recipe):
    """Sign all outputs before publication; rollback only newly owned links on failure."""
    prepared = []
    for output, product in outputs:
        validate(product)
        product = replace(
            product,
            provenance=replace(product.provenance, producer=f"pyesorex:{recipe.name}"),
        )
        prepared.append((Path(output), product))
    with destinations([p for p, _ in prepared]) as temporary:
        products = ui.FrameSet()
        for (output, product), path in zip(prepared, temporary, strict=True):
            properties = core.PropertyList()
            properties.append(
                core.Property(
                    "ESO PRO CATG", core.Type.STRING, CATEGORIES[type(product)]
                )
            )
            dfs.save_propertylist(
                used,
                recipe.parameters,
                used,
                recipe.name,
                properties,
                f"planetes/{__version__}",
                str(path),
            )
            fill_dfs_product(path, product)
            products.append(
                ui.Frame(
                    file=str(path),
                    tag=CATEGORIES[type(product)],
                    group=ui.Frame.FrameGroup.PRODUCT,
                    level=ui.Frame.FrameLevel.FINAL,
                    frameType=frame_type(type(product)),
                )
            )
        products.sign_products()
    return ui.FrameSet(
        [
            ui.Frame(
                file=str(output),
                tag=CATEGORIES[type(product)],
                group=ui.Frame.FrameGroup.PRODUCT,
                level=ui.Frame.FrameLevel.FINAL,
                frameType=frame_type(type(product)),
            )
            for output, product in prepared
        ]
    )


def save_product(output, product, used, recipe):
    """Single-product call using the same multi-output publication boundary."""
    return save_products([(output, product)], used, recipe)
