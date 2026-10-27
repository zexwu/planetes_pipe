"""Thin ESO role/parameter binding and signed, failure-safe product publication."""

from dataclasses import replace
from inspect import signature
from pathlib import Path

from cpl import core, dfs, ui

from planetes import __version__, configure_logging
from planetes.fits import (
    CATEGORIES,
    read_product,
    fill_dfs_product,
    product_destination,
)
from planetes.models import validate, WavelengthReference
from planetes.qc import measure_quality

QC_KEYWORDS = {
    "bad_pixel_fraction": "ESO QC BAD PIXEL FRACTION",
    "wavelength_min_um": "ESO QC WAVE MIN",
    "wavelength_max_um": "ESO QC WAVE MAX",
    "p2vm_condition_max": "ESO QC P2VM CONDITION MAX",
    "visibility_flag_fraction": "ESO QC VIS FLAG FRACTION",
}


def parameter_list(stage, function, names):
    """Map declared scientific defaults to ordinary PyCPL parameters."""
    descriptions = {
        "bad_threshold": "Positive multiplier of median temporal RMS in retained bad-pixel tests",
        "profile_width": "Positive full extraction aperture width in detector pixels",
        "wavelength_min_um": "Minimum of the P2VM target wavelength grid in um",
        "wavelength_max_um": "Maximum of the P2VM target wavelength grid in um",
        "wavelength_channels": "Integer channels; central band must contain at least two GD samples",
        "phase_correction": "Apply the phase-reference solution when a reference is supplied",
        "gd_search_min_um": "Lower coarse GD initialization bound in um; Newton refinement is unconstrained",
        "gd_search_max_um": "Upper coarse GD initialization bound in um; Newton refinement is unconstrained",
    }
    defaults = signature(function).parameters
    values = []
    for name in names:
        parameter = ui.ParameterValue(
            name=f"planetes.{stage}.{name}",
            context=f"planetes.{stage}",
            description=descriptions[name],
            default=defaults[name].default,
        )
        parameter.cli_alias = name.replace("_", "-")
        values.append(parameter)
    return ui.ParameterList(values)


def scientific_settings(recipe, settings):
    """Reset per invocation; pass the actual values to the same Python task."""
    configure_logging()
    for parameter in recipe.parameters:
        parameter.value = parameter.default
    for name, value in settings.items():
        recipe.parameters[name].value = value
    return {
        parameter.name.rsplit(".", 1)[-1]: parameter.value
        for parameter in recipe.parameters
    }


def select_inputs(frameset, required, optional=None):
    """Explicit role -> allowed scientific type -> CPL group, independent of tag spelling."""
    optional = {} if optional is None else optional
    roles = {**required, **optional}
    paths, objects = {}, {}
    for frame in frameset:
        if frame.tag not in roles:
            raise ValueError(f"Unexpected input tag: {frame.tag}")
        if frame.tag in paths:
            raise ValueError(f"Duplicate input tag: {frame.tag}")
        paths[frame.tag] = Path(frame.file)
    if missing := sorted(set(required) - paths.keys()):
        raise ValueError(f"Missing required input tags: {missing}")
    used = ui.FrameSet()
    for role, allowed in roles.items():
        if role not in paths:
            continue
        if allowed is not None:
            product = read_product(paths[role])
            if not isinstance(product, allowed):
                raise ValueError(f"Incorrect calibration category for role {role}")
            objects[role] = product
        used.append(
            ui.Frame(
                file=str(paths[role]),
                tag=role,
                group=ui.Frame.FrameGroup.RAW
                if allowed is None
                else ui.Frame.FrameGroup.CALIB,
                frameType=ui.Frame.FrameType.TABLE
                if allowed is WavelengthReference
                else ui.Frame.FrameType.IMAGE,
            )
        )
    return paths, objects, used


def save_product(output, product, used, recipe):
    """Sign a fully validated temporary product and publish without overwriting."""
    validate(product)
    product = replace(
        product,
        provenance=replace(
            product.provenance,
            producer=f"pyesorex:{recipe.name}",
            inputs={frame.tag: str(frame.file) for frame in used},
        ),
    )
    properties = core.PropertyList()
    category = CATEGORIES[type(product)]
    properties.append(core.Property("ESO PRO CATG", core.Type.STRING, category))
    if "phase_reference_applied" in product.provenance.parameters:
        properties.append(
            core.Property(
                "ESO DRS PHASE REF APPLIED",
                core.Type.BOOL,
                product.provenance.parameters["phase_reference_applied"],
            )
        )
    for name, value in measure_quality(product).items():
        properties.append(core.Property(QC_KEYWORDS[name], core.Type.DOUBLE, value))
    with product_destination(Path(output)) as temporary:
        dfs.save_propertylist(
            used,
            recipe.parameters,
            used,
            recipe.name,
            properties,
            f"planetes/{__version__}",
            str(temporary),
        )
        fill_dfs_product(temporary, product)
        temporary_products = ui.FrameSet(
            [
                ui.Frame(
                    file=str(temporary),
                    tag=category,
                    group=ui.Frame.FrameGroup.PRODUCT,
                    level=ui.Frame.FrameLevel.FINAL,
                    frameType=ui.Frame.FrameType.IMAGE,
                )
            ]
        )
        temporary_products.sign_products()
    return ui.FrameSet(
        [
            ui.Frame(
                file=str(output),
                tag=category,
                group=ui.Frame.FrameGroup.PRODUCT,
                level=ui.Frame.FrameLevel.FINAL,
                frameType=ui.Frame.FrameType.IMAGE,
            )
        ]
    )
