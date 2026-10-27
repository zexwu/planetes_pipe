"""Shared file selection and DFS product registration for the ESO recipe adapters."""

from pathlib import Path
from collections.abc import Sequence

from cpl import core, dfs, ui

from planetes import __version__
from planetes.algorithms.quality import measure_quality
from planetes.io.products import PRODUCT_CATEGORIES, append_product


QC_KEYWORDS = {
    "bad_pixel_fraction": "ESO QC BAD PIXEL FRACTION",
    "wavelength_min_um": "ESO QC WAVE MIN",
    "wavelength_max_um": "ESO QC WAVE MAX",
    "p2vm_condition_max": "ESO QC P2VM CONDITION MAX",
    "visibility_flag_fraction": "ESO QC VIS FLAG FRACTION",
}


def select_inputs(
    frameset: ui.FrameSet,
    required: Sequence[str],
    optional: Sequence[str] = (),
) -> tuple[dict[str, Path], ui.FrameSet]:
    """Select exactly one file per tag and retain a deterministic provenance order."""
    allowed = set(required) | set(optional)
    paths = {}
    for frame in frameset:
        if frame.tag not in allowed:
            raise ValueError(f"Unexpected input tag: {frame.tag}")
        if frame.tag in paths:
            raise ValueError(f"Duplicate input tag: {frame.tag}")
        paths[frame.tag] = Path(frame.file)
    if missing := sorted(set(required) - paths.keys()):
        raise ValueError(f"Missing required input tags: {', '.join(missing)}")
    used = ui.FrameSet()
    for tag in [*required, *optional]:
        if tag not in paths:
            continue
        used.append(
            ui.Frame(
                file=str(paths[tag]),
                tag=tag,
                group=ui.Frame.FrameGroup.CALIB
                if tag in PRODUCT_CATEGORIES.values()
                else ui.Frame.FrameGroup.RAW,
                frameType=ui.Frame.FrameType.IMAGE,
            )
        )
    return paths, used


def new_output(name: str) -> Path:
    """Reserve a new product name without overwriting an existing file."""
    output = Path(name)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing product: {output}")
    return output


def save_product(
    output: Path,
    product: object,
    used: ui.FrameSet,
    recipe: ui.PyRecipe,
    *,
    phase_reference_applied: bool | None = None,
) -> ui.FrameSet:
    """Use CPL DFS provenance and sign a schema-checked FITS payload."""
    properties = core.PropertyList()
    category = PRODUCT_CATEGORIES[type(product)]
    properties.append(core.Property("ESO PRO CATG", core.Type.STRING, category))
    if phase_reference_applied is not None:
        properties.append(
            core.Property(
                "ESO DRS PHASE REF APPLIED", core.Type.BOOL, phase_reference_applied
            )
        )
    for name, value in measure_quality(product).items():
        properties.append(core.Property(QC_KEYWORDS[name], core.Type.DOUBLE, value))
    dfs.save_propertylist(
        used,
        recipe.parameters,
        used,
        recipe.name,
        properties,
        f"planetes/{__version__}",
        str(output),
    )
    append_product(output, product)
    frame = ui.Frame(
        file=str(output),
        tag=category,
        group=ui.Frame.FrameGroup.PRODUCT,
        level=ui.Frame.FrameLevel.FINAL,
        frameType=ui.Frame.FrameType.IMAGE,
    )
    products = ui.FrameSet([frame])
    products.sign_products()
    return products
