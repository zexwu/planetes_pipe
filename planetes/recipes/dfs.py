"""CPL binding of explicit categories/parameters and verified product publication."""

from pathlib import Path

from astropy.io import fits as afits
from cpl import core, dfs, ui

from planetes import __version__
from planetes.io.fits import encode_product, read_product
from planetes.io.keywords import CATEGORIES, PRO_CATEGORY
from planetes.io.publication import publish_outputs
from planetes.qc import measure_quality
from planetes.products import P2VM, P2VMRed, Wave, WavelengthReference


def cpl_frame_type(cls):
    return (
        ui.Frame.FrameType.TABLE
        if cls in (WavelengthReference, Wave, P2VM, P2VMRed)
        else ui.Frame.FrameType.IMAGE
    )


def load_recipe_inputs(frameset, required, optional=None, *, multiple=None):
    """Load recipe-declared SOF roles; repeated roles return lists of paths."""
    optional = {} if optional is None else optional
    multiple = {} if multiple is None else multiple
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
        count = multiple.get(role, 1)
        if len(entries) != count:
            raise ValueError(
                f"Duplicate or incomplete input tag {role}: expected {count}, got {len(entries)}"
            )
        paths[role] = entries if role in multiple else entries[0]
        for path in entries:
            if allowed is not None:
                objects[role] = read_product(path, allowed)
            used.append(
                ui.Frame(
                    file=str(path),
                    tag=role,
                    group=ui.Frame.FrameGroup.RAW
                    if allowed is None
                    else ui.Frame.FrameGroup.CALIB,
                    frameType=cpl_frame_type(allowed),
                )
            )
    return paths, objects, used


def save_products(outputs, used, recipe, *, include_qc: bool = True):
    """Sign all outputs before publication; rollback only newly owned links on failure."""
    prepared = [(Path(output), product) for output, product in outputs]
    with publish_outputs([p for p, _ in prepared], verify=read_product) as temporary:
        products = ui.FrameSet()
        for (output, product), path in zip(prepared, temporary, strict=True):
            properties = core.PropertyList()
            properties.append(
                core.Property(PRO_CATEGORY, core.Type.STRING, CATEGORIES[type(product)])
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
            quality = measure_quality(product) if include_qc else None
            populate_dfs_product(path, product, quality=quality)
            products.append(
                ui.Frame(
                    file=str(path),
                    tag=CATEGORIES[type(product)],
                    group=ui.Frame.FrameGroup.PRODUCT,
                    level=ui.Frame.FrameLevel.FINAL,
                    frameType=cpl_frame_type(type(product)),
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
                frameType=cpl_frame_type(type(product)),
            )
            for output, product in prepared
        ]
    )


def save_product(output, product, used, recipe, *, include_qc: bool = True):
    """Single-product call using the same multi-output publication boundary."""
    return save_products([(output, product)], used, recipe, include_qc=include_qc)


def populate_dfs_product(path: Path, product, *, quality=None):
    """Fill only a freshly created DFS stub with the shared payload encoding."""
    with afits.open(path) as hdus:
        if (
            len(hdus) != 1
            or hdus[0].data is not None
            or hdus[0].header.get(PRO_CATEGORY) != CATEGORIES[type(product)]
            or "ESO PRO REC1 ID" not in hdus[0].header
        ):
            raise ValueError("Expected a new empty DFS product")
        encoded = encode_product(product, hdus[0].header, quality=quality)
    with afits.open(path, mode="update") as hdus:
        hdus[0].header = encoded[0].header
        for item in encoded[1:]:
            hdus.append(item)
        for item in hdus:
            item.add_checksum()
