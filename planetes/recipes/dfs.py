"""CPL binding of explicit categories/parameters and product publication."""

from pathlib import Path

from astropy.io import fits as afits
from cpl import core, dfs, ui

from planetes import __version__
from planetes.io.fits import encode_product
from planetes.io.keywords import CATEGORIES, PRO_CATEGORY, RAW_TYPES
from planetes.io.hdu import new_fits_file
from planetes.io.raw import read_raw_metadata
from planetes.products import P2VM, P2VMRed, Wave, WavelengthReference


def cpl_frame_type(cls):
    return (
        ui.Frame.FrameType.TABLE
        if cls in (WavelengthReference, Wave, P2VM, P2VMRed)
        else ui.Frame.FrameType.IMAGE
    )


def read_sof(frameset, required: dict[str, int], optional: tuple[str, ...] = ()):
    """Group SOF paths by tag, check counts and retain classified DFS input frames.

    Every value is a list of paths. Recipes load their calibration products
    explicitly; this function never reads them or changes the path value type.
    """
    grouped = {}
    for frame in frameset:
        if frame.tag not in required and frame.tag not in optional:
            raise ValueError(f"Unexpected input tag: {frame.tag}")
        grouped.setdefault(frame.tag, []).append(Path(frame.file))
    missing = set(required) - grouped.keys()
    if missing:
        raise ValueError(f"Missing required input tags: {sorted(missing)}")
    for tag, count in required.items():
        if len(grouped[tag]) != count:
            raise ValueError(
                f"Expected {count} {tag} input(s), got {len(grouped[tag])}"
            )
    for tag in optional:
        if len(grouped.get(tag, [])) > 1:
            raise ValueError(f"Expected at most one {tag} input")

    used = ui.FrameSet()
    for tag in (*required, *optional):
        for path in grouped.get(tag, []):
            used.append(
                ui.Frame(
                    file=str(path),
                    tag=tag,
                    group=ui.Frame.FrameGroup.RAW
                    if tag in RAW_TYPES or tag == "PHASE_REFERENCE"
                    else ui.Frame.FrameGroup.CALIB,
                    frameType=ui.Frame.FrameType.TABLE
                    if tag in ("LINE_LIST", "WAVE", "P2VM")
                    else ui.Frame.FrameType.IMAGE,
                )
            )
    return grouped, used


def read_shutter_id(path: Path, category: str) -> str:
    """Read the physical shutter combination used to map repeated SOF entries."""
    raw = read_raw_metadata(path, category=category)
    if None in raw.shutters:
        raise ValueError(
            f"Unknown shutters for {path}; use Python with an explicit file mapping"
        )
    return "".join(str(n) for n in range(4, 0, -1) if raw.shutters[n - 1])


def save_products(outputs, used, recipe):
    """Write/sign each (filename, product, quality) independently; never overwrite."""
    products = ui.FrameSet()
    for output, product, quality in outputs:
        with new_fits_file(output) as path:
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
            with afits.open(path, mode="update") as hdus:
                encoded = encode_product(product, hdus[0].header, quality=quality)
                hdus[0].header = encoded[0].header
                for item in encoded[1:]:
                    hdus.append(item)
                for item in hdus:
                    item.add_checksum()
            frame = ui.Frame(
                file=str(path),
                tag=CATEGORIES[type(product)],
                group=ui.Frame.FrameGroup.PRODUCT,
                level=ui.Frame.FrameLevel.FINAL,
                frameType=cpl_frame_type(type(product)),
            )
            ui.FrameSet([frame]).sign_products()
        frame.file = str(output)
        products.append(frame)
    return products
