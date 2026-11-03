"""SOF reading and product signing/registration for the PyEsoRex recipes."""

import os
from pathlib import Path

from cpl import ui

TABLE_CATEGORIES = ("WAVE", "P2VM", "SINGLE_SCI_P2VMRED")
STATIC_NAMES = {
    "DARK": "planetes_dark.fits",
    "BAD": "planetes_bad.fits",
    "FLAT": "planetes_flat.fits",
    "WAVE": "planetes_wave.fits",
    "P2VM": "planetes_p2vm.fits",
    "SINGLE_SCI_P2VMRED": "planetes_reduced.fits",
}


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


def publish(written: dict[str, Path], *, static_name: bool):
    """Sign (DATAMD5, PIPEFILE) and register written products as final PRODUCT frames.

    With static_name, each product is first renamed to its fixed recipe name;
    an existing file is never replaced.
    """
    products = ui.FrameSet()
    for category, path in written.items():
        if static_name:
            target = path.with_name(STATIC_NAMES[category])
            os.link(path, target)
            path.unlink()
            path = target
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
