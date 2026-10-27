"""Export trusted v0.2 scientific arrays under that version's PYTHONPATH, not the new API."""

from dataclasses import fields
import json
from pathlib import Path

import numpy as np


def export(directory: Path):
    """Read the frozen reference with its own decoder; never convert adopted products."""
    from planetes.io.products import read_product
    from planetes.product import FlatProduct, WaveProduct, P2VMProduct, ReducedProduct

    destination = directory / "arrays"
    destination.mkdir(exist_ok=False)
    manifest = {}
    for folder, stage, cls in [
        ("flat", "flat", FlatProduct),
        ("wave", "wave", WaveProduct),
        ("p2vm", "p2vm", P2VMProduct),
        ("science", "reduced", ReducedProduct),
    ]:
        product = read_product(directory / folder / f"planetes_{stage}.fits", cls)
        manifest[stage] = {}
        for item in fields(product):
            value = getattr(product, item.name)
            if isinstance(value, np.ndarray):
                filename = f"{stage}__{item.name}.npy"
                np.save(destination / filename, value, allow_pickle=False)
                manifest[stage][item.name] = {"array": filename}
            else:
                manifest[stage][item.name] = {"value": value}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
