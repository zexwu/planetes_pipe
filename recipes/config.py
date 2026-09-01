"""Configuration validation for the PLANETES pipeline."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any


def validate_config(config: dict[str, Any]) -> None:
    """Validate structural assumptions shared by all recipes."""
    for key in ("frames", "xrange", "yrange"):
        bounds = config.get(key, [0, None])
        if not isinstance(bounds, list) or len(bounds) != 2:
            raise ValueError(f"{key} must be a two-element YAML list")
        start, stop = bounds
        if not isinstance(start, int) or start < 0:
            raise ValueError(f"{key}[0] must be a non-negative integer")
        if stop is not None and (not isinstance(stop, int) or stop <= start):
            raise ValueError(f"{key}[1] must be null or an integer greater than {start}")

    telescopes = config.get("telescopes")
    baselines = config.get("baselines")
    if not isinstance(telescopes, list) or not telescopes:
        raise ValueError("telescopes must be a non-empty YAML list")
    if not isinstance(baselines, list) or not baselines:
        raise ValueError("baselines must be a non-empty YAML list")
    if not all(isinstance(name, str) for name in telescopes + baselines):
        raise ValueError("telescope and baseline names must be strings")
    if len(set(telescopes)) != len(telescopes):
        raise ValueError("telescope names must be unique")
    if len(set(baselines)) != len(baselines):
        raise ValueError("baseline names must be unique")

    expected_baselines = {
        telescopes[i] + telescopes[j]
        for i in range(len(telescopes))
        for j in range(i + 1, len(telescopes))
    }
    if set(baselines) != expected_baselines:
        raise ValueError(
            "baselines must contain each unique telescope pair exactly once; "
            f"expected {sorted(expected_baselines)}, got {sorted(baselines)}"
        )

    abcd_order = config.get("abcd_order", ["A", "C", "B", "D"])
    if not isinstance(abcd_order, list) or len(abcd_order) != 4:
        raise ValueError("abcd_order must be a four-element YAML list")
    if {str(label).upper() for label in abcd_order} != {"A", "B", "C", "D"}:
        raise ValueError("abcd_order must contain A, B, C, and D exactly once")

    calib = config.get("calib")
    if not isinstance(calib, Mapping):
        raise ValueError("calib must be a YAML mapping")
    flat_files = calib.get("flat")
    p2vm_files = calib.get("p2vm")
    if not isinstance(flat_files, list) or len(flat_files) != len(telescopes):
        raise ValueError("calib.flat must contain one file per telescope")
    if not isinstance(p2vm_files, list) or len(p2vm_files) != len(baselines):
        raise ValueError("calib.p2vm must contain one file per baseline")

    objects = config.get("object", {})
    if not isinstance(objects, Mapping):
        raise ValueError("object must be a YAML mapping")
    for name, entry in objects.items():
        if isinstance(entry, (str, Path)):
            continue
        if not isinstance(entry, Mapping) or "data" not in entry:
            raise ValueError(
                f"object.{name} must be a path or a mapping containing 'data'"
            )

    for key in ("products", "parameters"):
        value = config.get(key, {})
        if not isinstance(value, Mapping):
            raise ValueError(f"{key} must be a YAML mapping")
