"""Typed pipeline products and their current NPZ serialization."""

from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
from numpy.typing import NDArray


Array = NDArray[Any]


@dataclass(frozen=True, slots=True)
class FlatProduct:
    """Detector maps and sparse spectral traces."""

    profile_x_pixels: Array
    profile_y_pixels: Array
    flat_map_adu: Array
    dark_map_adu: Array
    bad_pixel_flags: Array
    trace_y_pixels: Array


@dataclass(frozen=True, slots=True)
class WaveProduct:
    """Wavelength at every extracted detector pixel."""

    wavelength_map_um: Array


@dataclass(frozen=True, slots=True)
class P2VMProduct:
    """V2PM calibration terms, P2VM inverse, and fit diagnostics."""

    p2vm: Array
    wavelength_um: Array
    baseline_regions: Mapping[int, list[int]]
    baseline_telescopes: Mapping[int, tuple[int, int]]
    transmission: Array
    coherence_amplitude: Array
    coherence_phase_rad: Array
    opd_um: Array
    group_delay_um: Array
    ellipse_phase_rad: Array
    ellipse_amplitude: Array


@dataclass(frozen=True, slots=True)
class ReducedProduct:
    """Reduced photometry and complex visibility observables."""

    photometric_flux_adu: Array
    coherent_flux_adu: Array
    visibility_amplitude: Array
    flags: Array
    wavelength_um: Array
    group_delay_um: Array
    differential_phase_deg: Array
    baseline_regions: Mapping[int, list[int]]
    baseline_telescopes: Mapping[int, tuple[int, int]]


ProductT = TypeVar("ProductT")


def read_product(path: Path, cls: type[ProductT]) -> ProductT:
    """Read an NPZ file into the requested product dataclass."""
    if not path.exists():
        raise FileNotFoundError(f"Pipeline product not found: {path}")
    expected = {field.name for field in fields(cls)}
    with np.load(path, allow_pickle=True) as data:
        actual = set(data.files)
        if actual != expected:
            _raise_schema_error(path, expected, actual)
        values = {name: _unpack(data[name]) for name in expected}
    return cls(**values)


def write_product(path: Path, product: object) -> None:
    """Write a product dataclass as an NPZ file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    values = {field.name: getattr(product, field.name) for field in fields(product)}
    np.savez(path, **values)


def _unpack(value: Any) -> Any:
    if isinstance(value, np.ndarray) and value.dtype == object and value.shape == ():
        return value.item()
    return value


def _raise_schema_error(path: Path, expected: set[str], actual: set[str]) -> None:
    details = []
    if missing := sorted(expected - actual):
        details.append(f"missing keys: {missing}")
    if extra := sorted(actual - expected):
        details.append(f"unexpected keys: {extra}")
    raise ValueError(f"Invalid product schema for {path.name}: " + "; ".join(details))
