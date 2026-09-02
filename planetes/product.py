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

    profile_map: Array
    profile_xs: Array
    profile_ys: Array
    flat_map: Array
    dark_map: Array
    bad_map: Array
    xs: Array
    ys: Array


@dataclass(frozen=True, slots=True)
class WaveProduct:
    """Wavelength at every extracted detector pixel."""

    wave_map: Array


@dataclass(frozen=True, slots=True)
class P2VMProduct:
    """Visibility-to-pixel model and its inverse."""

    p2vm: Array
    v2pm: Array
    opd_per_baseline: Array
    gd_per_baseline: Array
    wl_grid: Array
    bsl_to_reg: Mapping[int, list[int]]
    bsl_to_tel: Mapping[int, tuple[int, int]]
    ellipse_results: Array


@dataclass(frozen=True, slots=True)
class ReducedProduct:
    """Reduced photometry and complex visibility observables."""

    fluxdata: Array
    coherent_flux: Array
    visamp: Array
    flag: Array
    p2vmred: Array
    wl_grid: Array
    gdelay: Array
    visphi: Array
    bsl_to_reg: Mapping[int, list[int]]
    bsl_to_tel: Mapping[int, tuple[int, int]]


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
