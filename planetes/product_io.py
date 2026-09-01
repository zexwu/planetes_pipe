"""Stateless serialization for typed pipeline products."""

from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

import numpy as np

from .io import write_npz_atomic


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
    """Write a product dataclass atomically without copying its arrays."""
    if not is_dataclass(product) or isinstance(product, type):
        raise TypeError("product must be a dataclass instance")
    path.parent.mkdir(parents=True, exist_ok=True)
    values = {field.name: getattr(product, field.name) for field in fields(product)}
    write_npz_atomic(path, values)


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
