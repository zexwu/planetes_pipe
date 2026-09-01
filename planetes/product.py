"""Typed pipeline products and their current NPZ serialization."""

import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
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
class P2VMPreprocProduct:
    """Calibration spectra resampled onto the common wavelength grid."""

    spec_tel: Array
    spec_bsl: Array
    spec_wavesc: Array
    spec_flat: Array
    tel_regs: Mapping[int, list[int]]
    bsl_regs: Mapping[int, list[int]]
    bsl_to_reg: Mapping[int, list[int]]
    bsl_to_tel: Mapping[int, tuple[int, int]]
    wl_grid: Array


@dataclass(frozen=True, slots=True)
class ObjectPreprocProduct:
    """One target spectrum on the common wavelength grid."""

    spec: Array
    spec_flat: Array
    wl_grid: Array


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
    product = cls(**values)
    _validate_product(product)
    return product


def write_product(path: Path, product: object) -> None:
    """Write a product dataclass atomically without copying its arrays."""
    if not is_dataclass(product) or isinstance(product, type):
        raise TypeError("product must be a dataclass instance")
    _validate_product(product)
    path.parent.mkdir(parents=True, exist_ok=True)
    values = {field.name: getattr(product, field.name) for field in fields(product)}
    _write_npz_atomic(path, values)


def _unpack(value: Any) -> Any:
    if isinstance(value, np.ndarray) and value.dtype == object and value.shape == ():
        return value.item()
    return value


def _validate_product(product: object) -> None:
    """Check only axes that could otherwise be silently misinterpreted."""
    wavelength = getattr(product, "wl_grid", None)
    if wavelength is not None:
        _require(
            wavelength.ndim == 1
            and len(wavelength) > 1
            and np.all(np.isfinite(wavelength))
            and np.all(np.diff(wavelength) > 0),
            "wl_grid must be finite and strictly increasing",
        )

    if isinstance(product, WaveProduct):
        _require(product.wave_map.ndim == 2 and product.wave_map.shape[0] == 24,
                 "wave_map must have 24 outputs")
    elif isinstance(product, (P2VMPreprocProduct, P2VMProduct, ReducedProduct)):
        _validate_baseline_mapping(product.bsl_to_reg, product.bsl_to_tel)

    if isinstance(product, P2VMProduct):
        n_wave = len(product.wl_grid)
        _require(
            product.p2vm.shape == (16, 24, n_wave)
            and product.v2pm.shape == (24, 16, n_wave),
            "P2VM/V2PM axes do not match PLANETES",
        )
    elif isinstance(product, ReducedProduct):
        n_frame, _, n_wave = product.coherent_flux.shape
        _require(
            product.fluxdata.shape == (n_frame, 4, n_wave)
            and product.coherent_flux.shape
            == product.visamp.shape
            == product.flag.shape
            == product.visphi.shape
            and product.gdelay.shape == (n_frame, 6)
            and product.flag.dtype == np.bool_,
            "reduced axes do not match PLANETES",
        )


def _validate_baseline_mapping(
    baseline_regions: Mapping[int, list[int]],
    baseline_telescopes: Mapping[int, tuple[int, int]],
) -> None:
    baseline_ids = set(range(6))
    regions = [region for baseline in range(6) for region in baseline_regions.get(baseline, [])]
    telescope_pairs = {
        (telescope_1, telescope_2)
        for telescope_1 in range(4)
        for telescope_2 in range(telescope_1 + 1, 4)
    }
    valid = (
        set(baseline_regions) == set(baseline_telescopes) == baseline_ids
        and sorted(regions) == list(range(24))
        and set(baseline_telescopes.values()) == telescope_pairs
    )
    _require(valid, "baseline mapping must cover 6 pairs and 24 outputs")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"Invalid pipeline product: {message}")


def _raise_schema_error(path: Path, expected: set[str], actual: set[str]) -> None:
    details = []
    if missing := sorted(expected - actual):
        details.append(f"missing keys: {missing}")
    if extra := sorted(actual - expected):
        details.append(f"unexpected keys: {extra}")
    raise ValueError(f"Invalid product schema for {path.name}: " + "; ".join(details))


def _write_npz_atomic(path: Path, values: dict[str, Any]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez(stream, **values)
            stream.flush()
            os.fsync(stream.fileno())
        temporary_path.replace(path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
