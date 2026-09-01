"""Typed in-memory products produced by the reduction recipes."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence, Tuple, Union

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


# Temporary compatibility declarations for recipes not yet migrated below.

ProductId = Union[str, Tuple[str, str]]


@dataclass(frozen=True)
class ProductSpec:
    product_id: ProductId
    schema: Sequence[str]


FLAT_PRODUCT = ProductSpec(
    "flat",
    (
        "profile_map",
        "profile_xs",
        "profile_ys",
        "flat_map",
        "dark_map",
        "bad_map",
        "xs",
        "ys",
    ),
)

WAVE_PRODUCT = ProductSpec("wave", ("wave_map",))

WAVE_ABERR_PRODUCT = ProductSpec("wave_aberr", WAVE_PRODUCT.schema)

P2VM_PRODUCT = ProductSpec(
    "p2vm",
    (
        "p2vm",
        "v2pm",
        "opd_per_baseline",
        "gd_per_baseline",
        "wl_grid",
        "bsl_to_reg",
        "bsl_to_tel",
        "ellipse_results",
    ),
)

PREPROC_P2VM_PRODUCT = ProductSpec(
    ("preproc", "p2vm"),
    (
        "spec_tel",
        "spec_bsl",
        "spec_wavesc",
        "spec_flat",
        "tel_regs",
        "bsl_regs",
        "bsl_to_reg",
        "bsl_to_tel",
        "wl_grid",
    ),
)

PREPROC_OBJECT_SCHEMA = (
    "spec",
    "spec_flat",
    "wl_grid",
)

REDUCED_SCHEMA = (
    "fluxdata",
    "coherent_flux",
    "visamp",
    "flag",
    "p2vmred",
    "wl_grid",
    "gdelay",
    "visphi",
    "bsl_to_reg",
    "bsl_to_tel",
)


def preproc_product(name: str) -> ProductSpec:
    if name == "p2vm":
        return PREPROC_P2VM_PRODUCT
    return ProductSpec(("preproc", name), PREPROC_OBJECT_SCHEMA)


def reduced_product(name: str) -> ProductSpec:
    return ProductSpec(("reduced", name), REDUCED_SCHEMA)


def get_product_id(product: Union[ProductId, ProductSpec]) -> ProductId:
    if isinstance(product, ProductSpec):
        return product.product_id
    return product


def get_product_schema(product: Union[ProductId, ProductSpec]) -> Sequence[str] | None:
    if isinstance(product, ProductSpec):
        return product.schema
    return None


def validate_product_keys(
    keys: Iterable[str],
    schema: Sequence[str],
    product_name: Union[str, Path],
) -> None:
    """
    Validate a product against an explicit key schema.
    """
    key_set = set(keys)
    schema_set = set(schema)

    missing = sorted(schema_set - key_set)
    unexpected = sorted(key_set - schema_set)

    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing keys: {missing}")
        if unexpected:
            details.append(f"unexpected keys: {unexpected}")
        raise ValueError(
            f"Invalid product schema for {product_name}: " + "; ".join(details)
        )
