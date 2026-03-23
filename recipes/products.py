from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence, Tuple, Union

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
    "visdata",
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
