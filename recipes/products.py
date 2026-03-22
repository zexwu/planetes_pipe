from pathlib import Path
from typing import Iterable, Sequence, Union


FLAT_PRODUCT = (
    "profile_map",
    "profile_xs",
    "profile_ys",
    "flat_map",
    "dark_map",
    "bad_map",
    "xs",
    "ys",
)

WAVE_PRODUCT = (
    "wave_map",
)

P2VM_PRODUCT = (
    "p2vm",
    "v2pm",
    "opd_per_baseline",
    "gd_per_baseline",
    "wl_grid",
    "bsl_to_reg",
    "bsl_to_tel",
    "ellipse_results",
)

PREPROC_CALIB_PRODUCT = (
    "spec_tel",
    "spec_bsl",
    "spec_wavesc",
    "spec_flat",
    "tel_regs",
    "bsl_regs",
    "bsl_to_reg",
    "bsl_to_tel",
    "wl_grid",
)

PREPROC_OBJECT_PRODUCT = (
    "spec",
    "spec_flat",
    "wl_grid",
)

REDUCED_PRODUCT = (
    "fluxdata",
    "visdata",
    "p2vmred",
    "wl_grid",
    "gdelay",
    "visphi",
    "bsl_to_reg",
    "bsl_to_tel",
)


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
        raise ValueError(f"Invalid product schema for {product_name}: " + "; ".join(details))
