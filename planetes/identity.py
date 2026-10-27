"""Canonical scientific fingerprints and retained immutable encoding; no FITS or CPL."""

import json
from dataclasses import asdict, dataclass, fields, is_dataclass
from hashlib import sha256

import numpy as np
from numpy.typing import NDArray

from .instrument import ABCD_ORDER, BASELINES, TELESCOPES

Array = NDArray


@dataclass(frozen=True, slots=True)
class CanonicalRepresentation:
    """Original quantities in storage units, reusable only for unchanged scientific state."""

    decoded_digest: str
    quantities: dict[str, Array]


def _digest(product, quantities=None) -> str:
    digest = sha256()
    info = {
        "type": type(product).__name__,
        "parameters": product.provenance.parameters,
        "dependencies": product.provenance.dependencies,
        "instrument": [TELESCOPES, BASELINES, ABCD_ORDER],
    }
    for item in fields(product):
        if item.name == "provenance":
            continue
        value = getattr(product, item.name)
        if item.metadata:
            unit = item.metadata["unit"]
            if quantities is not None and item.name in quantities:
                value = quantities[item.name]
                unit = "m" if unit == "um" else unit
            values = np.ascontiguousarray(value, dtype=value.dtype.newbyteorder("<"))
            info[item.name] = [
                values.dtype.str,
                values.shape,
                unit,
                item.metadata["axes"],
            ]
            digest.update(values.tobytes())
        else:
            info[item.name] = asdict(value) if is_dataclass(value) else value
    digest.update(
        json.dumps(
            info,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
            default=lambda v: v.item(),
        ).encode()
    )
    return digest.hexdigest()


def canonical_quantities(product) -> dict[str, Array]:
    """Convert um to m once; retain exact original m values after validated decoding."""
    record = product.provenance._encoding
    if record is not None and record.decoded_digest == _digest(product):
        return record.quantities
    quantities = {}
    for f in fields(product):
        if not f.metadata:
            continue
        value = getattr(product, f.name)
        if f.metadata["unit"] == "um":
            quantities[f.name] = value / 1e6
        elif f.metadata.get("offset"):
            quantities[f.name] = (
                np.where(value < 0, value, value + f.metadata["offset"])
                if value.dtype.kind in "iu"
                else value + f.metadata["offset"]
            )
    return quantities


def fingerprint(product) -> str:
    """Hash scientific quantities in canonical storage units, including physical setup."""
    return _digest(product, canonical_quantities(product))


def retain_representation(product, quantities):
    """Attach immutable source quantities after decoding, without storing an HDU/header."""
    from dataclasses import replace

    snapshot = {}
    for name, values in (canonical_quantities(product) | quantities).items():
        saved = np.array(values, copy=True)
        saved.flags.writeable = False
        snapshot[name] = saved
    record = CanonicalRepresentation(_digest(product), snapshot)
    return replace(product, provenance=replace(product.provenance, _encoding=record))
