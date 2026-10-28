"""Shared FITS primitives for native arrays, metadata and checksums."""

import json

import numpy as np
from astropy.io import fits as afits


def _native(values):
    return np.array(values, dtype=values.dtype.newbyteorder("="), copy=True)


def _json_hdu(name, values):
    data = json.dumps(
        values, sort_keys=True, allow_nan=False, default=lambda v: v.item()
    ).encode("utf-8")
    return afits.ImageHDU(np.frombuffer(data, dtype=np.uint8).copy(), name=name)


def _read_json(hdus, name):
    return json.loads(hdus[name].data.tobytes().decode("utf-8"))


def _verify_checksums(hdus):
    for h in hdus:
        if h.verify_checksum() != 1 or h.verify_datasum() != 1:
            raise ValueError("Invalid or missing FITS checksum")
