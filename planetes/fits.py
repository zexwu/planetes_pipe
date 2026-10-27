"""Raw detector reading and schema-2 FITS representation of scientific objects."""

from contextlib import contextmanager
from dataclasses import asdict, fields
import json
import os
from pathlib import Path
import tempfile

import numpy as np
from astropy.io import fits as afits

from . import __version__
from .instrument import TELESCOPES, BASELINES, ABCD_ORDER
from .models import (
    DarkImage,
    DarkCalibration,
    FlatCalibration,
    WavelengthCalibration,
    P2VMCalibration,
    ReducedObservables,
    WavelengthReference,
    Provenance,
    fingerprint,
    validate,
)

SCHEMA_VERSION = 2
CATEGORIES = {
    DarkImage: "PLANETES_DARK_IMAGE",
    DarkCalibration: "PLANETES_DARK",
    FlatCalibration: "PLANETES_FLAT",
    WavelengthCalibration: "PLANETES_WAVE",
    P2VMCalibration: "PLANETES_P2VM",
    ReducedObservables: "PLANETES_REDUCED",
    WavelengthReference: "PLANETES_LINE_LIST",
}
CHUNK_FRAMES = 16


def iter_raw_cube(path: Path, *, hdu: int = 1):
    """Read the complete raw HDU-1 cube as float32, in 16-frame chunks."""
    with afits.open(path, memmap=True) as hdus:
        if len(hdus) <= hdu or hdus[hdu].data is None or hdus[hdu].data.ndim != 3:
            raise ValueError(
                f"Expected a three-dimensional raw detector cube in HDU 1: {path}"
            )
        data = hdus[hdu].data
        if not len(data):
            raise ValueError(f"Raw cube contains no frames: {path}")
        for start in range(0, len(data), CHUNK_FRAMES):
            yield np.asarray(data[start : start + CHUNK_FRAMES], dtype=np.float32)


def encode_product(product, header=None):
    """Encode one payload; Python and DFS publication share this exact representation."""
    validate(product)
    if type(product) not in CATEGORIES:
        raise TypeError(
            "Only complete science results can be saved as a reduced FITS product"
        )
    primary = afits.PrimaryHDU(header=None if header is None else header.copy())
    for key in list(primary.header):
        if key.startswith("ESO DPR "):
            del primary.header[key]
    primary.header["PLTSCHEM"] = SCHEMA_VERSION
    primary.header["PLTHASH"] = fingerprint(product)
    primary.header["PLTVERS"] = __version__
    primary.header["HIERARCH ESO PRO CATG"] = CATEGORIES[type(product)]
    primary.header["HIERARCH ESO PRO TYPE"] = (
        "STATIC" if isinstance(product, WavelengthReference) else "REDUCED"
    )
    primary.header["HIERARCH ESO PRO SCIENCE"] = isinstance(product, ReducedObservables)
    primary.header["PLTTELS"] = ",".join(TELESCOPES)
    primary.header["PLTBSLS"] = ",".join(BASELINES)
    primary.header["PLTABCD"] = ",".join(ABCD_ORDER)
    if isinstance(product, ReducedObservables):
        primary.header["PLTLEVEL"] = "INSTRUMENTAL"
    hdus = [primary]
    if isinstance(product, DarkCalibration):
        primary.header["NFRAMES"] = (
            product.n_frames,
            "Frames in temporal dark statistics",
        )
    for item in fields(product):
        if item.name == "provenance":
            continue
        value = getattr(product, item.name)
        if not item.metadata:
            if isinstance(product, WavelengthReference) and item.name == "origin":
                continue
            continue
        components = [(item.name.upper(), value)]
        if np.iscomplexobj(value):
            components = [
                (item.name.upper() + "_REAL", value.real),
                (item.name.upper() + "_IMAG", value.imag),
            ]
        for name, values in components:
            hdu = afits.ImageHDU(
                values.astype(np.uint8) if values.dtype == bool else values, name=name
            )
            hdu.header["BUNIT"] = item.metadata["unit"]
            hdu.header["PLTAXES"] = item.metadata["axes"]
            hdus.append(hdu)
    if isinstance(product, (P2VMCalibration, ReducedObservables)):
        baselines = sorted(product.baseline_regions)
        table = afits.BinTableHDU.from_columns(
            [
                afits.Column(
                    name="BASELINE_INDEX",
                    format="J",
                    array=baselines,
                ),
                afits.Column(
                    name="TELESCOPE_INDICES",
                    format="2J",
                    array=[product.baseline_telescopes[b] for b in baselines],
                ),
                afits.Column(
                    name="OUTPUT_INDICES",
                    format="4J",
                    array=[product.baseline_regions[b] for b in baselines],
                ),
            ],
            name="BASELINE_MAP",
        )
        table.header["INDEX0"] = (True, "All indices are zero based")
        table.header["PLTABCD"] = ",".join(ABCD_ORDER)
        hdus.append(table)
    metadata = asdict(product.provenance)
    data = json.dumps(
        metadata,
        sort_keys=True,
        allow_nan=False,
        ensure_ascii=False,
        default=lambda value: value.item(),
    ).encode("utf-8")
    hdus.append(
        afits.ImageHDU(np.frombuffer(data, dtype=np.uint8).copy(), name="PROVENANCE")
    )
    if isinstance(product, WavelengthReference):
        # Static reference is a proper wavelength/source table, not detector imagery.
        hdus[1] = afits.BinTableHDU.from_columns(
            [
                afits.Column(
                    name="WAVELENGTH_UM",
                    format="D",
                    unit="um",
                    array=product.wavelength_um,
                ),
                afits.Column(
                    name="ORIGIN",
                    format=f"{max(map(len,product.origin))}A",
                    array=product.origin,
                ),
            ],
            name="LINE_LIST",
        )
        hdus[1].header["PLTAXES"] = "wavelength"
    return afits.HDUList(hdus)


def read_product(path: Path, cls=None):
    """Read units, axes, checksum and content fingerprint before returning native arrays."""
    with afits.open(path, checksum=True) as hdus:
        head = hdus[0].header
        if head.get("PLTSCHEM") != SCHEMA_VERSION:
            raise ValueError(
                f"Expected schema-2 PLANETES product: {path}; regenerate old calibrations"
            )
        category = head.get("ESO PRO CATG")
        types = {value: key for key, value in CATEGORIES.items()}
        actual = types.get(category)
        if actual is None or (cls is not None and actual is not cls):
            raise ValueError(f"Unexpected product category {category}: {path}")
        for key, expected in [
            ("PLTTELS", TELESCOPES),
            ("PLTBSLS", BASELINES),
            ("PLTABCD", ABCD_ORDER),
        ]:
            if head.get(key) != ",".join(expected):
                raise ValueError(f"Incompatible instrument ordering {key}")
        for hdu in hdus:
            if hdu.verify_checksum() != 1 or hdu.verify_datasum() != 1:
                raise ValueError(f"Invalid or missing FITS checksum: {path}")
        meta = json.loads(hdus["PROVENANCE"].data.tobytes().decode("utf-8"))
        values = {}
        if actual is DarkCalibration:
            values["n_frames"] = head["NFRAMES"]
        if actual in (P2VMCalibration, ReducedObservables):
            table = hdus["BASELINE_MAP"]
            if table.header.get("INDEX0") is not True or table.header.get(
                "PLTABCD"
            ) != ",".join(ABCD_ORDER):
                raise ValueError("Invalid baseline-map index or output convention")
            values["baseline_regions"] = {
                int(row["BASELINE_INDEX"]): [int(v) for v in row["OUTPUT_INDICES"]]
                for row in table.data
            }
            values["baseline_telescopes"] = {
                int(row["BASELINE_INDEX"]): tuple(
                    int(v) for v in row["TELESCOPE_INDICES"]
                )
                for row in table.data
            }
        if actual is WavelengthReference:
            table = hdus["LINE_LIST"]
            if (
                table.columns["WAVELENGTH_UM"].unit != "um"
                or table.header.get("PLTAXES") != "wavelength"
            ):
                raise ValueError("Invalid static reference units or axes")
            values["wavelength_um"] = np.array(
                table.data["WAVELENGTH_UM"], dtype=np.float64
            )
            values["origin"] = tuple(str(v).strip() for v in table.data["ORIGIN"])
        else:
            for item in fields(actual):
                if not item.metadata:
                    continue
                names = [item.name.upper()]
                if item.name == "coherent_flux_adu":
                    names = [item.name.upper() + "_REAL", item.name.upper() + "_IMAG"]
                arrays = []
                for name in names:
                    hdu = hdus[name]
                    if (
                        hdu.header.get("BUNIT") != item.metadata["unit"]
                        or hdu.header.get("PLTAXES") != item.metadata["axes"]
                    ):
                        raise ValueError(f"Incompatible units or axes: {name}")
                    arrays.append(
                        np.array(
                            hdu.data, dtype=hdu.data.dtype.newbyteorder("="), copy=True
                        )
                    )
                if len(arrays) == 2:
                    data = np.empty(
                        arrays[0].shape,
                        dtype=np.result_type(arrays[0].dtype, np.complex64),
                    )
                    data.real, data.imag = arrays
                elif item.name == "flags":
                    if not ((arrays[0] == 0) | (arrays[0] == 1)).all():
                        raise ValueError("Invalid boolean visibility flags")
                    data = arrays[0].astype(bool)
                else:
                    data = arrays[0]
                values[item.name] = data
        values["provenance"] = Provenance(**meta)
        product = actual(**values)
        validate(product)
        if fingerprint(product) != head.get("PLTHASH"):
            raise ValueError(f"Scientific content fingerprint mismatch: {path}")
        return product


@contextmanager
def product_destination(path: Path):
    """Publish a verified temporary file atomically; never replace an existing target."""
    path = Path(path).absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(path):
        raise FileExistsError(f"Refusing to overwrite {path}")
    with tempfile.TemporaryDirectory(prefix=".planetes-", dir=path.parent) as directory:
        temporary = Path(directory) / path.name
        yield temporary
        read_product(temporary)
        # link fails atomically if another writer creates the destination first.
        os.link(temporary, path)


def write_product(path: Path, product) -> None:
    """Explicitly save a complete product; failures leave no final-name partial file."""
    encoded = encode_product(product)
    with product_destination(path) as temporary:
        encoded.writeto(temporary, checksum=True)


def fill_dfs_product(path: Path, product) -> None:
    """Populate only a fresh temporary DFS header; array encoding stays in this module."""
    with afits.open(path) as hdus:
        if (
            len(hdus) != 1
            or hdus[0].data is not None
            or hdus[0].header.get("ESO PRO CATG") != CATEGORIES[type(product)]
            or "ESO PRO REC1 ID" not in hdus[0].header
        ):
            raise ValueError("Expected a new empty DFS product")
        encoded = encode_product(product, hdus[0].header)
    with afits.open(path, mode="update") as hdus:
        hdus[0].header = encoded[0].header
        for hdu in encoded[1:]:
            hdus.append(hdu)
        for hdu in hdus:
            hdu.add_checksum()
