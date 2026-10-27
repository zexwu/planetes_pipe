"""Versioned FITS schemas for PLANETES calibration and instrumental science products."""

from dataclasses import fields
from pathlib import Path
from typing import TypeVar

import numpy as np
from astropy.io import fits

from ..instrument import (
    ABCD_ORDER,
    BASELINES,
    N_DATA,
    N_OUTPUTS,
    N_TELESCOPES,
    TELESCOPES,
)
from ..product import (
    FlatProduct,
    P2VMProduct,
    ReducedProduct,
    WaveProduct,
    WavelengthReference,
)


Product = FlatProduct | WaveProduct | P2VMProduct | ReducedProduct | WavelengthReference
ProductT = TypeVar(
    "ProductT",
    FlatProduct,
    WaveProduct,
    P2VMProduct,
    ReducedProduct,
    WavelengthReference,
)
SCHEMA_VERSION = 1
PRODUCT_CATEGORIES = {
    WavelengthReference: "PLANETES_LINE_LIST",
    ReducedProduct: "PLANETES_REDUCED",
    FlatProduct: "PLANETES_FLAT",
    WaveProduct: "PLANETES_WAVE",
    P2VMProduct: "PLANETES_P2VM",
}
# Python axis order is recorded explicitly; FITS image axes are reversed on disk.
ARRAY_SCHEMA = {
    "photometric_flux_adu": ("PHOTOMETRIC_FLUX", "adu", "frame,telescope,wavelength"),
    "coherent_flux_adu": (
        ("COHERENT_REAL", "COHERENT_IMAG"),
        "adu",
        "frame,baseline,wavelength",
    ),
    "visibility_amplitude": ("VISIBILITY_AMPLITUDE", "", "frame,baseline,wavelength"),
    "flags": ("FLAGS", "", "frame,baseline,wavelength"),
    "differential_phase_deg": (
        "DIFFERENTIAL_PHASE",
        "deg",
        "frame,baseline,wavelength",
    ),
    "profile_x_pixels": ("PROFILE_X", "pixel", "output,sample"),
    "profile_y_pixels": ("PROFILE_Y", "pixel", "output,sample"),
    "flat_map_adu": ("FLAT", "adu", "detector_y,detector_x"),
    "dark_map_adu": ("DARK", "adu", "detector_y,detector_x"),
    "bad_pixel_flags": ("BAD_PIXEL", "", "detector_y,detector_x"),
    "trace_y_pixels": ("TRACE_Y", "pixel", "output,detector_x"),
    "wavelength_map_um": ("WAVE_MAP", "um", "output,detector_x"),
    "p2vm": ("P2VM", "", "data_channel,output,wavelength"),
    "wavelength_um": ("WAVELENGTH", "um", "wavelength"),
    "transmission": ("TRANSMISSION", "", "output,telescope,wavelength"),
    "coherence_amplitude": ("COHERENCE_AMPLITUDE", "", "output,baseline,wavelength"),
    "coherence_phase_rad": ("COHERENCE_PHASE", "rad", "output,baseline,wavelength"),
    "opd_um": ("OPD", "um", "baseline,frame"),
    "group_delay_um": ("GROUP_DELAY", "um", "baseline,frame"),
    "ellipse_phase_rad": ("ELLIPSE_PHASE", "rad", "baseline,frame,wavelength"),
    "ellipse_amplitude": ("ELLIPSE_AMPLITUDE", "", "baseline,frame,wavelength"),
}


def write_product(
    path: Path,
    product: Product,
    *,
    header: fits.Header | None = None,
) -> None:
    """Write a new calibration FITS, preserving arrays without lossy conversion."""
    hdus = _product_hdus(product, header)
    path.parent.mkdir(parents=True, exist_ok=True)
    hdus.writeto(path, checksum=True)


def append_product(path: Path, product: Product) -> None:
    """Fill an empty CPL DFS product, verifying its identity before opening for updates."""
    with fits.open(path) as hdus:
        if (
            len(hdus) != 1
            or hdus[0].data is not None
            or hdus[0].header.get("ESO PRO CATG") != PRODUCT_CATEGORIES[type(product)]
            or "ESO PRO REC1 ID" not in hdus[0].header
        ):
            raise ValueError("Expected a new empty CPL DFS calibration product")
        encoded = _product_hdus(product, hdus[0].header)
    with fits.open(path, mode="update") as hdus:
        hdus[0].header = encoded[0].header
        for hdu in encoded[1:]:
            hdus.append(hdu)
        for hdu in hdus:
            hdu.add_checksum()


def _product_hdus(product: Product, header: fits.Header | None) -> fits.HDUList:
    """Encode the payload and schema without choosing a file creation policy."""
    validate_product(product)
    primary = fits.PrimaryHDU(header=None if header is None else header.copy())
    for key in list(primary.header):
        if key.startswith("ESO DPR "):
            del primary.header[key]
    primary.header["PLTSCHEM"] = SCHEMA_VERSION
    primary.header["HIERARCH ESO PRO CATG"] = PRODUCT_CATEGORIES[type(product)]
    primary.header["HIERARCH ESO PRO TYPE"] = (
        "STATIC" if isinstance(product, WavelengthReference) else "REDUCED"
    )
    primary.header["HIERARCH ESO PRO SCIENCE"] = isinstance(product, ReducedProduct)
    if isinstance(product, ReducedProduct):
        primary.header["PLTLEVEL"] = "INSTRUMENTAL"
    primary.header["PLTTELS"] = ",".join(TELESCOPES)
    primary.header["PLTBSLS"] = ",".join(BASELINES)
    primary.header["PLTABCD"] = ",".join(ABCD_ORDER)
    hdus = [primary]
    for field in fields(product):
        if field.name not in ARRAY_SCHEMA:
            continue
        name, unit, axes = _field_schema(field.name, type(product))
        values = np.asarray(getattr(product, field.name))
        if isinstance(name, tuple):
            components = [(name[0], values.real), (name[1], values.imag)]
        else:
            components = [
                (name, values.astype(np.uint8) if field.name == "flags" else values)
            ]
        for extension, array in components:
            hdu = fits.ImageHDU(array, name=extension)
            hdu.header["BUNIT"] = unit
            hdu.header["PLTAXES"] = axes
            hdus.append(hdu)
    if isinstance(product, FlatProduct):
        bad_pixels = next(hdu for hdu in hdus if hdu.name == "BAD_PIXEL")
        bad_pixels.header["FLAG1"] = "hot"
        bad_pixels.header["FLAG2"] = "dead"
        bad_pixels.header["FLAG4"] = "noisy"
    if isinstance(product, (P2VMProduct, ReducedProduct)):
        indices = sorted(product.baseline_telescopes)
        hdus.append(
            fits.BinTableHDU.from_columns(
                [
                    fits.Column(name="BASELINE", format="J", array=indices),
                    fits.Column(
                        name="TELESCOPE_INDEX",
                        format="2J",
                        array=[product.baseline_telescopes[index] for index in indices],
                    ),
                    fits.Column(
                        name="OUTPUT_INDEX",
                        format="4J",
                        array=[product.baseline_regions[index] for index in indices],
                    ),
                ],
                name="BASELINE_MAP",
            )
        )
        hdus[-1].header["INDEX0"] = True, "Indices use Python zero-based numbering"
        hdus[-1].header["PLTABCD"] = ",".join(ABCD_ORDER)
    return fits.HDUList(hdus)


def read_product(path: Path, cls: type[ProductT]) -> ProductT:
    """Read and validate the requested pilot calibration schema and HDU checksums."""
    with fits.open(path, checksum=True) as hdus:
        if hdus[0].header.get("PLTSCHEM") != SCHEMA_VERSION:
            raise ValueError(f"Unsupported PLANETES FITS schema: {path}")
        if hdus[0].header.get("ESO PRO CATG") != PRODUCT_CATEGORIES[cls]:
            raise ValueError(f"Unexpected calibration category: {path}")
        for key, value in [
            ("PLTTELS", TELESCOPES),
            ("PLTBSLS", BASELINES),
            ("PLTABCD", ABCD_ORDER),
        ]:
            if hdus[0].header.get(key) != ",".join(value):
                raise ValueError(f"Incompatible instrument ordering {key}: {path}")
        for hdu in hdus:
            if hdu.verify_checksum() != 1 or hdu.verify_datasum() != 1:
                raise ValueError(f"Missing or invalid FITS checksum: {path}")
        values = {}
        for field in fields(cls):
            if field.name not in ARRAY_SCHEMA:
                continue
            name, unit, axes = _field_schema(field.name, cls)
            components = name if isinstance(name, tuple) else (name,)
            arrays = []
            for extension in components:
                hdu = hdus[extension]
                if hdu.header.get("BUNIT") != unit or hdu.header.get("PLTAXES") != axes:
                    raise ValueError(
                        f"Incompatible units or axes in {path}[{extension}]"
                    )
                arrays.append(
                    np.array(
                        hdu.data, dtype=hdu.data.dtype.newbyteorder("="), copy=True
                    )
                )
            if isinstance(name, tuple):
                value = np.empty(
                    arrays[0].shape, dtype=np.result_type(arrays[0].dtype, np.complex64)
                )
                value.real, value.imag = arrays
            elif field.name == "flags":
                if not np.all((arrays[0] == 0) | (arrays[0] == 1)):
                    raise ValueError(f"Invalid boolean visibility flags: {path}")
                value = arrays[0].astype(bool)
            else:
                value = arrays[0]
            values[field.name] = value
        if cls in (P2VMProduct, ReducedProduct):
            table = hdus["BASELINE_MAP"]
            if table.header.get("INDEX0") is not True or len(table.data) != 6:
                raise ValueError(f"Unsupported baseline index convention: {path}")
            values["baseline_telescopes"] = {
                int(row["BASELINE"]): tuple(map(int, row["TELESCOPE_INDEX"]))
                for row in table.data
            }
            values["baseline_regions"] = {
                int(row["BASELINE"]): list(map(int, row["OUTPUT_INDEX"]))
                for row in table.data
            }
    product = cls(**values)
    validate_product(product)
    return product


def validate_product(product: Product) -> None:
    """Check scientific array dimensions and the fixed 4T/6B/24-output mappings."""
    if isinstance(product, WavelengthReference):
        wave = product.wavelength_um
        if wave.ndim != 1 or len(wave) < 3 or not np.all(np.diff(wave) > 0):
            raise ValueError(
                "Reference wavelengths must be a strictly increasing vector in um"
            )
    elif isinstance(product, ReducedProduct):
        flux = product.photometric_flux_adu
        if flux.ndim != 3:
            raise ValueError("Photometry must have axes (frame,telescope,wavelength)")
        n_frame, _, n_wave = flux.shape
        shapes = {
            "photometric_flux_adu": (n_frame, N_TELESCOPES, n_wave),
            "coherent_flux_adu": (n_frame, 6, n_wave),
            "visibility_amplitude": (n_frame, 6, n_wave),
            "flags": (n_frame, 6, n_wave),
            "wavelength_um": (n_wave,),
            "group_delay_um": (n_frame, 6),
            "differential_phase_deg": (n_frame, 6, n_wave),
        }
        for name, shape in shapes.items():
            if getattr(product, name).shape != shape:
                raise ValueError(f"Unexpected {name} shape; expected {shape}")
        if product.flags.dtype != bool or not np.iscomplexobj(
            product.coherent_flux_adu
        ):
            raise ValueError(
                "Science flags must be boolean and coherent flux must be complex"
            )
        _validate_baselines(product)
    elif isinstance(product, WaveProduct):
        if (
            product.wavelength_map_um.ndim != 2
            or product.wavelength_map_um.shape[0] != N_OUTPUTS
        ):
            raise ValueError("Wavelength map must have axes (24 outputs, detector_x)")
    elif isinstance(product, FlatProduct):
        shape = product.flat_map_adu.shape
        if (
            len(shape) != 2
            or product.dark_map_adu.shape != shape
            or product.bad_pixel_flags.shape != shape
        ):
            raise ValueError("Flat, dark, and bad-pixel detector shapes must agree")
        if product.profile_x_pixels.shape != product.profile_y_pixels.shape:
            raise ValueError("Sparse extraction X/Y shapes must agree")
        if product.profile_x_pixels.shape[
            0
        ] != N_OUTPUTS or product.trace_y_pixels.shape != (N_OUTPUTS, shape[1]):
            raise ValueError(
                "Flat extraction geometry must describe 24 detector outputs"
            )
    elif isinstance(product, P2VMProduct):
        n_wave = len(product.wavelength_um)
        n_frame = product.opd_um.shape[1]
        shapes = {
            "p2vm": (N_DATA, N_OUTPUTS, n_wave),
            "transmission": (N_OUTPUTS, N_TELESCOPES, n_wave),
            "coherence_amplitude": (N_OUTPUTS, 6, n_wave),
            "coherence_phase_rad": (N_OUTPUTS, 6, n_wave),
            "opd_um": (6, n_frame),
            "group_delay_um": (6, n_frame),
            "ellipse_phase_rad": (6, n_frame, n_wave),
            "ellipse_amplitude": (6, n_frame, n_wave),
        }
        for name, shape in shapes.items():
            if getattr(product, name).shape != shape:
                raise ValueError(f"Unexpected {name} shape; expected {shape}")
        _validate_baselines(product)


def _validate_baselines(product: P2VMProduct | ReducedProduct) -> None:
    """Check identities without imposing a different measured detector-output order."""
    expected_pairs = dict(
        enumerate(
            (
                (0, 1),
                (0, 2),
                (0, 3),
                (1, 2),
                (1, 3),
                (2, 3),
            )
        )
    )
    if dict(product.baseline_telescopes) != expected_pairs:
        raise ValueError(
            "Baseline/telescope identities must match the fixed instrument ordering"
        )
    if set(product.baseline_regions) != set(range(6)):
        raise ValueError("Baseline/output map must describe six baselines")
    outputs = [
        output for index in range(6) for output in product.baseline_regions[index]
    ]
    if any(
        len(regions) != 4 for regions in product.baseline_regions.values()
    ) or sorted(outputs) != list(range(N_OUTPUTS)):
        raise ValueError("Each detector output must occur in exactly one ABCD baseline")


def _field_schema(name: str, cls: type):
    extension, unit, axes = ARRAY_SCHEMA[name]
    if name == "group_delay_um" and cls is ReducedProduct:
        axes = "frame,baseline"
    return extension, unit, axes
