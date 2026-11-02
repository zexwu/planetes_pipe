"""Flat FITS layout, units and coordinate conversions."""

from pathlib import Path
from astropy.io import fits as afits
from .common import primary_header, write_quality, _verify_checksums, new_fits_file

from planetes_pipe.model.products import Flat
from .common import (
    _image,
    _column,
    _table,
    _detector_table,
    read_image,
    read_column,
    read_setup,
    one_based_pixels,
    zero_based_pixels,
)


def encode(product: Flat, header=None, *, quality=None) -> afits.HDUList:
    """Encode this product as FITS HDUs with the supplied header and QC."""
    primary = afits.PrimaryHDU(header=primary_header(product, header))
    hdus = []
    hdus += [
        _detector_table(),
        _image("PLANETES_FLAT_IMAGE", product.flat_map_adu, "adu"),
    ]
    columns = [
        _column("REGNAME", hdus[0].data["REGNAME"]),
        _column("CENTERY", one_based_pixels(product.trace_y_pixels), "pixel"),
    ]
    if product.trace_x_range is not None:
        columns.append(_column("XRANGE", one_based_pixels(product.trace_x_range), "pixel"))
    if product.profile_sigma_pixels is not None:
        columns.append(_column("WIDTH", product.profile_sigma_pixels, "pixel"))
    hdus += [
        _table("PROFILE_PARAMS", columns),
        _table(
            "PLANETES_PROFILE",
            [
                _column("PIXEL_X", one_based_pixels(product.profile_x_pixels), "pixel"),
                _column("PIXEL_Y", one_based_pixels(product.profile_y_pixels), "pixel"),
            ],
        ),
    ]
    if product.telescope_flux_adu is not None:
        hdus.append(_image("PLANETES_FLAT_FLUX", product.telescope_flux_adu, "adu"))
    hdus = afits.HDUList([primary, *hdus])
    write_quality(primary.header, quality)
    return hdus


def decode(hdus: afits.HDUList) -> Flat:
    """Decode this product from an already opened FITS file."""
    hdr = hdus[0].header
    values = {}
    values.update(
        flat_map_adu=read_image(hdus, "PLANETES_FLAT_IMAGE", "adu"),
        trace_y_pixels=zero_based_pixels(
            read_column(hdus, "PROFILE_PARAMS", "CENTERY", "pixel")
        ),
        profile_x_pixels=zero_based_pixels(
            read_column(hdus, "PLANETES_PROFILE", "PIXEL_X", "pixel")
        ),
        profile_y_pixels=zero_based_pixels(
            read_column(hdus, "PLANETES_PROFILE", "PIXEL_Y", "pixel")
        ),
    )
    if "XRANGE" in hdus["PROFILE_PARAMS"].columns.names:
        values["trace_x_range"] = zero_based_pixels(
            read_column(hdus, "PROFILE_PARAMS", "XRANGE", "pixel")
        ).astype(int)
    if "WIDTH" in hdus["PROFILE_PARAMS"].columns.names:
        values["profile_sigma_pixels"] = read_column(
            hdus, "PROFILE_PARAMS", "WIDTH", "pixel"
        )
    if "PLANETES_FLAT_FLUX" in hdus:
        values["telescope_flux_adu"] = read_image(hdus, "PLANETES_FLAT_FLUX", "adu")
    values["setup"] = read_setup(hdr)
    return Flat(**values)


def read(path: Path) -> Flat:
    """Read a checksummed FLAT product at computation precision."""
    with afits.open(path) as hdus:
        if (
            hdus[0].header.get("INSTRUME") != "PLANETES"
            or hdus[0].header.get("ESO PRO CATG") != "FLAT"
        ):
            raise ValueError("Expected a PLANETES FLAT product")
        _verify_checksums(hdus)
        return decode(hdus)


def write(path: Path, product: Flat, *, quality=None) -> None:
    """Save this product without overwriting a file or recomputing QC."""
    with encode(product, quality=quality) as hdus:
        with new_fits_file(path) as temporary:
            hdus.writeto(temporary, checksum=True)
