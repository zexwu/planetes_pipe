"""Bad FITS layout, units and coordinate conversions."""

from pathlib import Path
from astropy.io import fits as afits
from .common import primary_header, write_quality, _verify_checksums, new_fits_file

from planetes_pipe.model.products import Bad
from .common import _image, read_image


def encode(product: Bad, header=None, *, quality=None) -> afits.HDUList:
    """Encode this product as FITS HDUs with the supplied header and QC."""
    primary = afits.PrimaryHDU(header=primary_header(product, header))
    hdus = []
    hdus.append(_image("IMAGING_DATA_SC", product.bad_pixel_flags))
    hdus = afits.HDUList([primary, *hdus])
    write_quality(primary.header, quality)
    return hdus


def decode(hdus: afits.HDUList) -> Bad:
    """Decode this product from an already opened FITS file."""
    return Bad(read_image(hdus, "IMAGING_DATA_SC", ""))


def read(path: Path) -> Bad:
    """Read a checksummed BAD product at computation precision."""
    with afits.open(path) as hdus:
        if (
            hdus[0].header.get("INSTRUME") != "PLANETES"
            or hdus[0].header.get("ESO PRO CATG") != "BAD"
        ):
            raise ValueError("Expected a PLANETES BAD product")
        _verify_checksums(hdus)
        return decode(hdus)


def write(path: Path, product: Bad, *, quality=None, header=None) -> None:
    """Save this product without overwriting a file or recomputing QC."""
    with encode(product, header, quality=quality) as hdus:
        with new_fits_file(path) as temporary:
            hdus.writeto(temporary, checksum=True)
