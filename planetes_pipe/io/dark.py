"""Dark FITS layout, units and coordinate conversions."""

from pathlib import Path
from astropy.io import fits as afits
from .common import primary_header, write_quality, _verify_checksums, new_fits_file

from planetes_pipe.model.products import Dark
from .common import _image, read_image, read_setup


def encode(product: Dark, header=None, *, quality=None) -> afits.HDUList:
    """Encode this product as FITS HDUs with the supplied header and QC."""
    primary = afits.PrimaryHDU(header=primary_header(product, header))
    hdus = []
    if (
        product.temporal_rms_adu is None
        or product.n_frames is None
        or product.n_frames < 2
        or product.mean_adu.shape != product.temporal_rms_adu.shape
    ):
        raise ValueError("Saved DARK requires measured RMS and at least two frames")
    primary.header["NFRAMES"] = product.n_frames
    hdus += [
        _image("IMAGING_DATA_SC", product.mean_adu, "adu"),
        _image("IMAGING_ERR_SC", product.temporal_rms_adu, "adu"),
    ]
    hdus = afits.HDUList([primary, *hdus])
    write_quality(primary.header, quality)
    return hdus


def decode(hdus: afits.HDUList) -> Dark:
    """Decode this product from an already opened FITS file."""
    hdr = hdus[0].header
    return Dark(
        mean_adu=read_image(hdus, "IMAGING_DATA_SC", "adu"),
        temporal_rms_adu=read_image(hdus, "IMAGING_ERR_SC", "adu"),
        n_frames=hdr["NFRAMES"],
        setup=read_setup(hdr),
    )


def read(path: Path) -> Dark:
    """Read a checksummed DARK product at computation precision."""
    with afits.open(path) as hdus:
        if (
            hdus[0].header.get("INSTRUME") != "PLANETES"
            or hdus[0].header.get("ESO PRO CATG") != "DARK"
        ):
            raise ValueError("Expected a PLANETES DARK product")
        _verify_checksums(hdus)
        return decode(hdus)


def write(path: Path, product: Dark, *, quality=None) -> None:
    """Save this product without overwriting a file or recomputing QC."""
    with encode(product, quality=quality) as hdus:
        with new_fits_file(path) as temporary:
            hdus.writeto(temporary, checksum=True)
