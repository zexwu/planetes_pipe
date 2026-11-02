"""Wave FITS layout, units and coordinate conversions."""

import numpy as np
from pathlib import Path
from astropy.io import fits as afits
from .common import primary_header, write_quality, _verify_checksums, new_fits_file

from planetes_pipe.model.products import Wave
from .common import _column, _table, _detector_table, read_column


def encode(product: Wave, header=None, *, quality=None) -> afits.HDUList:
    """Encode this product as FITS HDUs with the supplied header and QC."""
    primary = afits.PrimaryHDU(header=primary_header(product, header))
    hdus = []
    hdus += [
        _detector_table(),
        _table(
            "WAVE_DATA_SC",
            [
                _column(f"DATA{i + 1}", row[None], "m")
                for i, row in enumerate(product.wavelength_map_m)
            ],
        ),
    ]
    if product.fit_residual_m is not None:
        hdus.append(
            _table(
                "PLANETES_WAVE_FIT",
                [
                    _column("RESIDUAL", product.fit_residual_m, "m"),
                    _column("X", product.fit_x_pixels, "pixel"),
                    _column("Y", product.fit_y_pixels, "pixel"),
                ],
            )
        )
    hdus = afits.HDUList([primary, *hdus])
    write_quality(primary.header, quality)
    return hdus


def decode(hdus: afits.HDUList) -> Wave:
    """Decode this product from an already opened FITS file."""
    values = {}
    values["wavelength_map_m"] = np.stack(
        [read_column(hdus, "WAVE_DATA_SC", f"DATA{i + 1}", "m")[0] for i in range(24)]
    )
    if "PLANETES_WAVE_FIT" in hdus:
        values.update(
            fit_residual_m=read_column(hdus, "PLANETES_WAVE_FIT", "RESIDUAL", "m"),
            fit_x_pixels=read_column(hdus, "PLANETES_WAVE_FIT", "X", "pixel"),
            fit_y_pixels=read_column(hdus, "PLANETES_WAVE_FIT", "Y", "pixel"),
        )
    return Wave(**values)


def read(path: Path) -> Wave:
    """Read a checksummed WAVE product at computation precision."""
    with afits.open(path) as hdus:
        if (
            hdus[0].header.get("INSTRUME") != "PLANETES"
            or hdus[0].header.get("ESO PRO CATG") != "WAVE"
        ):
            raise ValueError("Expected a PLANETES WAVE product")
        _verify_checksums(hdus)
        return decode(hdus)


def write(path: Path, product: Wave, *, quality=None) -> None:
    """Save this product without overwriting a file or recomputing QC."""
    with encode(product, quality=quality) as hdus:
        with new_fits_file(path) as temporary:
            hdus.writeto(temporary, checksum=True)
