"""P2VMRed FITS layout, units and coordinate conversions."""

import numpy as np
from pathlib import Path
from astropy.io import fits as afits
from .common import primary_header, write_quality, _verify_checksums, new_fits_file

from planetes_pipe.model.instrument import layout
from planetes_pipe.model.products import P2VMRed
from .common import _column, _table, _detector_table, read_column


def encode(product: P2VMRed, header=None, *, quality=None) -> afits.HDUList:
    """Encode this product as FITS HDUs with the supplied header and QC."""
    primary = afits.PrimaryHDU(header=primary_header(product, header))
    hdus = []
    hdus += [
        _detector_table(product.baseline_outputs),
        _table(
            "PLANETES_WAVELENGTH", [_column("WAVELENGTH", product.wavelength_m, "m")]
        ),
    ]
    if product.group_delay_m is None or product.differential_phase_deg is None:
        raise ValueError("Saved P2VMRED requires GD and differential phase")
    nf, _, nw = product.photometric_flux_adu.shape
    hdus += [
        _table(
            "PLANETES_FLUX",
            [
                _column("FRAME", np.repeat(np.arange(1, nf + 1), 4)),
                _column("STA_INDEX", np.tile(np.asarray(layout.telescopes, dtype=int), nf)),
                _column(
                    "FLUXDATA", product.photometric_flux_adu.reshape(-1, nw), "adu"
                ),
            ],
        )
    ]
    pairs = np.asarray(
        [[int(layout.telescopes[t]) for t in layout.baseline_pairs[b]] for b in range(6)]
    )
    hdus.append(
        _table(
            "PLANETES_VIS",
            [
                _column("FRAME", np.repeat(np.arange(1, nf + 1), 6)),
                _column("STA_INDEX", np.tile(pairs, (nf, 1))),
                _column("VISDATA", product.coherent_flux_adu.reshape(-1, nw), "adu"),
                _column("VISAMP", product.visibility_amplitude.reshape(-1, nw)),
                _column("FLAG", product.flags.reshape(-1, nw)),
                _column("GDELAY", product.group_delay_m.reshape(-1), "m"),
                _column(
                    "DIFFPHASE", product.differential_phase_deg.reshape(-1, nw), "deg"
                ),
            ],
        )
    )
    hdus = afits.HDUList([primary, *hdus])
    write_quality(primary.header, quality)
    return hdus


def decode(hdus: afits.HDUList) -> P2VMRed:
    """Decode this product from an already opened FITS file."""
    values = {}
    values["wavelength_m"] = read_column(hdus, "PLANETES_WAVELENGTH", "WAVELENGTH", "m")
    labels = list(hdus["IMAGING_DETECTOR_SC"].data["REGNAME"])
    values["baseline_outputs"] = {
        i: [labels.index(name + "-" + a) for a in layout.abcd_order]
        for i, name in enumerate(layout.baselines)
    }
    nw = len(values["wavelength_m"])
    nf = len(hdus["PLANETES_FLUX"].data) // 4
    for ext, count in [("PLANETES_FLUX", 4), ("PLANETES_VIS", 6)]:
        if not np.array_equal(
            read_column(hdus, ext, "FRAME"), np.repeat(np.arange(1, nf + 1), count)
        ):
            raise ValueError("Invalid frame row order")
    if not np.array_equal(
        read_column(hdus, "PLANETES_FLUX", "STA_INDEX"),
        np.tile(np.asarray(layout.telescopes, dtype=int), nf),
    ):
        raise ValueError("Invalid telescope identity")
    expected = np.tile(
        np.array([[int(v) for v in name] for name in layout.baselines]), (nf, 1)
    )
    if not np.array_equal(read_column(hdus, "PLANETES_VIS", "STA_INDEX"), expected):
        raise ValueError("Invalid baseline identity")
    values.update(
        photometric_flux_adu=read_column(
            hdus, "PLANETES_FLUX", "FLUXDATA", "adu"
        ).reshape(nf, 4, nw),
        coherent_flux_adu=read_column(hdus, "PLANETES_VIS", "VISDATA", "adu").reshape(
            nf, 6, nw
        ),
        visibility_amplitude=read_column(hdus, "PLANETES_VIS", "VISAMP").reshape(
            nf, 6, nw
        ),
        flags=read_column(hdus, "PLANETES_VIS", "FLAG").reshape(nf, 6, nw),
        group_delay_m=read_column(hdus, "PLANETES_VIS", "GDELAY", "m").reshape(nf, 6),
        differential_phase_deg=read_column(
            hdus, "PLANETES_VIS", "DIFFPHASE", "deg"
        ).reshape(nf, 6, nw),
    )
    return P2VMRed(**values)


def read(path: Path) -> P2VMRed:
    """Read a checksummed SINGLE_SCI_P2VMRED product at computation precision."""
    with afits.open(path) as hdus:
        if (
            hdus[0].header.get("INSTRUME") != "PLANETES"
            or hdus[0].header.get("ESO PRO CATG") != "SINGLE_SCI_P2VMRED"
        ):
            raise ValueError("Expected a PLANETES SINGLE_SCI_P2VMRED product")
        _verify_checksums(hdus)
        return decode(hdus)


def write(path: Path, product: P2VMRed, *, quality=None) -> None:
    """Save this product without overwriting a file or recomputing QC."""
    with encode(product, quality=quality) as hdus:
        with new_fits_file(path) as temporary:
            hdus.writeto(temporary, checksum=True)
