from __future__ import annotations
from dataclasses import dataclass
from astropy.io.fits import HDUList

from .oi_array import OI_ARRAY
from .oi_flux import OI_FLUX
from .oi_t3 import OI_T3
from .oi_vis import OI_VIS
from .oi_vis2 import OI_VIS2
from .oi_wavelength import OI_WAVELENGTH


@dataclass(slots=True, frozen=True)
class OI:
    array: OI_ARRAY
    wavelength: OI_WAVELENGTH
    flux: OI_FLUX
    vis: OI_VIS
    vis2: OI_VIS2
    t3: OI_T3
    extver: int

    @classmethod
    def load(cls, hdul: HDUList, extver: int) -> "OI":
        """Load a coherent interferometric dataset for a given EXTVER.

        Parameters
        ----------
        hdul : HDUList
            FITS HDU list containing the OIFITS tables.
        extver : int
            EXTVER identifier for tables that support multiple versions.

        Returns
        -------
        OI
            Immutable container with array, wavelength, flux, vis, and t3 tables.
        """

        return cls(
            array=OI_ARRAY(hdul),
            wavelength=OI_WAVELENGTH(hdul, extver),
            flux=OI_FLUX(hdul, extver),
            vis=OI_VIS(hdul, extver),
            vis2=OI_VIS2(hdul, extver),
            t3=OI_T3(hdul, extver),
            extver=extver,
        )

    def reshape(self) -> "OI":
        """In-place reshape of vis/vis2/t3 tables to [n_dit, n_baseline|n_tri, ...]."""
        self.flux.reshape()
        self.vis.reshape()
        self.vis2.reshape()
        self.t3.reshape()
        return self
