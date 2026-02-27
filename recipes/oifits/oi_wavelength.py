from __future__ import annotations
from .base import HDUModel
from numpy.typing import NDArray
from typing import Optional


class OI_WAVELENGTH(HDUModel):
    EXTNAME = "OI_WAVELENGTH"
    COLUMNS = [
        ("EFF_WAVE", True),
        ("EFF_BAND", False),
    ]

    eff_wave: NDArray
    eff_band: Optional[NDArray]

    def _post_decode(self) -> None:
        return

    __doc__ = """Wavelength table decoder (``OI_WAVELENGTH``).

    Exposes effective wavelength (`eff_wave`) and optional bandpass (`eff_band`).
    """
