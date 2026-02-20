from __future__ import annotations
from typing import Optional
import numpy as np

from .base import HDUModel


class OI_WAVELENGTH(HDUModel):
    EXTNAME = "OI_WAVELENGTH"
    COLUMNS = [
        ("EFF_WAVE", True),
        ("EFF_BAND", False),
    ]

    eff_wave: np.ndarray
    eff_band: Optional[np.ndarray]

    def _post_decode(self) -> None:
        return
