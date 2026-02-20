from __future__ import annotations
from typing import Optional
import numpy as np

from .base import HDUModel


class OI_FLUX(HDUModel):
    EXTNAME = "OI_FLUX"
    COLUMNS = [
        ("STA_INDEX", True),
        ("MJD", True),
        ("FLAG", True),
        ("FLUX", True), ("FLUXERR", True),
    ]

    mjd: np.ndarray

    sta_index: np.ndarray

    flag: np.ndarray
    fluxdata: np.ndarray
    fluxerr: np.ndarray


    def _post_decode(self) -> None:
        return
