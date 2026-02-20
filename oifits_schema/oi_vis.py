from __future__ import annotations
from typing import Optional
import numpy as np

from .base import HDUModel


class OI_VIS(HDUModel):
    EXTNAME = "OI_VIS"
    COLUMNS = [
        ("MJD", True),
        ("STA_INDEX", True),
        ("UCOORD", True), ("VCOORD", True),
        ("FLAG", True),
        ("VISDATA", True), ("VISERR", True),
        ("VISAMP", True), ("VISAMPERR", True),
        ("VISPHI", False), ("VISPHIERR", False),
    ]

    mjd: np.ndarray

    sta_index: np.ndarray
    ucoord: np.ndarray
    vcoord: np.ndarray

    flag: np.ndarray

    visdata: np.ndarray
    viserr: np.ndarray

    visamp: np.ndarray
    visamperr: np.ndarray

    visphi: Optional[np.ndarray]
    visphierr: Optional[np.ndarray]

    def _post_decode(self) -> None:
        return
