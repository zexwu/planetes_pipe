from typing import Any, Optional
import numpy as np
from .base import HDUModel


class OI_T3(HDUModel):
    EXTNAME = "OI_T3"
    COLUMNS = [
        ("MJD", True),
        ("STA_INDEX", True),
        ("U1COORD", True), ("V1COORD", True),
        ("U2COORD", True), ("V2COORD", True),
        ("FLAG", True),
        ("T3PHI", True), ("T3PHIERR", True),
        ("T3AMP", False), ("T3AMPERR", False),
    ]

    extver: int
    mjd: np.ndarray

    sta_index: np.ndarray
    u1coord: np.ndarray
    v1coord: np.ndarray
    u2coord: np.ndarray
    v2coord: np.ndarray

    flag: np.ndarray

    t3phi: np.ndarray
    t3phierr: np.ndarray

    t3amp: Optional[np.ndarray]
    t3amperr: Optional[np.ndarray]

    def _post_decode(self) -> None:
        return
