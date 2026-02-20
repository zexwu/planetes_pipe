from __future__ import annotations
from typing import Optional
import numpy as np

from .base import HDUModel


class OI_ARRAY(HDUModel):
    EXTNAME = "OI_ARRAY"
    COLUMNS = [
        ("STA_INDEX", True),
        ("STA_NAME", True),
        ("STAXYZ", True),

        ("TEL_NAME", False),
        ("DIAMETER", False),

        ("FOV", False),
        ("FOVTYPE", False),
    ]

    frame: Optional[str]  # header keyword usually

    sta_index: np.ndarray
    sta_name: np.ndarray
    sta_xyz: np.ndarray

    tel_name: Optional[np.ndarray]
    diameter: Optional[np.ndarray]

    fov: Optional[np.ndarray]
    fovtype: Optional[np.ndarray]

    def _post_decode(self) -> None:
        self.frame = self.header.get("FRAME")
        self.sta_name = [i.strip() for i in self.sta_name]
        if self.tel_name is not None:
            self.tel_name = [i.strip() for i in self.tel_name]
