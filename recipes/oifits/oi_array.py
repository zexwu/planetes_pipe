from __future__ import annotations
from .base import HDUModel
from numpy.typing import NDArray
from typing import Iterable, Optional

import numpy as np


class OI_ARRAY(HDUModel):
    EXTNAME = "OI_ARRAY"
    COLUMNS = [
        ("TEL_NAME", True),
        ("STA_NAME", True),
        ("STA_INDEX", True),
        ("DIAMETER", True),
        ("STAXYZ", True),
        ("FOV", False),
        ("FOVTYPE", False),
    ]

    tel_name: NDArray
    sta_name: NDArray
    sta_index: NDArray
    diameter: NDArray
    staxyz: NDArray

    fov: Optional[NDArray]
    fovtype: Optional[NDArray]

    def _post_decode(self) -> None:
        # normalize to native Python strings so lookups accept str keys
        self.sta_name = np.char.strip(self.sta_name).astype(str)
        self.tel_name = np.char.strip(self.tel_name).astype(str)
        self._tel_to_idx = dict(zip(self.sta_name, self.sta_index))
        self._idx_to_tel = dict(zip(self.sta_index, self.sta_name))

    def tel_to_idx(self, tel_name: str) -> int:
        return self._tel_to_idx[tel_name]

    def idx_to_tel(self, sta_index: int) -> str:
        return self._idx_to_tel[sta_index]

    def idx_to_name(self, idx: Iterable)-> str:
        return "".join(list(map(self.idx_to_tel, idx)))

    __doc__ = """Array geometry table decoder (``OI_ARRAY``).

    Contains station indices, names, positions, and optional telescope metadata
    used by other tables to relate baselines to physical stations.
    """
