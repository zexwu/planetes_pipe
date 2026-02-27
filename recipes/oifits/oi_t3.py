from __future__ import annotations
from .base import HDUModel, ReshapeMixin
from numpy.typing import NDArray
from typing import Optional

import numpy as np


class OI_T3(HDUModel, ReshapeMixin):
    EXTNAME = "OI_T3"
    COLUMNS = [
        ("MJD", True),
        ("TIME", True),
        ("INT_TIME", True),
        ("T3AMP", True), ("T3AMPERR", True),
        ("T3PHI", True), ("T3PHIERR", True),
        ("U1COORD", True), ("V1COORD", True),
        ("U2COORD", True), ("V2COORD", True),
        ("STA_INDEX", True),
        ("FLAG", True),
        ("CORRINDX_T3AMP", False),
        ("CORRINDX_T3PHI", False),
    ]

    mjd: NDArray
    time: NDArray
    int_time: NDArray
    t3phi: NDArray
    t3phierr: NDArray
    t3amp: NDArray
    t3amperr: NDArray
    u1coord: NDArray
    v1coord: NDArray
    u2coord: NDArray
    v2coord: NDArray
    sta_index: NDArray
    flag: NDArray

    corrindx_t3amp: Optional[NDArray] = None
    corrindx_t3phi: Optional[NDArray] = None

    # User defined attributes
    n_tri: int = 0
    n_dit: int = 0

    def _post_decode(self) -> None:
        n_tri = len(np.unique(self.sta_index, axis=0))
        n_dit = self.mjd.shape[0] // n_tri
        if n_tri * n_dit != self.mjd.shape[0]:
            raise ValueError("Data length must be divisible by n_tri to determine n_dit")
        self.n_tri = int(n_tri)
        self.n_dit = int(n_dit)
        return

    def reshape(self) -> None:
        """In-place reshape into [n_dit, n_tri, ...] grids."""
        fields = [i[0].lower() for i in self.COLUMNS]
        self._reshape_fields(fields, self.n_dit, self.n_tri, inplace=True)

    def flatten(self, *, inplace: bool = True) -> dict[str, np.ndarray]:
        """Flatten reshaped fields back into row-major (nrow, ...) arrays."""
        fields = [i[0].lower() for i in self.COLUMNS]
        return self._flatten_fields(fields, self.n_dit, self.n_tri, inplace=inplace)

    __doc__ = """Triple product table decoder (``OI_T3``).

    Provides closure phase (`t3phi`) and amplitude (`t3amp`) data with
    associated errors and baseline coordinates.
    """
