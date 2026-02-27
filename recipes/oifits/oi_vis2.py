from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Optional

from .base import HDUModel, ReshapeMixin


class OI_VIS2(HDUModel, ReshapeMixin):
    EXTNAME = "OI_VIS2"
    COLUMNS = [
        ("TIME", True),
        ("MJD", True),
        ("INT_TIME", True),
        ("VIS2DATA", True),
        ("VIS2ERR", True),
        ("UCOORD", True),
        ("VCOORD", True),
        ("STA_INDEX", True),
        ("FLAG", True),
        ("CORRINDX_VIS2DATA", False),
    ]

    time: NDArray
    mjd: NDArray
    int_time: NDArray
    vis2data: NDArray
    vis2err: NDArray
    ucoord: NDArray
    vcoord: NDArray
    sta_index: NDArray
    flag: NDArray

    corrindx_vis2data: Optional[NDArray]

    # Derived shapes
    n_bsl: int = 0
    n_dit: int = 0

    def _post_decode(self) -> None:
        self.n_bsl = len(np.unique(self.sta_index, axis=0))
        self.n_dit = self.mjd.shape[0] // self.n_bsl
        if self.n_bsl * self.n_dit != self.mjd.shape[0]:
            raise ValueError("Data length must be divisible by n_bsl to determine n_dit")
        return

    def reshape(self) -> None:
        """In-place reshape into [n_dit, n_bsl, ...] grids."""
        fields = [i[0].lower() for i in self.COLUMNS]
        self._reshape_fields(fields, self.n_dit, self.n_bsl, inplace=True)

    def flatten(self, *, inplace: bool = True) -> dict[str, np.ndarray]:
        """Flatten reshaped fields back into row-major (nrow, ...) arrays."""
        fields = [i[0].lower() for i in self.COLUMNS]
        return self._flatten_fields(fields, self.n_dit, self.n_bsl, inplace=inplace)

    __doc__ = """Squared visibility table decoder (``OI_VIS2``).

    Fields map directly to OIFITS binary table columns. See class attributes for
    available columns and the instance properties for numpy arrays.
    """
