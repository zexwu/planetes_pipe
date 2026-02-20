from typing import Any, ClassVar, List, Optional, Sequence, Tuple

import numpy as np
from astropy.io import fits

from .utils import header_whitelist, require_extname


class HDUModel:
    """
    Base class for EXTNAME-named binary table HDUs.

    Subclasses define:
      EXTNAME: str
      COLUMNS: list[tuple[colname, required(bool)]]

    It will create attributes with lowercased names:
      e.g. 'STA_INDEX' -> self.sta_index
    """

    EXTNAME: ClassVar[str]
    COLUMNS: ClassVar[Sequence[Tuple[str, bool]]] = ()

    # common metadata
    header: dict[str, Any]
    insname: Optional[str]
    arrname: Optional[str]

    def __init__(
        self,
        hdul: List[fits.BinTableHDU],
        extver: Optional[int] = None,
        header_keys: Optional[list[str]] = None,
    ):
        require_extname(hdul, self.EXTNAME)

        hdu = hdul[(self.EXTNAME, extver)]
        hdr = hdu.header
        data = hdu.data

        default_keys = ["EXTNAME", "EXTVER", "INSNAME", "ARRNAME", "DATE-OBS", "OBJECT"]
        keys = default_keys if header_keys is None else header_keys
        self.header = header_whitelist(hdr, keys)

        self.extver = int(hdr.get("EXTVER", 1))
        self.insname = hdr.get("INSNAME")
        self.arrname = hdr.get("ARRNAME")

        # columns
        for colname, required in self.COLUMNS:
            attr = colname.lower()
            if colname in data.names:
                setattr(self, attr, np.asarray(data[colname]))
            elif required:
                raise KeyError(f"Missing column {colname} in {self.EXTNAME}")
            else:
                setattr(self, attr, None)

        self._post_decode()

    def _post_decode(self) -> None:
        """Subclass hook."""
        return
