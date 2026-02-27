from __future__ import annotations
from typing import Any, ClassVar, Optional, Sequence, Tuple, Iterable, Mapping, TypeVar
from collections.abc import Sequence as SeqABC
from types import MappingProxyType

import numpy as np
from astropy.io import fits


def require_extname(hdul: fits.HDUList, name: str) -> None:
    target = name.strip().upper()
    for h in hdul:
        ext = (h.header.get("EXTNAME") or h.name or "").strip().upper()
        if ext == target:
            return
    raise KeyError(f"Missing HDU EXTNAME={target}")


def header_whitelist(hdr: fits.Header, keys: Iterable[str]) -> dict[str, Any]:
    KU = {k.upper() for k in keys}
    return {k: hdr.get(k) for k in hdr.keys() if k.upper() in KU}

T_HDUModel = TypeVar("T_HDUModel", bound="HDUModel")


class HDUModel:
    """Base class for EXTNAME-named binary table HDUs.

    Subclasses define ``EXTNAME`` and ``COLUMNS`` (name, required) and gain
    lower-case attributes for each column (e.g. ``STA_INDEX`` -> ``sta_index``).
    """

    EXTNAME: ClassVar[str]
    COLUMNS: ClassVar[Sequence[Tuple[str, bool]]] = ()

    # common metadata
    header: dict[str, Any]

    extver: Optional[int]
    insname: Optional[str]
    arrname: Optional[str]

    @classmethod
    def from_attrs(
        cls: type[T_HDUModel],
        *,
        extver: int = 1,
        insname: Optional[str] = None,
        arrname: Optional[str] = None,
        header: Optional[Mapping[str, Any]] = None,
        header_keys: Optional[Sequence[str]] = None,
        strict: bool = True,
        **attrs: Any,
    ) -> T_HDUModel:
        """Construct an HDUModel instance from already-available column arrays.

        This bypasses FITS decoding and is useful for creating derived products.

        Parameters
        ----------
        extver, insname, arrname :
            Common OIFITS header identifiers.
        header :
            Optional mapping used to populate ``metadata`` (a whitelisted subset
            of keys, unless ``header_keys`` is provided).
        header_keys :
            Keys to keep from ``header`` for ``metadata``. Defaults to the same
            set used by FITS decoding.
        **attrs :
            Column arrays keyed by either lower-case attribute names
            (e.g. ``visamp``) or FITS column names (e.g. ``VISAMP``).
        """

        obj = cls.__new__(cls)

        default_keys = ["EXTNAME", "EXTVER", "INSNAME", "ARRNAME", "DATE-OBS", "OBJECT", "FRAME"]
        keys = default_keys if header_keys is None else list(header_keys)

        src = {} if header is None else dict(header)
        src_upper = {str(k).upper(): v for k, v in src.items()}
        meta: dict[str, Any] = {k.upper(): src_upper.get(k.upper()) for k in keys}

        meta["EXTNAME"] = cls.EXTNAME
        meta["EXTVER"] = int(extver)
        if insname is not None:
            meta["INSNAME"] = insname
        if arrname is not None:
            meta["ARRNAME"] = arrname

        obj.header = MappingProxyType(meta)
        obj.extver = int(extver)
        obj.insname = insname
        obj.arrname = arrname

        # Normalize provided attributes to allow both VISAMP and visamp keys.
        attrs_lc = {str(k).lower(): v for k, v in attrs.items()}

        for colname, required in cls.COLUMNS:
            attr = colname.lower()
            value = attrs.get(colname, attrs_lc.get(attr))
            if value is None and required and strict:
                raise KeyError(f"Missing column {colname} for {cls.EXTNAME}")
            else:
                setattr(obj, attr, value)

        obj._post_decode()
        return obj


    def __init__(
        self,
        hdul: fits.HDUList,
        extver: Optional[int] = None,
        header_keys: Optional[list[str]] = None,
        strict: bool = True,
    ) -> None:
        require_extname(hdul, self.EXTNAME)

        if extver is None:
            hdu = hdul[self.EXTNAME]
        else:
            hdu = hdul[(self.EXTNAME, extver)]
        hdr = hdu.header
        data = hdu.data

        default_keys = ["EXTNAME", "EXTVER", "INSNAME", "ARRNAME", "DATE-OBS", "OBJECT", "FRAME"]
        keys = default_keys if header_keys is None else header_keys
        self.header = MappingProxyType(header_whitelist(hdr, keys))

        self.extver = int(hdr.get("EXTVER", 1))
        self.insname = hdr.get("INSNAME")
        self.arrname = hdr.get("ARRNAME")

        # columns
        for colname, required in self.COLUMNS:
            attr = colname.lower()
            if colname in data.names:
                setattr(self, attr, data[colname])
            elif required and strict:
                raise KeyError(f"Missing column {colname} in {self.EXTNAME}")
            else:
                setattr(self, attr, None)

        self._post_decode()

    def _post_decode(self) -> None:
        """Subclass hook for value cleanup/validation after decode."""
        return

    # Properties for common metadata; MappingProxyType keeps header read-only.
    @property
    def metadata(self) -> Mapping[str, Any]:
        """Immutable header subset (EXTNAME/EXTVER/INSNAME/ARRNAME etc.)."""
        return self.header

    @property
    def extver_id(self) -> Optional[int]:
        """Alias for ``EXTVER`` header value."""
        return self.extver

    def __repr__(self) -> str:
        parts: list[str] = []

        # identity
        parts.append(f"{self.__class__.__name__}(")

        # header-like identifiers
        for attr in ["extver", "insname", "arrname"]:
            parts.append(f"  {attr:10s}= {getattr(self, attr)!r},")

        # columns: show dtype/shape only
        for colname, _required in self.COLUMNS:
            attr = colname.lower()
            v: Any = getattr(self, attr, None)
            if v is None: continue

            a = np.asarray(v)
            # if a.dtype.kind in ("U", "S", "O") and a.ndim == 1 and a.size <= 8:
            if a.ndim == 1 and a.size <= 8:
                # short string lists: show a few values
                parts.append(f"  {attr:10s}= {a!r},")
            else:
                shape = str(a.shape)+ ","
                dtype = "'" + str(a.dtype) + "'"
                parts.append(f"  {attr:10s}= array(shape={shape:10s}dtype={dtype:6s}),")

        parts.append(")")

        return "\n".join(parts)

    def to_hdu(
        self,
        *,
        extver: Optional[int] = None,
        header_overrides: Optional[Mapping[str, Any]] = None,
        flatten: bool = True,
        strict: bool = True,
    ) -> fits.BinTableHDU:
        """Encode this model back into a FITS BinTableHDU.

        Notes
        -----
        - This is a best-effort encoder intended for writing derived products.
        - If ``flatten`` is True and this instance mixes in ``ReshapeMixin``,
          the returned HDU uses row-major shapes (nrow, ...) even if the object
          has been reshaped to (n_dit, n_bsl|n_tri|n_tel, ...).
        """

        flattened: dict[str, np.ndarray] = {}
        if flatten and isinstance(self, ReshapeMixin):
            flattened = self.flatten(inplace=False)

        arrays: dict[str, np.ndarray] = {}
        for colname, required in self.COLUMNS:
            attr = colname.lower()
            value: Any = getattr(self, attr, None)
            if value is None:
                if required and strict:
                    raise KeyError(f"Missing value for required column {colname} in {self.EXTNAME}")
                continue

            if flattened and attr in flattened:
                arr = np.asarray(flattened[attr])
            else:
                arr = np.asarray(value)
            arrays[colname] = arr

        if not arrays:
            raise ValueError(f"No columns available to encode for {self.EXTNAME}")

        nrow: Optional[int] = None
        for colname, arr in arrays.items():
            if arr.ndim < 1:
                raise ValueError(f"Column {colname} must have at least 1 dimension (rows)")
            if nrow is None:
                nrow = int(arr.shape[0])
            elif int(arr.shape[0]) != nrow:
                raise ValueError(f"Column {colname} has {arr.shape[0]} rows, expected {nrow}")
        assert nrow is not None

        dtype_fields: list[tuple[Any, ...]] = []
        for colname, arr in arrays.items():
            if arr.ndim == 1:
                dtype_fields.append((colname, arr.dtype))
            else:
                dtype_fields.append((colname, arr.dtype, arr.shape[1:]))

        rec = np.empty(nrow, dtype=np.dtype(dtype_fields))
        for colname, arr in arrays.items():
            rec[colname] = arr

        hdr = fits.Header()
        # Start from the decoded (whitelisted) header subset, then normalize required keys.
        for k, v in self.header.items():
            if v is None:
                continue
            hdr[k] = v

        hdr["EXTNAME"] = self.EXTNAME
        hdr["EXTVER"] = int(extver if extver is not None else (self.extver or 1))
        if self.insname is not None:
            hdr["INSNAME"] = self.insname
        if self.arrname is not None:
            hdr["ARRNAME"] = self.arrname

        if header_overrides:
            for k, v in header_overrides.items():
                if v is None:
                    hdr.remove(k, ignore_missing=True)
                else:
                    hdr[k] = v

        return fits.BinTableHDU(data=rec, header=hdr)


class ReshapeMixin:
    """Shared helper for reshaping time-ordered rows into [outer, inner, ...] grids."""

    def _reshape_fields(
        self,
        fields: SeqABC[str],
        outer: int,
        inner: int,
        *,
        inplace: bool = True,
    ) -> dict[str, np.ndarray]:
        result: dict[str, np.ndarray] = {}
        for name in fields:
            value = getattr(self, name, None)
            if value is None: continue
            arr = np.asarray(value)

            # idempotent: already shaped as (outer, inner, ...)
            if arr.ndim >= 2 and arr.shape[0] == outer and arr.shape[1] == inner:
                reshaped = arr
            else:
                if arr.shape[0] != outer * inner:
                    raise ValueError(f"Data length of {name} must be {outer} * {inner}")
                tail = arr.shape[1:]
                shape = (outer, inner, *tail) if tail else (outer, inner)
                reshaped = arr.reshape(shape)

            if inplace: setattr(self, name, reshaped)
            result[name] = reshaped
        return result

    def _flatten_fields(
        self,
        fields: SeqABC[str],
        outer: int,
        inner: int,
        *,
        inplace: bool = True,
    ) -> dict[str, np.ndarray]:
        """Inverse of ``_reshape_fields`` for values shaped as (outer, inner, ...)."""
        result: dict[str, np.ndarray] = {}
        for name in fields:
            value = getattr(self, name, None)
            if value is None: continue
            arr = np.asarray(value)

            # idempotent: already flat as (outer*inner, ...)
            if arr.ndim >= 2 and arr.shape[0] == outer and arr.shape[1] == inner:
                tail = arr.shape[2:]
                shape = (outer * inner, *tail) if tail else (outer * inner,)
                flattened = arr.reshape(shape)
            else:
                flattened = arr

            if inplace: setattr(self, name, flattened)
            result[name] = flattened
        return result
