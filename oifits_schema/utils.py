from typing import Any, Iterable, List
from astropy.io import fits


def require_extname(hdul: List[fits.BinTableHDU], name: str) -> None:
    for h in hdul:
        ext = (h.header.get("EXTNAME") or h.name or "").strip().upper()
        if ext == name: return
    raise TypeError(f"Expected HDU {name}, got {ext}")


def header_whitelist(hdr: fits.Header, keys: Iterable[str]) -> dict[str, Any]:
    KU = {k.upper() for k in keys}
    return {k: hdr.get(k) for k in hdr.keys() if k.upper() in KU}
