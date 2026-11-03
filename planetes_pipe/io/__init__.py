"""Raw and product FITS I/O, independent of CPL execution."""

from planetes_pipe.model.products import Bad, Dark, Flat, P2VM, P2VMRed, Wave

from . import bad, dark, flat, p2vm, raw, science, wave

# Product class -> module with its read/encode/write functions.
CODECS = {Dark: dark, Bad: bad, Flat: flat, Wave: wave, P2VM: p2vm, P2VMRed: science}

__all__ = ["CODECS", "bad", "dark", "flat", "p2vm", "raw", "science", "wave"]
