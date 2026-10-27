"""Access the packaged static wavelength reference used by the wave recipe."""

from importlib.resources import files
from pathlib import Path

from numpy.typing import NDArray

from ..product import WavelengthReference
from .products import read_product


def wavelength_reference_path() -> Path:
    """Return the installed static FITS line list, whose source ECSV is also retained."""
    return Path(str(files("planetes").joinpath("data/wavelength_lines.fits")))


def load_wavelength_lines(path: Path | None = None) -> NDArray:
    """Read laboratory reference wavelengths in micrometres."""
    return read_product(
        wavelength_reference_path() if path is None else path, WavelengthReference
    ).wavelength_um
