"""In-memory scientific data contracts, independent of files and runners."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from numpy.typing import NDArray


Array = NDArray[Any]


@dataclass(frozen=True, slots=True)
class WavelengthReference:
    """Static laboratory line wavelengths in micrometres."""

    wavelength_um: Array


@dataclass(frozen=True, slots=True)
class FlatProduct:
    """Detector maps and sparse spectral traces."""

    profile_x_pixels: Array
    profile_y_pixels: Array
    flat_map_adu: Array
    dark_map_adu: Array
    bad_pixel_flags: Array
    trace_y_pixels: Array


@dataclass(frozen=True, slots=True)
class WaveProduct:
    """Wavelength at every extracted detector pixel."""

    wavelength_map_um: Array


@dataclass(frozen=True, slots=True)
class P2VMProduct:
    """V2PM calibration terms, P2VM inverse, and fit diagnostics."""

    p2vm: Array
    wavelength_um: Array
    baseline_regions: Mapping[int, list[int]]
    baseline_telescopes: Mapping[int, tuple[int, int]]
    transmission: Array
    coherence_amplitude: Array
    coherence_phase_rad: Array
    opd_um: Array
    group_delay_um: Array
    ellipse_phase_rad: Array
    ellipse_amplitude: Array


@dataclass(frozen=True, slots=True)
class ReducedProduct:
    """Reduced photometry and complex visibility observables."""

    photometric_flux_adu: Array
    coherent_flux_adu: Array
    visibility_amplitude: Array
    flags: Array
    wavelength_um: Array
    group_delay_um: Array
    differential_phase_deg: Array
    baseline_regions: Mapping[int, list[int]]
    baseline_telescopes: Mapping[int, tuple[int, int]]


@dataclass(frozen=True, slots=True)
class CalibrationSpectra:
    """Extracted calibration cubes passed directly to the P2VM recipe."""

    telescope_spectra: NDArray
    baseline_spectra: NDArray
    phase_reference: NDArray
    baseline_regions: Mapping[int, list[int]]
    baseline_telescopes: Mapping[int, tuple[int, int]]
    wavelength_um: NDArray
