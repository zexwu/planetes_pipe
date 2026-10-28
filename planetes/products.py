"""Simple in-memory results for the PLANETES scientific steps.

No FITS, CPL, provenance, hashes, or validation framework belongs here.
Units and array axes are documented beside the data they describe.
"""

from dataclasses import dataclass
from numpy.typing import NDArray
from .instrument import DetectorSetup

Array = NDArray


@dataclass(slots=True)
class DarkImage:
    """Mean image for subtraction only, without a measured temporal RMS."""

    mean_adu: Array  # (y, x)


@dataclass(slots=True)
class Dark:
    """Full DARK product: mean, temporal RMS and frame count."""

    mean_adu: Array  # (y, x)
    temporal_rms_adu: Array  # (y, x)
    n_frames: int
    setup: DetectorSetup | None = None


DarkInput = Dark | DarkImage


@dataclass(slots=True)
class BadPixelMap:
    """Bad-pixel flags: hot=1, dead=2, noisy=4."""

    bad_pixel_flags: Array  # (y, x)


@dataclass(slots=True)
class Flat:
    """Detector flat and sparse extraction geometry; indices are zero-based."""

    profile_x_pixels: Array  # (output, sample), -1 sentinel
    profile_y_pixels: Array  # (output, sample), -1 sentinel
    flat_map_adu: Array  # (y, x)
    trace_y_pixels: Array  # (output, x)
    profile_sigma_pixels: Array | None = None  # (output, x)
    telescope_flux_adu: Array | None = None  # (telescope,)
    setup: DetectorSetup | None = None


@dataclass(slots=True)
class Wave:
    """Wavelength solution in metres for each detector output."""

    wavelength_map_m: Array  # (output, x)
    fit_residual_m: Array | None = None
    fit_x_pixels: Array | None = None
    fit_y_pixels: Array | None = None


@dataclass(slots=True)
class WavelengthReference:
    """Static laboratory line list for WAVE calibration."""

    wavelength_m: Array
    origin: tuple[str, ...]


@dataclass(slots=True)
class P2VM:
    """Pixel-to-visibility matrix and retained calibration scan diagnostics."""

    p2vm: Array  # (data channel, output, wavelength)
    wavelength_m: Array  # (wavelength,)
    baseline_regions: dict[int, list[int]]
    baseline_telescopes: dict[int, tuple[int, int]]
    transmission: Array  # (output, telescope, wavelength)
    coherence_amplitude: Array  # (output, baseline, wavelength)
    coherence_phase_rad: Array  # (output, baseline, wavelength)
    opd_m: Array  # (baseline, frame)
    group_delay_m: Array  # (baseline, frame)
    ellipse_phase_rad: Array  # (baseline, frame, wavelength)
    ellipse_amplitude: Array  # (baseline, frame, wavelength)
    dc_term_adu: Array | None = None  # (output, baseline, wavelength)
    fit_rms_adu: Array | None = None  # (output, baseline, wavelength)


@dataclass(slots=True)
class P2VMRed:
    """Per-frame P2VM reduction, NOT an averaged VIS/OIFITS product.

    GD and differential phase may be omitted for live previews; a saved
    P2VMRED product requires both.
    """

    photometric_flux_adu: Array  # (frame, telescope, wavelength)
    coherent_flux_adu: Array  # complex (frame, baseline, wavelength)
    visibility_amplitude: Array  # (frame, baseline, wavelength)
    flags: Array  # bool (frame, baseline, wavelength)
    wavelength_m: Array
    baseline_regions: dict[int, list[int]]
    baseline_telescopes: dict[int, tuple[int, int]]
    group_delay_m: Array | None = None  # (frame, baseline)
    differential_phase_deg: Array | None = None  # (frame, baseline, wavelength)


@dataclass(slots=True)
class CalibrationSpectra:
    """In-memory P2VM fit inputs, not a FITS product."""

    telescope_spectra: Array  # (telescope, output, frame, wavelength)
    baseline_spectra: Array  # (baseline, output, frame, wavelength)
    phase_reference: Array  # (output, frame, wavelength)
    baseline_regions: dict[int, list[int]]
    baseline_telescopes: dict[int, tuple[int, int]]
    wavelength_m: Array
