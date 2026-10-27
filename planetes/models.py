"""Scientific arrays, units, axes and calibration relationships; no file or runner code."""

from dataclasses import dataclass, field, fields
from hashlib import sha256
from itertools import combinations
import json

import numpy as np
from numpy.typing import NDArray

from .instrument import (
    N_BASELINES,
    N_DATA,
    N_OUTPUTS,
    N_TELESCOPES,
    TELESCOPES,
    BASELINES,
    ABCD_ORDER,
)

Array = NDArray


def array(unit: str, axes: str):
    """Declare scientific units and Python axis order once, alongside the field."""
    return field(metadata={"unit": unit, "axes": axes})


@dataclass(frozen=True, slots=True)
class Provenance:
    """Actual producer, settings, source roles and parent scientific fingerprints."""

    producer: str = "planetes"
    parameters: dict = field(default_factory=dict)
    inputs: dict[str, str] = field(default_factory=dict)
    dependencies: dict[str, str] = field(default_factory=dict)


def provenance_field():
    return field(default_factory=Provenance, kw_only=True)


@dataclass(frozen=True, slots=True)
class DarkImage:
    """Subtraction-only image; temporal noise and bad pixels have not been measured."""

    mean_adu: Array = array("adu", "detector_y,detector_x")
    provenance: Provenance = provenance_field()


@dataclass(frozen=True, slots=True)
class DarkCalibration:
    """Temporal mean/RMS and the retained hot/dead/noisy bit flags from a stack."""

    mean_adu: Array = array("adu", "detector_y,detector_x")
    temporal_rms_adu: Array = array("adu", "detector_y,detector_x")
    bad_pixel_flags: Array = array("", "detector_y,detector_x")
    n_frames: int
    provenance: Provenance = provenance_field()


Dark = DarkCalibration | DarkImage


@dataclass(frozen=True, slots=True)
class FlatCalibration:
    """Flat response and extraction geometry, including paired (-1,-1) sentinels."""

    profile_x_pixels: Array = array("pixel", "output,sample")
    profile_y_pixels: Array = array("pixel", "output,sample")
    flat_map_adu: Array = array("adu", "detector_y,detector_x")
    trace_y_pixels: Array = array("pixel", "output,detector_x")
    provenance: Provenance = provenance_field()


@dataclass(frozen=True, slots=True)
class WavelengthCalibration:
    """Wavelength at each detector sample in the parent flat's extraction geometry."""

    wavelength_map_um: Array = array("um", "output,detector_x")
    provenance: Provenance = provenance_field()


@dataclass(frozen=True, slots=True)
class WavelengthReference:
    """Static laboratory wavelengths and per-line source labels."""

    wavelength_um: Array = array("um", "wavelength")
    origin: tuple[str, ...]
    provenance: Provenance = provenance_field()


@dataclass(frozen=True, slots=True)
class P2VMCalibration:
    """Full 24-output physical response, inverse matrix and retained fit diagnostics."""

    p2vm: Array = array("", "data_channel,output,wavelength")
    wavelength_um: Array = array("um", "wavelength")
    baseline_regions: dict[int, list[int]]
    baseline_telescopes: dict[int, tuple[int, int]]
    transmission: Array = array("", "output,telescope,wavelength")
    coherence_amplitude: Array = array("", "output,baseline,wavelength")
    coherence_phase_rad: Array = array("rad", "output,baseline,wavelength")
    opd_um: Array = array("um", "baseline,frame")
    group_delay_um: Array = array("um", "baseline,frame")
    ellipse_phase_rad: Array = array("rad", "baseline,frame,wavelength")
    ellipse_amplitude: Array = array("", "baseline,frame,wavelength")
    provenance: Provenance = provenance_field()


@dataclass(frozen=True, slots=True)
class VisibilityObservables:
    """Instrumental observables; even one frame retains a length-one frame axis."""

    photometric_flux_adu: Array = array("adu", "frame,telescope,wavelength")
    coherent_flux_adu: Array = array("adu", "frame,baseline,wavelength")
    visibility_amplitude: Array = array("", "frame,baseline,wavelength")
    flags: Array = array("", "frame,baseline,wavelength")
    wavelength_um: Array = array("um", "wavelength")
    baseline_regions: dict[int, list[int]]
    baseline_telescopes: dict[int, tuple[int, int]]
    provenance: Provenance = provenance_field()


@dataclass(frozen=True, slots=True)
class ReducedObservables(VisibilityObservables):
    """Complete instrumental science product, with measured GD and differential phase."""

    group_delay_um: Array = array("um", "frame,baseline")
    differential_phase_deg: Array = array("deg", "frame,baseline,wavelength")


@dataclass(frozen=True, slots=True)
class CalibrationSpectra:
    """In-memory spectra: telescope/baseline, output, frame, wavelength."""

    telescope_spectra: Array
    baseline_spectra: Array
    phase_reference: Array
    baseline_regions: dict[int, list[int]]
    baseline_telescopes: dict[int, tuple[int, int]]
    wavelength_um: Array
    provenance: Provenance = provenance_field()


def fingerprint(product) -> str:
    """Hash scientific content and parent dependencies, independent of file encoding."""
    digest = sha256()
    info = {
        "type": type(product).__name__,
        "parameters": product.provenance.parameters,
        "dependencies": product.provenance.dependencies,
        "instrument": [TELESCOPES, BASELINES, ABCD_ORDER],
    }
    for item in fields(product):
        if item.name == "provenance":
            continue
        value = getattr(product, item.name)
        if item.metadata:
            values = np.ascontiguousarray(value, dtype=value.dtype.newbyteorder("<"))
            info[item.name] = [values.dtype.str, values.shape, dict(item.metadata)]
            digest.update(values.tobytes())
        else:
            info[item.name] = value
    digest.update(
        json.dumps(
            info,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
            default=lambda value: value.item(),
        ).encode()
    )
    return digest.hexdigest()


def positive(value: float, name: str) -> float:
    """Validate a finite positive scientific parameter."""
    if isinstance(value, (bool, np.bool_)) or not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def validate_gd_range(search_range: tuple[float, float]) -> None:
    """Bounds describe coarse initialization; Newton refinement remains unconstrained."""
    low, high = search_range
    if not np.isfinite([low, high, high - low]).all() or not low < high:
        raise ValueError("GD search requires finite lower < upper bounds")
    if not np.isfinite(np.arange(low, high, 0.01)).all():
        raise ValueError("GD coarse grid must be finite")


def wavelength_grid(low=1.15, high=1.30, channels=151) -> Array:
    """Construct the explicit P2VM grid, with enough central-band GD samples."""
    positive(low, "wavelength minimum")
    positive(high, "wavelength maximum")
    if (
        not low < high
        or isinstance(channels, (bool, np.bool_))
        or not isinstance(channels, (int, np.integer))
    ):
        raise ValueError(
            "Wavelength grid requires minimum < maximum and integer channels"
        )
    if (3 * channels) // 4 - channels // 4 < 2:
        raise ValueError("P2VM requires at least two central-band wavelength samples")
    result = np.linspace(low, high, channels)
    validate_wavelength(result)
    return result


def validate_wavelength(wave: Array) -> None:
    """Reject nonfinite, nonpositive, repeated or reversed wavelength coordinates."""
    if (
        wave.ndim < 1
        or wave.shape[-1] < 2
        or not np.isfinite(wave).all()
        or not (wave > 0).all()
        or not (np.diff(wave, axis=-1) > 0).all()
    ):
        raise ValueError(
            "Wavelength coordinates must be finite, positive and strictly increasing"
        )


def require_parent(child, role: str, parent) -> None:
    """Reject same-shaped calibrations built from a different scientific parent."""
    if child.provenance.dependencies.get(role) != fingerprint(parent):
        raise ValueError(
            f"Incompatible {role} calibration dependency in {type(child).__name__}"
        )


def validate(product) -> None:
    """Check scientific shapes, index safety and the fixed instrumental conventions."""
    required = {
        FlatCalibration: ("dark",),
        WavelengthCalibration: ("flat",),
        P2VMCalibration: ("flat", "wave"),
    }
    for role in required.get(type(product), ()):
        value = product.provenance.dependencies.get(role, "")
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError(
                f"Missing scientific {role} dependency in {type(product).__name__}"
            )
    for item in fields(product):
        if item.metadata:
            value = getattr(product, item.name)
            if not isinstance(value, np.ndarray) or value.ndim != len(
                item.metadata["axes"].split(",")
            ):
                raise ValueError(
                    f"Invalid axes for {item.name}: {item.metadata['axes']}"
                )
            if value.dtype.kind not in "fiucb":
                raise ValueError(f"Non-numeric scientific array: {item.name}")
            if not isinstance(product, VisibilityObservables) and (
                np.iscomplexobj(value) or not np.isfinite(value).all()
            ):
                raise ValueError(
                    f"Calibration array must be finite and real: {item.name}"
                )
    if isinstance(product, (DarkImage, DarkCalibration)):
        shape = product.mean_adu.shape
        if not np.isfinite(product.mean_adu).all() or 0 in shape:
            raise ValueError("Dark image must be finite and nonempty")
        if isinstance(product, DarkCalibration):
            if (
                isinstance(product.n_frames, (bool, np.bool_))
                or not isinstance(product.n_frames, (int, np.integer))
                or product.n_frames < 2
                or product.temporal_rms_adu.shape != shape
                or product.bad_pixel_flags.shape != shape
            ):
                raise ValueError(
                    "Complete dark calibration requires matching mean/RMS/flags and >=2 frames"
                )
            if (
                product.bad_pixel_flags.dtype.kind not in "iu"
                or ((product.bad_pixel_flags < 0) | (product.bad_pixel_flags > 7)).any()
            ):
                raise ValueError(
                    "Dark bad-pixel flags must use the retained integer bits 1,2,4"
                )
            if (
                not np.isfinite(product.temporal_rms_adu).all()
                or (product.temporal_rms_adu < 0).any()
            ):
                raise ValueError("Temporal RMS must be finite and nonnegative")
    elif isinstance(product, FlatCalibration):
        ny, nx = product.flat_map_adu.shape
        xs, ys = product.profile_x_pixels, product.profile_y_pixels
        if (
            xs.shape != ys.shape
            or xs.shape[0] != N_OUTPUTS
            or product.trace_y_pixels.shape != (N_OUTPUTS, nx)
        ):
            raise ValueError(
                "Flat geometry must describe 24 outputs with matching X/Y indices"
            )
        if xs.dtype.kind not in "iu" or ys.dtype.kind not in "iu":
            raise ValueError("Extraction indices must be integers")
        valid = (xs >= 0) & (ys >= 0)
        if (
            not (((xs == -1) & (ys == -1)) | valid).all()
            or (xs[valid] >= nx).any()
            or (ys[valid] >= ny).any()
            or not valid.any(axis=1).all()
        ):
            raise ValueError("Invalid extraction indices or unpaired (-1,-1) sentinels")
    elif isinstance(product, WavelengthCalibration):
        if product.wavelength_map_um.shape[0] != N_OUTPUTS:
            raise ValueError("Wavelength map must describe 24 outputs")
        validate_wavelength(product.wavelength_map_um)
    elif isinstance(product, WavelengthReference):
        validate_wavelength(product.wavelength_um)
        if len(product.wavelength_um) < 3 or len(product.origin) != len(
            product.wavelength_um
        ):
            raise ValueError(
                "Static wavelengths require matching source labels and >=3 lines"
            )
    elif isinstance(product, P2VMCalibration):
        validate_wavelength(product.wavelength_um)
        nw, nf = len(product.wavelength_um), product.opd_um.shape[1]
        expected = {
            "p2vm": (N_DATA, N_OUTPUTS, nw),
            "transmission": (N_OUTPUTS, N_TELESCOPES, nw),
            "coherence_amplitude": (N_OUTPUTS, N_BASELINES, nw),
            "coherence_phase_rad": (N_OUTPUTS, N_BASELINES, nw),
            "opd_um": (N_BASELINES, nf),
            "group_delay_um": (N_BASELINES, nf),
            "ellipse_phase_rad": (N_BASELINES, nf, nw),
            "ellipse_amplitude": (N_BASELINES, nf, nw),
        }
        for name, shape in expected.items():
            if getattr(product, name).shape != shape:
                raise ValueError(f"Invalid {name} shape; expected {shape}")
        _validate_baselines(product)
    elif isinstance(product, VisibilityObservables):
        validate_wavelength(product.wavelength_um)
        nf, nt, nw = product.photometric_flux_adu.shape
        if nt != N_TELESCOPES or nw != len(product.wavelength_um):
            raise ValueError(
                "Photometry must describe four telescopes on the wavelength grid"
            )
        for name in ("coherent_flux_adu", "visibility_amplitude", "flags"):
            if getattr(product, name).shape != (nf, N_BASELINES, nw):
                raise ValueError(f"Invalid {name} shape")
        if product.flags.dtype != bool or not np.iscomplexobj(
            product.coherent_flux_adu
        ):
            raise ValueError(
                "Visibility flags must be boolean and coherent flux complex"
            )
        if isinstance(product, ReducedObservables) and (
            product.group_delay_um.shape != (nf, N_BASELINES)
            or product.differential_phase_deg.shape != (nf, N_BASELINES, nw)
        ):
            raise ValueError("Invalid science GD/phase axes")
        _validate_baselines(product)
    else:
        raise TypeError(f"Unsupported scientific object: {type(product).__name__}")


def _validate_baselines(product) -> None:
    expected = dict(enumerate(combinations(range(N_TELESCOPES), 2)))
    if dict(product.baseline_telescopes) != expected or set(
        product.baseline_regions
    ) != set(range(N_BASELINES)):
        raise ValueError("Baseline identities must match the fixed telescope ordering")
    rows = list(product.baseline_regions.values())
    if any(len(row) != 4 for row in rows) or sorted(
        x for row in rows for x in row
    ) != list(range(N_OUTPUTS)):
        raise ValueError("Each output must occur once in the six ABCD baselines")
