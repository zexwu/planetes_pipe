"""Scientific arrays, units, axes and calibration relationships; no file or runner code."""

import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from hashlib import sha256
from itertools import combinations

import numpy as np
from numpy.typing import NDArray

from .instrument import (
    ABCD_ORDER,
    BASELINES,
    N_BASELINES,
    N_DATA,
    N_OUTPUTS,
    N_TELESCOPES,
    TELESCOPES,
    DetectorSetup,
)

Array = NDArray


def array(unit: str, axes: str, *, offset: int = 0):
    """Declare scientific units and Python axis order once, alongside the field."""
    return field(metadata={"unit": unit, "axes": axes, "offset": offset})


@dataclass(frozen=True, slots=True)
class Provenance:
    """Actual producer, settings, source roles and parent scientific fingerprints."""

    producer: str = "planetes"
    parameters: dict = field(default_factory=dict)
    inputs: dict[str, str] = field(default_factory=dict)
    dependencies: dict[str, str] = field(default_factory=dict)
    unverified: tuple[str, ...] = ()
    qc_status: dict[str, str] = field(default_factory=dict)


def provenance_field():
    return field(default_factory=Provenance, kw_only=True)


@dataclass(frozen=True, slots=True)
class DarkImage:
    """Subtraction-only image; temporal noise and bad pixels have not been measured."""

    mean_adu: Array = array("adu", "detector_y,detector_x")
    provenance: Provenance = provenance_field()


@dataclass(frozen=True, slots=True)
class DarkCalibration:
    """Measured temporal mean/population RMS and actual stack frame count."""

    mean_adu: Array = array("adu", "detector_y,detector_x")
    temporal_rms_adu: Array = array("adu", "detector_y,detector_x")
    n_frames: int
    setup: DetectorSetup | None = field(default=None, kw_only=True)
    provenance: Provenance = provenance_field()


@dataclass(frozen=True, slots=True)
class BadPixelMap:
    """Retained hot=1/dead=2/noisy=4 bits, derived from the named full DARK."""

    bad_pixel_flags: Array = array("", "detector_y,detector_x")
    provenance: Provenance = provenance_field()


Dark = DarkCalibration | DarkImage


@dataclass(frozen=True, slots=True)
class FlatCalibration:
    """Flat response and extraction geometry, including paired (-1,-1) sentinels."""

    profile_x_pixels: Array = array("pixel", "output,sample", offset=1)
    profile_y_pixels: Array = array("pixel", "output,sample", offset=1)
    flat_map_adu: Array = array("adu", "detector_y,detector_x")
    trace_y_pixels: Array = array("pixel", "output,detector_x", offset=1)
    profile_sigma_pixels: Array = field(
        default_factory=lambda: np.empty((0, 0)),
        metadata={"unit": "pixel", "axes": "output,detector_x"},
        kw_only=True,
    )
    telescope_flux_adu: Array = field(
        default_factory=lambda: np.empty(0),
        metadata={"unit": "adu", "axes": "telescope"},
        kw_only=True,
    )
    setup: DetectorSetup | None = field(default=None, kw_only=True)
    provenance: Provenance = provenance_field()


@dataclass(frozen=True, slots=True)
class WavelengthCalibration:
    """Wavelength at each detector sample in the parent flat's extraction geometry."""

    wavelength_map_m: Array = array("m", "output,detector_x")
    fit_residual_m: Array = field(
        default_factory=lambda: np.empty(0),
        metadata={"unit": "m", "axes": "line"},
        kw_only=True,
    )
    fit_x_pixels: Array = field(
        default_factory=lambda: np.empty(0),
        metadata={"unit": "pixel", "axes": "line"},
        kw_only=True,
    )
    fit_y_pixels: Array = field(
        default_factory=lambda: np.empty(0),
        metadata={"unit": "pixel", "axes": "line"},
        kw_only=True,
    )
    provenance: Provenance = provenance_field()


@dataclass(frozen=True, slots=True)
class WavelengthReference:
    """Static laboratory wavelengths and per-line source labels."""

    wavelength_m: Array = array("m", "wavelength")
    origin: tuple[str, ...]
    provenance: Provenance = provenance_field()


@dataclass(frozen=True, slots=True)
class P2VMCalibration:
    """Full 24-output instrumental response, inverse and fit diagnostics on one grid."""

    p2vm: Array = array("", "data_channel,output,wavelength")
    wavelength_m: Array = array("m", "wavelength")
    baseline_regions: dict[int, list[int]]
    baseline_telescopes: dict[int, tuple[int, int]]
    transmission: Array = array("", "output,telescope,wavelength")
    coherence_amplitude: Array = array("", "output,baseline,wavelength")
    coherence_phase_rad: Array = array("rad", "output,baseline,wavelength")
    opd_m: Array = array("m", "baseline,frame")
    group_delay_m: Array = array("m", "baseline,frame")
    ellipse_phase_rad: Array = array("rad", "baseline,frame,wavelength")
    ellipse_amplitude: Array = array("", "baseline,frame,wavelength")
    dc_term_adu: Array = field(
        default_factory=lambda: np.empty((0, 0, 0)),
        metadata={"unit": "adu", "axes": "output,baseline,wavelength"},
        kw_only=True,
    )
    fit_rms_adu: Array = field(
        default_factory=lambda: np.empty((0, 0, 0)),
        metadata={"unit": "adu", "axes": "output,baseline,wavelength"},
        kw_only=True,
    )
    provenance: Provenance = provenance_field()


@dataclass(frozen=True, slots=True)
class VisibilityObservables:
    """Per-frame instrumental photometry, complex coherent flux, VISAMP and flags."""

    photometric_flux_adu: Array = array("adu", "frame,telescope,wavelength")
    coherent_flux_adu: Array = array("adu", "frame,baseline,wavelength")
    visibility_amplitude: Array = array("", "frame,baseline,wavelength")
    flags: Array = array("", "frame,baseline,wavelength")
    wavelength_m: Array = array("m", "wavelength")
    baseline_regions: dict[int, list[int]]
    baseline_telescopes: dict[int, tuple[int, int]]
    provenance: Provenance = provenance_field()


@dataclass(frozen=True, slots=True)
class ReducedObservables(VisibilityObservables):
    """Per-frame instrumental P2VMRED including GD [m] and differential phase [deg]."""

    group_delay_m: Array = array("m", "frame,baseline")
    differential_phase_deg: Array = array("deg", "frame,baseline,wavelength")


@dataclass(frozen=True, slots=True)
class CalibrationSpectra:
    """In-memory spectra: telescope/baseline, output, frame, wavelength."""

    telescope_spectra: Array
    baseline_spectra: Array
    phase_reference: Array
    baseline_regions: dict[int, list[int]]
    baseline_telescopes: dict[int, tuple[int, int]]
    wavelength_m: Array
    provenance: Provenance = provenance_field()


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
    critical = {
        FlatCalibration: ("flat_map_adu", "trace_y_pixels"),
        WavelengthCalibration: ("wavelength_map_m",),
        P2VMCalibration: (
            "p2vm",
            "transmission",
            "coherence_amplitude",
            "coherence_phase_rad",
        ),
    }
    for name in critical.get(type(product), ()):
        if not np.isfinite(getattr(product, name)).all():
            raise ValueError(f"Calibration array must be finite: {name}")
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
            ):
                raise ValueError(
                    "Complete dark calibration requires matching mean/RMS and >=2 frames"
                )
            if (
                not np.isfinite(product.temporal_rms_adu).all()
                or (product.temporal_rms_adu < 0).any()
            ):
                raise ValueError("Temporal RMS must be finite and nonnegative")
    elif isinstance(product, BadPixelMap):
        flags = product.bad_pixel_flags
        if (
            0 in flags.shape
            or flags.dtype.kind not in "iu"
            or ((flags < 0) | (flags > 7)).any()
        ):
            raise ValueError("BAD must use the retained integer bits 1,2,4")
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
        if product.wavelength_map_m.shape[0] != N_OUTPUTS:
            raise ValueError("Wavelength map must describe 24 outputs")
        validate_wavelength(product.wavelength_map_m)
    elif isinstance(product, WavelengthReference):
        validate_wavelength(product.wavelength_m)
        if len(product.wavelength_m) < 3 or len(product.origin) != len(
            product.wavelength_m
        ):
            raise ValueError(
                "Static wavelengths require matching source labels and >=3 lines"
            )
    elif isinstance(product, P2VMCalibration):
        validate_wavelength(product.wavelength_m)
        nw, nf = len(product.wavelength_m), product.opd_m.shape[1]
        expected = {
            "p2vm": (N_DATA, N_OUTPUTS, nw),
            "transmission": (N_OUTPUTS, N_TELESCOPES, nw),
            "coherence_amplitude": (N_OUTPUTS, N_BASELINES, nw),
            "coherence_phase_rad": (N_OUTPUTS, N_BASELINES, nw),
            "opd_m": (N_BASELINES, nf),
            "group_delay_m": (N_BASELINES, nf),
            "ellipse_phase_rad": (N_BASELINES, nf, nw),
            "ellipse_amplitude": (N_BASELINES, nf, nw),
        }
        for name, shape in expected.items():
            if getattr(product, name).shape != shape:
                raise ValueError(f"Invalid {name} shape; expected {shape}")
        _validate_baselines(product)
    elif isinstance(product, VisibilityObservables):
        validate_wavelength(product.wavelength_m)
        nf, nt, nw = product.photometric_flux_adu.shape
        if nt != N_TELESCOPES or nw != len(product.wavelength_m):
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
            product.group_delay_m.shape != (nf, N_BASELINES)
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


def fingerprint(product) -> str:
    """Identify calibration contents and settings independently of paths and QC."""
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
            unit = item.metadata["unit"]
            if item.metadata.get("offset"):
                offset = item.metadata["offset"]
                value = (
                    np.where(value < 0, value, value + offset)
                    if value.dtype.kind in "iu"
                    else value + offset
                )
            values = np.ascontiguousarray(value, dtype=value.dtype.newbyteorder("<"))
            info[item.name] = [
                values.dtype.str,
                values.shape,
                unit,
                item.metadata["axes"],
            ]
            digest.update(values.tobytes())
        else:
            info[item.name] = asdict(value) if is_dataclass(value) else value
    digest.update(
        json.dumps(
            info,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
            default=lambda v: v.item(),
        ).encode()
    )
    return digest.hexdigest()
