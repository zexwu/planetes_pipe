"""Typed configuration for the PLANETES reduction recipes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Self, TypeVar

import yaml


@dataclass(frozen=True, slots=True)
class FrameRange:
    """Half-open frame or detector-pixel range."""

    start: int = 0
    stop: int | None = None

    def __post_init__(self) -> None:
        if isinstance(self.start, bool) or not isinstance(self.start, int) or self.start < 0:
            raise ValueError("range start must be a non-negative integer")
        if self.stop is not None and (
            isinstance(self.stop, bool)
            or not isinstance(self.stop, int)
            or self.stop <= self.start
        ):
            raise ValueError(f"range stop must be greater than {self.start}")

    @property
    def slice(self) -> slice:
        return slice(self.start, self.stop)


@dataclass(frozen=True, slots=True)
class Crop:
    """Detector crop in FITS array coordinates."""

    x: FrameRange
    y: FrameRange


class ABCD(StrEnum):
    """Interferometric output labels."""

    A = "A"
    B = "B"
    C = "C"
    D = "D"


@dataclass(frozen=True, slots=True)
class InstrumentConfig:
    """Instrument ordering and dimensions used by all recipes."""

    telescopes: tuple[str, ...]
    baselines: tuple[str, ...]
    abcd_order: tuple[ABCD, ...]

    def __post_init__(self) -> None:
        if not self.telescopes or len(set(self.telescopes)) != len(self.telescopes):
            raise ValueError("instrument.telescopes must be non-empty and unique")
        if not self.baselines or len(set(self.baselines)) != len(self.baselines):
            raise ValueError("instrument.baselines must be non-empty and unique")
        expected = {
            self.telescopes[i] + self.telescopes[j]
            for i in range(len(self.telescopes))
            for j in range(i + 1, len(self.telescopes))
        }
        if set(self.baselines) != expected:
            raise ValueError(
                "instrument.baselines must contain each telescope pair exactly once; "
                f"expected {sorted(expected)}, got {sorted(self.baselines)}"
            )
        if len(self.abcd_order) != 4 or set(self.abcd_order) != set(ABCD):
            raise ValueError("instrument.abcd_order must contain A, B, C, and D once")

    @property
    def n_telescopes(self) -> int:
        return len(self.telescopes)

    @property
    def n_baselines(self) -> int:
        return len(self.baselines)

    @property
    def n_outputs(self) -> int:
        return 4 * self.n_baselines

    @property
    def n_data(self) -> int:
        return self.n_telescopes + 2 * self.n_baselines

    @property
    def real_slice(self) -> slice:
        return slice(self.n_telescopes, self.n_telescopes + self.n_baselines)

    @property
    def imag_slice(self) -> slice:
        return slice(self.n_telescopes + self.n_baselines, self.n_data)


@dataclass(frozen=True, slots=True)
class CubeInput:
    """One FITS cube, its optional dark, and selected frames."""

    data: Path
    dark: Path | None
    frames: FrameRange


@dataclass(frozen=True, slots=True)
class CalibrationInputs:
    """Raw calibration files and their frame range."""

    frames: FrameRange
    dark: Path
    flats: Mapping[str, Path]
    wavelength: CubeInput
    p2vm: Mapping[str, Path]
    phase_reference: CubeInput | None


@dataclass(frozen=True, slots=True)
class FlatParameters:
    profile_width: int = 10
    bad_threshold: float = 5.0

    def __post_init__(self) -> None:
        if self.profile_width <= 0 or self.bad_threshold <= 0:
            raise ValueError("flat profile width and bad-pixel threshold must be positive")


@dataclass(frozen=True, slots=True)
class WavelengthParameters:
    detection_sigma: float = 3.0
    surface_degree: int = 3
    aberration_degree: int = 3
    dispersion_degree: int = 3
    ransac_iterations: int = 1_000_000

    def __post_init__(self) -> None:
        degrees = self.surface_degree, self.aberration_degree, self.dispersion_degree
        if self.detection_sigma <= 0 or min(degrees) < 0 or self.ransac_iterations <= 0:
            raise ValueError("invalid wavelength calibration parameter")


@dataclass(frozen=True, slots=True)
class ResamplingParameters:
    wavelength_min_um: float = 1.15
    wavelength_max_um: float = 1.30
    wavelength_channels: int = 151

    def __post_init__(self) -> None:
        if self.wavelength_max_um <= self.wavelength_min_um:
            raise ValueError("resampling wavelength maximum must exceed the minimum")
        if self.wavelength_channels < 2:
            raise ValueError("resampling needs at least two wavelength channels")


@dataclass(frozen=True, slots=True)
class P2VMParameters:
    phase_correction: bool = True


@dataclass(frozen=True, slots=True)
class PlotParameters:
    full: bool = False


@dataclass(frozen=True, slots=True)
class RecipeParameters:
    flat: FlatParameters
    wavelength: WavelengthParameters
    resampling: ResamplingParameters
    p2vm: P2VMParameters
    plot: PlotParameters


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """Complete immutable pipeline configuration."""

    schema_version: int
    source: Path
    data_dir: Path
    output_dir: Path
    chunk_frames: int
    log_level: str
    automatic_plots: bool
    crop: Crop
    instrument: InstrumentConfig
    calibration: CalibrationInputs
    targets: Mapping[str, CubeInput]
    parameters: RecipeParameters

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError(f"Unsupported schema_version {self.schema_version}; expected 1")
        if self.chunk_frames <= 0:
            raise ValueError("chunk_frames must be positive")
        if not isinstance(self.automatic_plots, bool):
            raise ValueError("automatic_plots must be true or false")

    @classmethod
    def from_yaml(cls, path: Path | str) -> Self:
        """Load, validate, and resolve a YAML configuration."""
        source = Path(path).expanduser().resolve()
        if not source.exists():
            raise FileNotFoundError(f"Configuration file not found: {source}")
        raw = _mapping(yaml.safe_load(source.read_text(encoding="utf-8")), "root")
        _check_keys(raw, {field.name for field in fields(cls)} - {"source"}, "root")

        data_dir = _path(source.parent, raw.get("data_dir", "."), "data_dir")
        output_dir = _path(source.parent, raw.get("output_dir", "reduced"), "output_dir")

        crop_raw = _mapping(raw.get("crop"), "crop")
        _check_keys(crop_raw, {"x", "y"}, "crop")
        crop = Crop(
            x=_range(crop_raw.get("x", [0, None]), "crop.x"),
            y=_range(crop_raw.get("y", [0, None]), "crop.y"),
        )

        instrument_raw = _mapping(raw.get("instrument"), "instrument")
        _check_keys(
            instrument_raw, {"telescopes", "baselines", "abcd_order"}, "instrument"
        )
        telescopes = _names(instrument_raw.get("telescopes"), "instrument.telescopes")
        baselines = _names(instrument_raw.get("baselines"), "instrument.baselines")
        try:
            abcd_order = tuple(
                ABCD(str(label).upper())
                for label in _sequence(
                    instrument_raw.get("abcd_order", ["A", "C", "B", "D"]),
                    "instrument.abcd_order",
                )
            )
        except ValueError as exc:
            raise ValueError("instrument.abcd_order contains an invalid label") from exc
        instrument = InstrumentConfig(telescopes, baselines, abcd_order)

        calibration_raw = _mapping(raw.get("calibration"), "calibration")
        _check_keys(
            calibration_raw,
            {"frames", "dark", "flats", "wavelength", "p2vm", "phase_reference"},
            "calibration",
        )
        calibration_frames = _range(
            calibration_raw.get("frames", [0, None]), "calibration.frames"
        )
        wavelength = _cube(
            calibration_raw.get("wavelength"),
            data_dir,
            calibration_frames,
            "calibration.wavelength",
        )
        if wavelength.dark is None:
            raise ValueError("calibration.wavelength.dark is required")
        phase_raw = calibration_raw.get("phase_reference")
        calibration = CalibrationInputs(
            frames=calibration_frames,
            dark=_path(data_dir, calibration_raw.get("dark"), "calibration.dark"),
            flats=MappingProxyType(
                _named_paths(
                    calibration_raw.get("flats"),
                    data_dir,
                    telescopes,
                    "calibration.flats",
                )
            ),
            wavelength=wavelength,
            p2vm=MappingProxyType(
                _named_paths(
                    calibration_raw.get("p2vm"),
                    data_dir,
                    baselines,
                    "calibration.p2vm",
                )
            ),
            phase_reference=(
                None
                if phase_raw is None
                else _cube(
                    phase_raw,
                    data_dir,
                    calibration_frames,
                    "calibration.phase_reference",
                )
            ),
        )

        targets_raw = _mapping(raw.get("targets", {}), "targets")
        targets = MappingProxyType(
            {
                str(name): _cube(value, data_dir, calibration_frames, f"targets.{name}")
                for name, value in targets_raw.items()
            }
        )

        parameters_raw = _mapping(raw.get("parameters", {}), "parameters")
        _check_keys(
            parameters_raw,
            {"flat", "wavelength", "resampling", "p2vm", "plot"},
            "parameters",
        )
        parameters = RecipeParameters(
            flat=_build(FlatParameters, parameters_raw.get("flat", {}), "parameters.flat"),
            wavelength=_build(
                WavelengthParameters,
                parameters_raw.get("wavelength", {}),
                "parameters.wavelength",
            ),
            resampling=_build(
                ResamplingParameters,
                parameters_raw.get("resampling", {}),
                "parameters.resampling",
            ),
            p2vm=_build(P2VMParameters, parameters_raw.get("p2vm", {}), "parameters.p2vm"),
            plot=_build(PlotParameters, parameters_raw.get("plot", {}), "parameters.plot"),
        )

        return cls(
            schema_version=raw.get("schema_version"),
            source=source,
            data_dir=data_dir,
            output_dir=output_dir,
            chunk_frames=raw.get("chunk_frames", 16),
            log_level=str(raw.get("log_level", "INFO")).upper(),
            automatic_plots=raw.get("automatic_plots", False),
            crop=crop,
            instrument=instrument,
            calibration=calibration,
            targets=targets,
            parameters=parameters,
        )

    def as_mapping(self) -> dict[str, Any]:
        """Return a resolved, YAML-serializable configuration mapping."""
        return _plain(self, skip={"source"})

    def save(self, path: Path) -> None:
        """Write the resolved configuration used for a reduction."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.safe_dump(self.as_mapping(), sort_keys=False), encoding="utf-8"
        )

    def legacy_mapping(self) -> dict[str, Any]:
        """Return the old recipe mapping during the context-removal migration."""
        calibration: dict[str, Any] = {
            "dark": self.calibration.dark,
            "wave": self.calibration.wavelength.data,
            "wave_dark": self.calibration.wavelength.dark,
            "flat": [self.calibration.flats[name] for name in self.instrument.telescopes],
            "p2vm": [self.calibration.p2vm[name] for name in self.instrument.baselines],
        }
        if phase := self.calibration.phase_reference:
            calibration["wavesc"] = phase.data
            if phase.dark is not None:
                calibration["wavesc_dark"] = phase.dark
        targets = {
            name: {
                "data": target.data,
                **({"dark": target.dark} if target.dark is not None else {}),
            }
            for name, target in self.targets.items()
        }
        return {
            "frames": [self.calibration.frames.start, self.calibration.frames.stop],
            "xrange": [self.crop.x.start, self.crop.x.stop],
            "yrange": [self.crop.y.start, self.crop.y.stop],
            "output_dir": self.output_dir,
            "data_dir": self.data_dir,
            "log_level": self.log_level,
            "io_chunk_frames": self.chunk_frames,
            "plot_to_pdf": self.automatic_plots,
            "telescopes": list(self.instrument.telescopes),
            "baselines": list(self.instrument.baselines),
            "abcd_order": [label.value for label in self.instrument.abcd_order],
            "calib": calibration,
            "object": targets,
            "products": {
                "flat": "flat.npz",
                "p2vm": "p2vm.npz",
                "wave": "wave.npz",
                "wave_aberr": "wave_aberr.npz",
                "preproc": "{}_preproc.npz",
                "reduced": "{}_reduced.npz",
            },
            "parameters": {
                "flat": {
                    "profile_width": self.parameters.flat.profile_width,
                    "bad_thresh": self.parameters.flat.bad_threshold,
                },
                "wave": {
                    "sigma": self.parameters.wavelength.detection_sigma,
                    "deg": self.parameters.wavelength.surface_degree,
                    "aber_deg": self.parameters.wavelength.aberration_degree,
                    "disp_deg": self.parameters.wavelength.dispersion_degree,
                    "niter": self.parameters.wavelength.ransac_iterations,
                },
                "preproc": {
                    "nwave": self.parameters.resampling.wavelength_channels,
                    "min_wave": self.parameters.resampling.wavelength_min_um,
                    "max_wave": self.parameters.resampling.wavelength_max_um,
                    "object": "p2vm",
                },
                "p2vm": {"phase_correction": self.parameters.p2vm.phase_correction},
                "plot": {"full": self.parameters.plot.full},
            },
        }


ConfigT = TypeVar("ConfigT")


def _build(cls: type[ConfigT], value: Any, name: str) -> ConfigT:
    values = dict(_mapping(value, name))
    _check_keys(values, {field.name for field in fields(cls)}, name)
    return cls(**values)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a YAML mapping")
    return value


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a YAML list")
    return value


def _check_keys(value: Mapping[str, Any], allowed: set[str], name: str) -> None:
    if unknown := sorted(set(value) - allowed):
        raise ValueError(f"Unknown keys in {name}: {unknown}")


def _range(value: Any, name: str) -> FrameRange:
    values = _sequence(value, name)
    if len(values) != 2:
        raise ValueError(f"{name} must contain [start, stop]")
    return FrameRange(values[0], values[1])


def _names(value: Any, name: str) -> tuple[str, ...]:
    values = _sequence(value, name)
    if not values or not all(isinstance(item, str) and item for item in values):
        raise ValueError(f"{name} must contain non-empty strings")
    return tuple(values)


def _path(base: Path, value: Any, name: str) -> Path:
    if not isinstance(value, (str, Path)):
        raise ValueError(f"{name} must be a path")
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _named_paths(
    value: Any,
    base: Path,
    names: Sequence[str],
    section: str,
) -> dict[str, Path]:
    values = _mapping(value, section)
    if set(values) != set(names):
        raise ValueError(f"{section} keys must be {list(names)}")
    return {name: _path(base, values[name], f"{section}.{name}") for name in names}


def _cube(value: Any, base: Path, frames: FrameRange, name: str) -> CubeInput:
    values = _mapping(value, name)
    _check_keys(values, {"data", "dark", "frames"}, name)
    dark = values.get("dark")
    return CubeInput(
        data=_path(base, values.get("data"), f"{name}.data"),
        dark=None if dark is None else _path(base, dark, f"{name}.dark"),
        frames=frames if "frames" not in values else _range(values["frames"], f"{name}.frames"),
    )


def _plain(value: Any, *, skip: set[str] | None = None) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, FrameRange):
        return [value.start, value.stop]
    if is_dataclass(value) and not isinstance(value, type):
        excluded = skip or set()
        return {
            field.name: _plain(getattr(value, field.name))
            for field in fields(value)
            if field.name not in excluded
        }
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value
