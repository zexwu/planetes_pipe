"""Small path configuration for PLANETES reductions."""

from __future__ import annotations

import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import yaml

from .instrument import BASELINES, TELESCOPES


@dataclass(frozen=True, slots=True)
class Dataset:
    """One detector dataset and its optional dark cube."""

    data: Path
    dark: Path | None = None


@dataclass(frozen=True, slots=True)
class WavelengthInput:
    """Wavelength-lamp cube and its required dark cube."""

    data: Path
    dark: Path


@dataclass(frozen=True, slots=True)
class CalibrationInputs:
    """Raw files used to derive the calibration products."""

    dark: Path
    flats: dict[str, Path]
    wavelength: WavelengthInput
    p2vm: dict[str, Path]
    phase_reference: Dataset | None


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """Input/output paths and the one retained calibration mode switch."""

    source: Path
    data_dir: Path
    output_dir: Path
    calibration: CalibrationInputs
    targets: dict[str, Dataset]
    phase_correction: bool = True

    @classmethod
    def from_yaml(cls, path: Path | str) -> Self:
        source = Path(path).expanduser().resolve()
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping):
            raise ValueError("Configuration root must be a YAML mapping")
        if raw.get("schema_version") != 1:
            raise ValueError("Unsupported configuration schema_version")

        data_dir = _resolve(source.parent, raw.get("data_dir", "."))
        calibration = raw["calibration"]
        phase_reference = calibration.get("phase_reference")
        phase_correction = raw.get("phase_correction", True)
        if not isinstance(phase_correction, bool):
            raise ValueError("phase_correction must be true or false")

        return cls(
            source=source,
            data_dir=data_dir,
            output_dir=_resolve(source.parent, raw.get("output_dir", "reduced")),
            calibration=CalibrationInputs(
                dark=_resolve(data_dir, calibration["dark"]),
                flats=_named_paths(data_dir, calibration["flats"], TELESCOPES),
                wavelength=_wavelength_input(data_dir, calibration["wavelength"]),
                p2vm=_named_paths(data_dir, calibration["p2vm"], BASELINES),
                phase_reference=(
                    None
                    if phase_reference is None
                    else _dataset(data_dir, phase_reference)
                ),
            ),
            targets={
                str(name): _dataset(data_dir, value)
                for name, value in raw.get("targets", {}).items()
            },
            phase_correction=phase_correction,
        )

    def copy_used_config(self, path: Path) -> None:
        """Copy the exact user configuration used for this run."""
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.source, path)

    def target(self, name: str) -> Dataset:
        """Return one configured target with a useful unknown-name error."""
        try:
            return self.targets[name]
        except KeyError as exc:
            available = ", ".join(self.targets)
            raise ValueError(
                f"Unknown target {name!r}; available targets: {available}"
            ) from exc


def _dataset(base: Path, value: Mapping[str, Any]) -> Dataset:
    dark = value.get("dark")
    return Dataset(
        data=_resolve(base, value["data"]),
        dark=None if dark is None else _resolve(base, dark),
    )


def _wavelength_input(base: Path, value: Mapping[str, Any]) -> WavelengthInput:
    return WavelengthInput(
        data=_resolve(base, value["data"]),
        dark=_resolve(base, value["dark"]),
    )


def _named_paths(
    base: Path, values: Mapping[str, Any], names: tuple[str, ...]
) -> dict[str, Path]:
    return {name: _resolve(base, values[name]) for name in names}


def _resolve(base: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()
