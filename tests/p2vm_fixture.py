"""Deterministic four-telescope calibration signals for P2VM regression tests."""

from itertools import combinations
from pathlib import Path

import numpy as np

from planetes.instrument import BASELINES, TELESCOPES, DetectorSetup
from planetes.io.fits import write_product
from planetes.io.acquisition import CaptureMetadata, write_raw
from planetes.products import (
    P2VMSpectra,
    Dark,
    Flat,
    Wave,
)


def calibration_arrays(n_wave: int = 151, n_frame: int = 32):
    """Return noiseless ABCD spectra with known beam identities and optical paths."""
    wavelength = np.linspace(1.15e-6, 1.30e-6, n_wave)
    telescope_pairs = dict(enumerate(combinations(range(4), 2)))
    regions = {
        baseline: list(range(4 * baseline, 4 * baseline + 4)) for baseline in range(6)
    }
    telescope = np.zeros((4, 24, n_frame, n_wave))
    baseline = np.zeros((6, 24, n_frame, n_wave))
    phase_reference = np.full((24, n_frame, n_wave), 1000.0)
    abcd_phase = np.array([0.0, np.pi, np.pi / 2, 3 * np.pi / 2])
    scan = np.linspace(-7e-6, 7e-6, n_frame)
    time = np.linspace(-1.0, 1.0, n_frame)
    optical_paths = np.array([0.0, 1e-6, 1.8e-6, 3e-6])[:, None] * time
    for index, (first, second) in telescope_pairs.items():
        output = regions[index]
        telescope[first, output] = 500.0
        telescope[second, output] = 500.0
        phase = 2 * np.pi * scan[:, None] / wavelength
        baseline[index, output] = 1000.0 + 400.0 * np.cos(
            phase[None, :, :] + abcd_phase[:, None, None]
        )
        reference_phase = (
            2
            * np.pi
            * (optical_paths[second] - optical_paths[first])[:, None]
            / wavelength
        )
        phase_reference[output] += 400.0 * np.cos(
            reference_phase[None, :, :] + abcd_phase[:, None, None]
        )
    flat = Flat(
        profile_x_pixels=np.tile(np.arange(n_wave), (24, 1)),
        profile_y_pixels=np.repeat(np.arange(24)[:, None], n_wave, axis=1),
        flat_map_adu=np.ones((24, n_wave)),
        trace_y_pixels=np.repeat(np.arange(24)[:, None], n_wave, axis=1).astype(float),
        setup=setup(n_wave),
    )
    wave = Wave(
        wavelength_map_m=np.tile(wavelength, (24, 1)),
    )
    return (
        telescope,
        baseline,
        phase_reference,
        regions,
        telescope_pairs,
        wavelength,
        flat,
        wave,
    )


def setup(n_wave=151, height=24):
    return DetectorSetup(
        exposure_s=0.01,
        origin_x=0,
        origin_y=0,
        width=n_wave,
        height=height,
        binning_x=1,
        binning_y=1,
        detector="synthetic",
        orientation="native",
    )


def metadata(category, active=(), n_wave=151, height=24):
    return CaptureMetadata(
        category,
        tuple(str(n) in active for n in range(1, 5)),
        setup(n_wave, height),
        "command",
    )


def input_rows(paths):
    return [
        (tag, path)
        for tag, value in paths.items()
        for path in (value if isinstance(value, list) else [value])
    ]


def write_recipe_inputs(directory: Path):
    """Use repeated raw tags and real physical-shutter metadata, shuffled by callers."""
    telescope, baseline, reference, _, _, _, flat, wave = calibration_arrays()
    directory.mkdir(parents=True, exist_ok=True)
    paths = {}
    for tag, product in [("FLAT", flat), ("WAVE", wave)]:
        path = directory / f"{tag.lower()}.fits"
        write_product(path, product)
        paths[tag] = path
    dark = Dark(
        np.zeros((24, 151), dtype=np.float32),
        np.zeros((24, 151), dtype=np.float32),
        32,
        setup=setup(),
    )
    for tag in ("DARK", "PHASE_DARK_CALIB"):
        path = directory / f"{tag.lower()}.fits"
        write_product(path, dark)
        paths[tag] = path
    for tag, names, cubes in [
        ("FLAT_RAW", TELESCOPES, telescope),
        ("P2VM_RAW", BASELINES, baseline),
    ]:
        paths[tag] = []
        for name, spectra in zip(names, cubes):
            path = directory / f"{tag.lower()}_{name}.fits"
            write_raw(
                path, spectra.transpose(1, 0, 2).astype(np.float32), metadata(tag, name)
            )
            paths[tag].append(path)
    path = directory / "phase_reference.fits"
    write_raw(
        path,
        reference.transpose(1, 0, 2).astype(np.float32),
        metadata("SINGLE_SCI_RAW", "1234"),
    )
    paths["PHASE_REFERENCE"] = path
    return paths


def calibration_spectra(n_wave=15):
    """Array-level spectra with explicit telescope and baseline mappings."""
    values = calibration_arrays(n_wave=n_wave)
    return P2VMSpectra(*values[:4], values[5])
