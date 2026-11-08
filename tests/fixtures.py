"""Synthetic raw files and calibrations shared by the tests (four telescopes, six baselines)."""
from itertools import combinations
from pathlib import Path
from typing import NamedTuple

import numpy as np
from astropy.io import fits

from planetes_pipe.instrument import layout
from planetes_pipe.io.keywords import (
    DET_ID,
    DIT,
    DPR_TYPE,
    INSTRUME,
    WIN_BINX,
    WIN_BINY,
    WIN_NX,
    WIN_NY,
    WIN_STRX,
    WIN_STRY,
)
from planetes_pipe.io.raw import RAW_TYPES, write_shutters
from planetes_pipe.steps.dark import Bad, Dark, write_bad, write_dark
from planetes_pipe.steps.flat import Flat, write_flat
from planetes_pipe.steps.p2vm import P2VM, P2VMSpectra, write_p2vm
from planetes_pipe.steps.science import P2VMRed, write_reduced
from planetes_pipe.steps.wave import Wave, write_wave

WRITERS = {Dark: write_dark, Bad: write_bad, Flat: write_flat, Wave: write_wave,
           P2VM: write_p2vm, P2VMRed: write_reduced}


class RawMetadata(NamedTuple):
    """Header content of a synthetic raw file."""

    category: str
    shutters: tuple
    detector: fits.Header


def write_product(path: Path, product) -> None:
    WRITERS[type(product)](path, product)


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
        detector=detector_cards(n_wave),
        trace_x_range=np.tile([0, n_wave - 1], (24, 1)),
    )
    wave = Wave(wavelength_map_m=np.tile(wavelength, (24, 1)))
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


def cards(**values) -> fits.Header:
    """DET cards from Keyword=value pairs, e.g. cards(DIT=0.1, WIN_NX=5); None is left out."""
    keywords = {"DET_ID": DET_ID, "DIT": DIT, "WIN_STRX": WIN_STRX, "WIN_STRY": WIN_STRY,
                "WIN_NX": WIN_NX, "WIN_NY": WIN_NY, "WIN_BINX": WIN_BINX, "WIN_BINY": WIN_BINY}
    header = fits.Header()
    for name, value in values.items():
        if value is not None:
            keywords[name].set(header, value)
    return header


def detector_cards(n_wave=151, height=24) -> fits.Header:
    """The synthetic detector: full window, DIT 0.01 s."""
    return cards(DET_ID="synthetic", WIN_STRX=1, WIN_STRY=1, WIN_NX=n_wave, WIN_NY=height,
                 WIN_BINX=1, WIN_BINY=1, DIT=0.01)


def metadata(category, active=(), n_wave=151, height=24):
    return RawMetadata(
        category, tuple(str(n) in active for n in range(1, 5)), detector_cards(n_wave, height)
    )


def write_raw(path: Path, cube, physical: RawMetadata):
    """Write only the known physical cards needed by these synthetic fixtures."""
    header = fits.Header()
    INSTRUME.set(header, "PLANETES")
    DPR_TYPE.set(header, RAW_TYPES[physical.category])
    header.extend(physical.detector.cards)
    write_shutters(header, physical.shutters)
    image = fits.ImageHDU(cube, name="IMAGING_DATA_SC")
    image.header["BUNIT"] = "adu"
    hdus = [fits.PrimaryHDU(header=header), image]
    if physical.category == "P2VM_RAW":  # Scans record their baseline's OPD sign.
        moving = 1 + min(n for n, opened in enumerate(physical.shutters) if opened)  # "43" -> 3
        scan = fits.BinTableHDU.from_columns([
            fits.Column(name="FIRST_FRAME", format="K", array=[0]),
            fits.Column(name="NFRAMES", format="K", array=[len(cube)]),
            fits.Column(name="BEFORE_MM", format="D", array=[0.0]),
            fits.Column(name="AFTER_MM", format="D", array=[0.0]),
        ], name="DL_SCAN")
        scan.header["MOVINGDL"] = moving
        scan.header["OPDGAIN"] = -1.0  # None of the synthetic baselines is reversed.
        hdus.append(scan)
    fits.HDUList(hdus).writeto(path, checksum=True)


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
        detector=detector_cards(),
    )
    for tag in ("DARK", "PHASE_DARK_CALIB"):
        path = directory / f"{tag.lower()}.fits"
        write_dark(path, dark)
        paths[tag] = path
    for tag, names, cubes in [
        ("FLAT_RAW", layout.telescopes, telescope),
        ("P2VM_RAW", layout.baselines, baseline),
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
    return P2VMSpectra(values[0], values[1], values[3], values[5])
