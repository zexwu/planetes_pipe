"""Deterministic four-telescope calibration signals for P2VM regression tests."""

from itertools import combinations
from pathlib import Path

import numpy as np
from astropy.io import fits

from planetes.instrument import BASELINES, TELESCOPES
from planetes.io.products import write_product
from planetes.product import FlatProduct, WaveProduct


def calibration_arrays(n_wave: int = 151, n_frame: int = 32):
    """Return noiseless ABCD spectra with known beam identities and optical paths."""
    wavelength = np.linspace(1.15, 1.30, n_wave)
    telescope_pairs = dict(enumerate(combinations(range(4), 2)))
    regions = {
        baseline: list(range(4 * baseline, 4 * baseline + 4)) for baseline in range(6)
    }
    telescope = np.zeros((4, 24, n_frame, n_wave))
    baseline = np.zeros((6, 24, n_frame, n_wave))
    phase_reference = np.full((24, n_frame, n_wave), 1000.0)
    abcd_phase = np.array([0.0, np.pi, np.pi / 2, 3 * np.pi / 2])
    scan = np.linspace(-7.0, 7.0, n_frame)
    time = np.linspace(-1.0, 1.0, n_frame)
    optical_paths = np.array([0.0, 1.0, 1.8, 3.0])[:, None] * time
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
    flat = FlatProduct(
        profile_x_pixels=np.tile(np.arange(n_wave), (24, 1)),
        profile_y_pixels=np.repeat(np.arange(24)[:, None], n_wave, axis=1),
        flat_map_adu=np.ones((24, n_wave)),
        dark_map_adu=np.zeros((24, n_wave)),
        bad_pixel_flags=np.zeros((24, n_wave), dtype=np.int16),
        trace_y_pixels=np.repeat(np.arange(24)[:, None], n_wave, axis=1).astype(float),
    )
    wave = WaveProduct(wavelength_map_um=np.tile(wavelength, (24, 1)))
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


def write_recipe_inputs(directory: Path) -> dict[str, Path]:
    """Write compact synthetic detector cubes and typed upstream calibration FITS."""
    telescope, baseline, reference, _, _, _, flat, wave = calibration_arrays()
    directory.mkdir(parents=True, exist_ok=True)
    paths = {}
    header = fits.Header({"INSTRUME": "PLANETES", "ORIGIN": "PLANETES-SIMULATION"})
    for tag, product in [("PLANETES_FLAT", flat), ("PLANETES_WAVE", wave)]:
        path = directory / f"{tag.lower()}.fits"
        write_product(path, product, header=header)
        paths[tag] = path
    cubes = {
        f"TEL_FLAT_{name}": telescope[index] for index, name in enumerate(TELESCOPES)
    }
    cubes.update(
        {f"P2VM_RAW_{name}": baseline[index] for index, name in enumerate(BASELINES)}
    )
    cubes["PHASE_REFERENCE"] = reference
    cubes["PHASE_DARK"] = np.zeros_like(reference)
    for tag, spectra in cubes.items():
        path = directory / f"{tag.lower()}.fits"
        raw_header = header.copy()
        raw_header["HIERARCH ESO DPR CATG"] = "CALIB"
        raw_header["HIERARCH ESO DPR TYPE"] = tag
        raw_header["HIERARCH ESO DPR TECH"] = "INTERFEROMETRY"
        raw_header["ARCFILE"] = path.name
        cube = np.transpose(spectra, (1, 0, 2)).astype(np.float32)
        fits.HDUList([fits.PrimaryHDU(header=raw_header), fits.ImageHDU(cube)]).writeto(
            path, checksum=True
        )
        paths[tag] = path
    return paths
