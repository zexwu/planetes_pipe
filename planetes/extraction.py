"""Array-only spectral extraction and wavelength resampling."""

from dataclasses import dataclass
from pathlib import Path
from .fits import iter_raw_cube
from .instrument import BASELINES, TELESCOPES, N_OUTPUTS, N_TELESCOPES
from .models import (
    CalibrationSpectra,
    Provenance,
    fingerprint,
    require_parent,
    validate,
    validate_wavelength,
)


import numpy as np
from numba import njit, prange
from numpy.typing import NDArray


@njit(parallel=True, fastmath=True, cache=True)
def extract_spec_sparse(cube: NDArray, ys: NDArray, xs: NDArray) -> NDArray:
    """
    Extract spectra from a cube using sparse pixel indices per output channel.

    Args:
        cube (NDArray): Input cube with shape `(n_frame, ny, nx)`.
        ys (NDArray): Per-output Y indices; negative values are invalid sentinels.
        xs (NDArray): Per-output X indices; negative values are invalid sentinels.

    Returns:
        NDArray: Extracted spectra with shape `(n_reg, n_frame, nx)`.
    """
    n_frame, _, nx = cube.shape
    n_reg = len(ys)
    output = np.zeros((n_reg, n_frame, nx), dtype=np.float32)

    for o in prange(n_reg):
        y_idx = ys[o]
        x_idx = xs[o]
        for f in range(n_frame):
            for i in range(len(y_idx)):
                x = x_idx[i]
                y = y_idx[i]
                if x < 0 or y < 0:
                    continue
                output[o, f, x] += cube[f, y, x]
    return output


@njit(parallel=True, cache=True)
def twopx_interp(spec: NDArray, wave_map: NDArray, wl_grid: NDArray) -> NDArray:
    """
    Resample spectra from detector pixels onto a uniform wavelength grid.

    Args:
        spec (NDArray): Input spectra with shape `(n_reg, n_frame, n_pixels)`.
        wave_map (NDArray): Wavelength at each detector pixel with shape `(n_reg, n_pixels)`.
        wl_grid (NDArray): Target wavelength grid with shape `(n_wave,)`.

    Returns:
        NDArray: Resampled spectra with shape `(n_reg, n_frame, n_wave)`.
    """
    n_reg, n_frame, _ = spec.shape
    n_wave = len(wl_grid)

    wn_map = 1.0 / wave_map
    wn_grid = 1.0 / wl_grid

    out = np.zeros((n_reg, n_frame, n_wave))

    for r in prange(n_reg):
        # We reverse (::-1) because np.interp expects increasing X values
        # and wavenumber usually decreases as pixel index increases.
        x_in = wn_map[r, ::-1]

        for f in range(n_frame):
            y_in = spec[r, f, ::-1]
            out[r, f, :] = np.interp(wn_grid, x_in, y_in)

    return out


def _snapshot(values):
    result = np.array(values, copy=True)
    result.flags.writeable = False
    return result


@dataclass(frozen=True, slots=True)
class SpectralExtractor:
    """Prepared, read-only extraction geometry and response on one explicit target grid."""

    profile_x_pixels: NDArray
    profile_y_pixels: NDArray
    detector_shape: tuple[int, int]
    setup: object
    wavelength_map_um: NDArray
    wavelength_um: NDArray
    detector_flat_spectra: NDArray
    resampled_flat_spectra: NDArray

    def extract(self, cube: NDArray, dark) -> NDArray:
        """Subtract one explicitly supplied dark and extract every frame of this chunk."""
        if (
            cube.ndim != 3
            or cube.shape[1:] != self.detector_shape
            or dark.mean_adu.shape != self.detector_shape
        ):
            raise ValueError("Raw cube, dark and extraction detector geometry differ")
        spec = extract_spec_sparse(
            cube - dark.mean_adu, self.profile_y_pixels, self.profile_x_pixels
        )
        aligned = twopx_interp(
            spec / self.detector_flat_spectra,
            self.wavelength_map_um,
            self.wavelength_um,
        )
        aligned *= self.resampled_flat_spectra
        return aligned

    def extract_file(self, path: Path, dark) -> NDArray:
        """Read and extract a whole file, retaining all frames and the 16-frame chunking."""
        from .fits import check_raw_calibrations

        check_raw_calibrations(path, dark, flat_setup=self.setup)
        return np.concatenate(
            [self.extract(chunk, dark) for chunk in iter_raw_cube(path)], axis=1
        )


def prepare_extraction(flat, wave, wavelength_um: NDArray) -> SpectralExtractor:
    """Validate scientific parents and coverage once, then snapshot the required arrays."""
    validate(flat)
    validate(wave)
    validate_wavelength(wavelength_um)
    require_parent(wave, "flat", flat)
    if wave.wavelength_map_um.shape[1] != flat.flat_map_adu.shape[1]:
        raise ValueError("Flat and wavelength detector dimensions differ")
    for output, row in enumerate(wave.wavelength_map_um):
        if wavelength_um[0] < row[0] or wavelength_um[-1] > row[-1]:
            raise ValueError(
                f"Output {output}: target {wavelength_um[0]:.9g}..{wavelength_um[-1]:.9g} um outside calibration {row[0]:.9g}..{row[-1]:.9g} um"
            )
    xs, ys = _snapshot(flat.profile_x_pixels), _snapshot(flat.profile_y_pixels)
    wave_map, grid = _snapshot(wave.wavelength_map_um), _snapshot(wavelength_um)
    spec_flat_detector = extract_spec_sparse(flat.flat_map_adu[None, :, :], ys, xs)
    spec_flat_detector = np.clip(spec_flat_detector, a_min=1e-8, a_max=None)
    spec_flat = 1 / twopx_interp(1 / spec_flat_detector, wave_map, grid)
    return SpectralExtractor(
        xs,
        ys,
        flat.flat_map_adu.shape,
        flat.setup,
        wave_map,
        grid,
        _snapshot(spec_flat_detector),
        _snapshot(spec_flat),
    )


def extract_calibration_files(
    flat,
    wave,
    main_dark,
    *,
    telescope_flats,
    baseline_cubes,
    wavelength_um,
    phase_reference=None,
    phase_dark=None,
) -> CalibrationSpectra:
    """Extract calibration files and infer the retained physical output mapping."""
    from .fits import identify_raw, check_raw_calibrations, read_raw_metadata
    from .models import DarkCalibration

    if not isinstance(main_dark, DarkCalibration) or (
        phase_dark is not None and not isinstance(phase_dark, DarkCalibration)
    ):
        raise ValueError("Formal file tasks require complete DARK calibrations")
    telescope_flats = identify_raw(telescope_flats, "FLAT_RAW")
    baseline_cubes = identify_raw(baseline_cubes, "P2VM_RAW")
    validate(main_dark)
    if phase_dark is not None and phase_reference is None:
        raise ValueError("phase_dark requires phase_reference")
    model = prepare_extraction(flat, wave, wavelength_um)
    spec_bsl = np.asarray(
        [model.extract_file(baseline_cubes[name], main_dark) for name in BASELINES]
    )
    spec_tel = np.asarray(
        [model.extract_file(telescope_flats[name], main_dark) for name in TELESCOPES]
    )
    phase_reference_spectra = np.empty(
        (N_OUTPUTS, 0, len(model.wavelength_um)), dtype=np.float64
    )
    dependencies = {
        "flat": fingerprint(flat),
        "wave": fingerprint(wave),
        "main_dark": fingerprint(main_dark),
    }
    unknown = []
    for path in [*telescope_flats.values(), *baseline_cubes.values()]:
        unknown.extend(check_raw_calibrations(path, main_dark, flat))
    inputs = {
        **{f"tel_flat_{k}": str(v) for k, v in telescope_flats.items()},
        **{f"baseline_{k}": str(v) for k, v in baseline_cubes.items()},
    }
    if phase_reference is not None:
        read_raw_metadata(phase_reference, category="SINGLE_SCI_RAW")
        selected_dark = main_dark if phase_dark is None else phase_dark
        phase_reference_spectra = model.extract_file(phase_reference, selected_dark)
        unknown.extend(check_raw_calibrations(phase_reference, selected_dark, flat))
        dependencies["phase_dark"] = fingerprint(selected_dark)
        inputs["phase_reference"] = str(phase_reference)
    tel_regs = {}
    for telescope, flux in enumerate(spec_tel):
        mean_flux = flux.mean(axis=(1, 2))
        active = np.argsort(mean_flux)[-N_OUTPUTS // 2 :]
        tel_regs[telescope] = np.sort(active).tolist()

    bsl_regs = {}
    for baseline, flux in enumerate(spec_bsl):
        std_flux = flux.std(axis=1).mean(axis=-1)
        active = np.argsort(std_flux)[-4:]
        bsl_regs[baseline] = np.sort(active).tolist()

    bsl_to_reg = {}
    bsl_to_tel = {}
    for telescope_1 in range(N_TELESCOPES):
        for telescope_2 in range(N_TELESCOPES):
            if telescope_1 >= telescope_2:
                continue
            baseline_name = TELESCOPES[telescope_1] + TELESCOPES[telescope_2]
            baseline = BASELINES.index(baseline_name)
            from_telescopes = set(tel_regs[telescope_1]) & set(tel_regs[telescope_2])
            from_modulation = set(bsl_regs[baseline])
            if from_telescopes != from_modulation:
                raise ValueError(
                    f"Region mapping mismatch for baseline {baseline_name}: "
                    f"{sorted(from_telescopes)} from flats, "
                    f"{sorted(from_modulation)} from modulation"
                )
            bsl_to_reg[baseline] = sorted(bsl_regs[baseline])
            bsl_to_tel[baseline] = (telescope_1, telescope_2)

    return CalibrationSpectra(
        spec_tel,
        spec_bsl,
        phase_reference_spectra,
        bsl_to_reg,
        bsl_to_tel,
        model.wavelength_um,
        provenance=Provenance(
            producer="planetes.extraction",
            inputs=inputs,
            dependencies=dependencies,
            unverified=tuple(sorted(set(unknown))),
        ),
    )
