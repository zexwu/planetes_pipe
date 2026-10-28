"""P2VMRED step: per-frame instrumental fluxes, visibility, group delay and phase."""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from . import log
from .extraction import SpectralExtractor, extract_spectra_from_file, prepare_extraction
from .instrument import (
    BASELINE_PAIRS,
    IMAG_SLICE,
    N_BASELINES,
    N_DATA,
    N_OUTPUTS,
    N_TELESCOPES,
    REAL_SLICE,
    _check_baselines,
)
from .io.fits import read_product
from .io.hdu import as_compute_array
from .io.raw import check_raw_calibrations, read_raw_metadata
from .products import (
    Dark,
    Flat,
    P2VM,
    P2VMRed,
    Wave,
)


def reduce_science(
    path: Path,
    dark: Dark,
    flat: Flat,
    wave: Wave,
    p2vm: P2VM,
    *,
    gd_search_min_m: float = -100e-6,
    gd_search_max_m: float = 100e-6,
) -> P2VMRed:
    """Apply selected calibrations to every raw frame and return instrumental P2VMRED.

    Extraction uses the P2VM's saved wavelength grid. No averaging is performed;
    fluxes are ADU, GD and its search bounds are metres, differential phase is deg.
    """

    raw = read_raw_metadata(path, category="SINGLE_SCI_RAW")
    check_raw_calibrations(raw, dark, flat)
    reducer = FrameReducer(flat, wave, p2vm)
    spectra = extract_spectra_from_file(path, reducer.extractor, dark)
    return reduce_spectra(
        spectra,
        p2vm,
        gd_search_min_m=gd_search_min_m,
        gd_search_max_m=gd_search_max_m,
    )


def reduce_spectra(
    spectra: NDArray,
    p2vm: P2VM,
    *,
    gd_search_min_m: float = -100e-6,
    gd_search_max_m: float = 100e-6,
    include_phase: bool = True,
) -> P2VMRed:
    """Apply P2VM; include_phase adds group delay and differential phase."""
    _check_baselines(p2vm.baseline_outputs)
    if (
        p2vm.wavelength_m.ndim != 1
        or p2vm.matrix.shape != (N_DATA, N_OUTPUTS, len(p2vm.wavelength_m))
        or spectra.ndim != 3
        or spectra.shape[0] != N_OUTPUTS
        or spectra.shape[-1] != len(p2vm.wavelength_m)
    ):
        raise ValueError("Science spectra and P2VM output/wavelength dimensions differ")
    log.info("Applying P2VM to extract visibility for science spectra...")
    reconstructed_data = np.einsum("box,ofx->fbx", p2vm.matrix, spectra)
    photometric_flux_adu = reconstructed_data[:, :N_TELESCOPES, :]
    coherent_flux_adu = (
        reconstructed_data[:, REAL_SLICE, :] + 1j * reconstructed_data[:, IMAG_SLICE, :]
    )
    visibility_amplitude, flags = normalize_coherent_flux(
        coherent_flux_adu,
        photometric_flux_adu,
    )

    if not include_phase:
        return P2VMRed(
            photometric_flux_adu,
            coherent_flux_adu,
            visibility_amplitude,
            flags,
            p2vm.wavelength_m,
            p2vm.baseline_outputs,
        )

    group_delay_m = np.zeros((len(coherent_flux_adu), N_BASELINES))
    log.info("Computing group delay for science spectra...")
    for baseline in range(N_BASELINES):
        group_delay_m[:, baseline] = estimate_group_delay(
            coherent_flux_adu[:, baseline],
            p2vm.wavelength_m,
            search_range=(gd_search_min_m, gd_search_max_m),
        )

    differential_phase_deg = coherent_flux_adu * np.exp(
        -2j * np.pi * group_delay_m[:, :, None] / p2vm.wavelength_m[None, :]
    )
    differential_phase_deg *= differential_phase_deg.mean(axis=2)[:, :, None].conj()
    differential_phase_deg = np.angle(differential_phase_deg, deg=True)
    differential_phase_deg -= differential_phase_deg.mean(axis=2)[:, :, None]
    differential_phase_deg = (differential_phase_deg + 180) % 360 - 180

    product = P2VMRed(
        photometric_flux_adu=photometric_flux_adu,
        coherent_flux_adu=coherent_flux_adu,
        visibility_amplitude=visibility_amplitude,
        flags=flags,
        wavelength_m=p2vm.wavelength_m,
        group_delay_m=group_delay_m,
        differential_phase_deg=differential_phase_deg,
        baseline_outputs=p2vm.baseline_outputs,
    )
    log.info("Reduction for science spectra completed.")

    log.info("--- Step: REDUCE science spectra [DONE] ---")
    return product


def normalize_coherent_flux(
    coherent_flux: NDArray,
    flux: NDArray,
) -> tuple[NDArray, NDArray]:
    """Normalize the six fixed baselines and flag non-positive/non-finite denominators."""
    visamp = np.full(coherent_flux.shape, np.nan)
    flag = np.ones(coherent_flux.shape, dtype=bool)

    for baseline, (telescope_1, telescope_2) in enumerate(BASELINE_PAIRS):
        flux_product = flux[:, telescope_1, :] * flux[:, telescope_2, :]
        valid = (
            (flux_product > 0)
            & np.isfinite(flux_product)
            & np.isfinite(coherent_flux[:, baseline, :])
        )
        visamp[:, baseline, :][valid] = np.abs(
            coherent_flux[:, baseline, :][valid]
        ) / np.sqrt(flux_product[valid])
        flag[:, baseline, :] = ~valid

    return visamp, flag


def estimate_group_delay(
    visdata: NDArray,
    wavelength: NDArray,
    search_range: tuple[float, float] = (-30e-6, 30e-6),
    n_newton: int = 5,
) -> NDArray:
    """Estimate group delay by grid search followed by Newton refinement."""
    k = -2j * np.pi / wavelength
    w = -2 * np.pi / wavelength

    gd_grid = np.arange(search_range[0], search_range[1], 1e-8)
    phasors_grid = np.exp(gd_grid[:, None] * k[None, :])
    coherence = np.abs(np.tensordot(visdata, phasors_grid, axes=([-1], [-1])))

    best = np.argmax(coherence, axis=-1)
    gd_current = gd_grid[best]

    for _ in range(n_newton):
        gd_exp = gd_current[:, None]
        rotated = visdata * np.exp(1j * w * gd_exp)
        sum_0 = np.sum(rotated, axis=-1)
        sum_1 = np.sum(rotated * (1j * w), axis=-1)
        sum_2 = np.sum(rotated * (-(w**2)), axis=-1)

        sum_0_conjugate = np.conj(sum_0)
        gradient = 2 * np.real(sum_1 * sum_0_conjugate)
        hessian = 2 * np.real(sum_2 * sum_0_conjugate + sum_1 * np.conj(sum_1))
        difference = np.zeros_like(gradient)
        valid = hessian != 0
        difference[valid] = gradient[valid] / hessian[valid]
        gd_current = gd_current - difference

    return gd_current


@dataclass(slots=True)
class FrameReducer:
    """Load/prepare calibration once; raw unbinned frames always use the formal core."""

    flat: Flat
    wave: Wave
    p2vm: P2VM
    extractor: SpectralExtractor = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.extractor = prepare_extraction(
            self.flat, self.wave, self.p2vm.wavelength_m
        )

    @property
    def n_baselines(self) -> int:
        return N_BASELINES

    @property
    def wavelength_m(self) -> NDArray:
        return self.extractor.wavelength_m

    def reduce_frame(
        self,
        raw: NDArray,
        *,
        dark: Dark,
        include_phase: bool = False,
        gd_search_min_m: float = -100e-6,
        gd_search_max_m: float = 100e-6,
    ) -> P2VMRed:
        """Reduce one raw frame, subtracting dark once; phase outputs are optional."""
        cube = as_compute_array(raw)[None, :, :]
        spectra = self.extractor.extract(cube, dark)
        return reduce_spectra(
            spectra,
            self.p2vm,
            include_phase=include_phase,
            gd_search_min_m=gd_search_min_m,
            gd_search_max_m=gd_search_max_m,
        )


def load_reducer(directory: Path) -> FrameReducer:
    """Read a calibration directory and prepare one array-only live reducer."""
    directory = Path(directory).expanduser().resolve()
    return FrameReducer(
        read_product(directory / "planetes_flat.fits"),
        read_product(directory / "planetes_wave.fits"),
        read_product(directory / "planetes_p2vm.fits"),
    )
