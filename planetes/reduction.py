"""
Science data reduction module for the PLANETES pipeline.

This module handles the reduction of science target data, including
visibility extraction, group delay computation, and phase calculation.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from . import log
from .extraction import extract_file, prepare_extraction
from .instrument import IMAG_SLICE, N_BASELINES, N_TELESCOPES, REAL_SLICE
from .io.fits import read_product
from .io.raw import check_raw_calibrations
from .products import (
    P2VMCalibration,
    Provenance,
    ReducedObservables,
    VisibilityObservables,
    fingerprint,
    require_parent,
    validate,
    validate_gd_range,
)

PARAMETERS = {
    "gd_search_min_m": (
        "planetes.reduce.gd-search-min-m",
        "gd-search-min-m",
        "Coarse science GD lower initialization [m]",
    ),
    "gd_search_max_m": (
        "planetes.reduce.gd-search-max-m",
        "gd-search-max-m",
        "Coarse science GD upper initialization [m]",
    ),
}


def reduce_file(
    path: Path,
    dark,
    flat,
    wave,
    p2vm: P2VMCalibration,
    *,
    gd_search_min_m=-100e-6,
    gd_search_max_m=100e-6,
) -> ReducedObservables:
    """Read the complete raw science cube, apply explicit calibrations and return results."""

    unknown = check_raw_calibrations(path, dark, flat, category="SINGLE_SCI_RAW")
    validate(dark)
    reducer = FrameReducer(flat, wave, p2vm)
    spectra = extract_file(path, reducer.extractor, dark)
    result = reduce_spectra(
        spectra,
        p2vm,
        gd_search_min_m=gd_search_min_m,
        gd_search_max_m=gd_search_max_m,
    )
    info = Provenance(
        producer="planetes.reduction",
        unverified=unknown,
        parameters={
            "gd_search_min_m": gd_search_min_m,
            "gd_search_max_m": gd_search_max_m,
        },
        inputs={"science_raw": str(path)},
        dependencies={
            "dark": fingerprint(dark),
            "flat": fingerprint(flat),
            "wave": fingerprint(wave),
            "p2vm": fingerprint(p2vm),
        },
    )
    return replace(result, provenance=info)


def reduce_spectra(
    spectra: NDArray,
    p2vm: P2VMCalibration,
    *,
    gd_search_min_m=-100e-6,
    gd_search_max_m=100e-6,
    diagnostics=True,
):
    """Apply P2VM and calculate the retained instrumental visibility observables."""
    log.info("Applying P2VM to extract visibility for science spectra...")
    p2vmred = np.einsum("box,ofx->fbx", p2vm.p2vm, spectra)
    photometric_flux_adu = p2vmred[:, :N_TELESCOPES, :]
    coherent_flux_adu = p2vmred[:, REAL_SLICE, :] + 1j * p2vmred[:, IMAG_SLICE, :]
    visibility_amplitude, flags = normalize_visibility(
        coherent_flux_adu,
        photometric_flux_adu,
        p2vm.baseline_telescopes,
    )

    if not diagnostics:
        return VisibilityObservables(
            photometric_flux_adu,
            coherent_flux_adu,
            visibility_amplitude,
            flags,
            p2vm.wavelength_m,
            p2vm.baseline_regions,
            p2vm.baseline_telescopes,
        )

    group_delay_m = np.zeros((len(coherent_flux_adu), N_BASELINES))
    log.info("Computing group delay for science spectra...")
    for baseline in range(N_BASELINES):
        group_delay_m[:, baseline] = compute_gdelay(
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

    product = ReducedObservables(
        photometric_flux_adu=photometric_flux_adu,
        coherent_flux_adu=coherent_flux_adu,
        visibility_amplitude=visibility_amplitude,
        flags=flags,
        wavelength_m=p2vm.wavelength_m,
        group_delay_m=group_delay_m,
        differential_phase_deg=differential_phase_deg,
        baseline_regions=p2vm.baseline_regions,
        baseline_telescopes=p2vm.baseline_telescopes,
    )
    log.info("Reduction for science spectra completed.")

    log.info("--- Step: REDUCE science spectra [DONE] ---")
    return product


def normalize_visibility(
    coherent_flux: NDArray,
    flux: NDArray,
    baseline_telescopes: Mapping[int, tuple[int, int]],
) -> tuple[NDArray, NDArray]:
    """Normalize coherent flux and flag non-positive or non-finite denominators."""
    visamp = np.full(coherent_flux.shape, np.nan, dtype=float)
    flag = np.ones(coherent_flux.shape, dtype=bool)

    for baseline, (telescope_1, telescope_2) in baseline_telescopes.items():
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


def compute_gdelay(
    visdata: NDArray,
    wavelength: NDArray,
    search_range: tuple[float, float] = (-30e-6, 30e-6),
    n_newton: int = 5,
) -> NDArray:
    """Estimate group delay by grid search followed by Newton refinement."""
    validate_gd_range(search_range)
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

    flat: object
    wave: object
    p2vm: P2VMCalibration
    calibration_dir: Path | None = None
    extractor: object = field(init=False, repr=False)

    def __post_init__(self):
        validate(self.p2vm)
        require_parent(self.p2vm, "flat", self.flat)
        require_parent(self.p2vm, "wave", self.wave)
        self.extractor = prepare_extraction(
            self.flat, self.wave, self.p2vm.wavelength_m
        )
        from copy import deepcopy
        from dataclasses import fields

        snapshots = {}
        for item in fields(self.p2vm):
            value = getattr(self.p2vm, item.name)
            if item.metadata:
                value = np.array(value, copy=True)
                value.flags.writeable = False
            else:
                value = deepcopy(value)
            snapshots[item.name] = value
        self.p2vm = P2VMCalibration(**snapshots)

    @property
    def n_baselines(self):
        return len(self.p2vm.baseline_telescopes)

    @property
    def wavelength_m(self):
        return self.extractor.wavelength_m

    def reduce_frame(
        self,
        raw: NDArray,
        *,
        dark,
        diagnostics=False,
        gd_search_min_m=-100e-6,
        gd_search_max_m=100e-6,
    ):
        """Reduce one RAW frame; caller must not subtract its selected dark beforehand."""
        cube = np.asarray(raw, dtype=np.float32)[None, :, :]
        spectra = self.extractor.extract(cube, dark)
        return reduce_spectra(
            spectra,
            self.p2vm,
            diagnostics=diagnostics,
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
        directory,
    )
