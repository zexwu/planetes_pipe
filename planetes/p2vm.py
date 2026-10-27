"""P2VM step: input sequence, spectral calibration, parameters and physical response."""

from collections.abc import Mapping
from dataclasses import replace

import numpy as np
from numpy.typing import NDArray

from . import log
from .extraction import extract_file, prepare_extraction
from .instrument import (
    BASELINES,
    IMAG_SLICE,
    N_BASELINES,
    N_DATA,
    N_OUTPUTS,
    N_TELESCOPES,
    REAL_SLICE,
    TELESCOPES,
)
from .io.raw import check_raw_calibrations, identify_raw
from .products import (
    CalibrationSpectra,
    DarkCalibration,
    P2VMCalibration,
    Provenance,
    fingerprint,
    positive,
    validate_gd_range,
    validate_wavelength,
)
from .reduction import compute_gdelay

PARAMETERS = {
    "phase_correction": (
        "planetes.calib.phase-correction",
        "phase-correction",
        "Apply supplied phase reference",
    ),
    "wavelength_min_m": (
        "planetes.calib.wavelength-min-m",
        "wavelength-min-m",
        "P2VM target minimum [m]",
    ),
    "wavelength_max_m": (
        "planetes.calib.wavelength-max-m",
        "wavelength-max-m",
        "P2VM target maximum [m]",
    ),
    "wavelength_channels": (
        "planetes.calib.wavelength-channels",
        "wavelength-channels",
        "Integer target grid channels",
    ),
    "gd_search_min_m": (
        "planetes.calib.gd-search-min-m",
        "gd-search-min-m",
        "Coarse GD lower initialization [m]; Newton remains unconstrained",
    ),
    "gd_search_max_m": (
        "planetes.calib.gd-search-max-m",
        "gd-search-max-m",
        "Coarse GD upper initialization [m]; Newton remains unconstrained",
    ),
}


def calibrate_p2vm_file(
    flat,
    wave,
    main_dark,
    *,
    telescope_flats,
    baseline_cubes,
    phase_reference=None,
    phase_dark=None,
    phase_correction=True,
    wavelength_min_m=1.15e-6,
    wavelength_max_m=1.30e-6,
    wavelength_channels=151,
    gd_search_min_m=-30e-6,
    gd_search_max_m=30e-6,
) -> P2VMCalibration:
    """Read the P2VM sequence, extract spectra and return the calibrated response."""

    if phase_dark is not None and phase_reference is None:
        raise ValueError("phase_dark requires phase_reference")
    if not isinstance(phase_correction, bool):
        raise ValueError("phase_correction must be bool")
    grid = wavelength_grid(wavelength_min_m, wavelength_max_m, wavelength_channels)
    spectra = extract_calibration_files(
        flat,
        wave,
        main_dark,
        telescope_flats=telescope_flats,
        baseline_cubes=baseline_cubes,
        wavelength_m=grid,
        phase_reference=phase_reference if phase_correction else None,
        phase_dark=phase_dark if phase_correction else None,
    )
    result = compute_p2vm(
        spectra,
        phase_correction=phase_correction,
        gd_search_min_m=gd_search_min_m,
        gd_search_max_m=gd_search_max_m,
    )
    params = {
        **result.provenance.parameters,
        "wavelength_min_m": wavelength_min_m,
        "wavelength_max_m": wavelength_max_m,
        "wavelength_channels": wavelength_channels,
        "phase_dark_reused": bool(
            phase_reference is not None and phase_correction and phase_dark is None
        ),
    }
    return replace(result, provenance=replace(result.provenance, parameters=params))


def compute_p2vm(
    spectra: CalibrationSpectra,
    *,
    phase_correction: bool = True,
    gd_search_min_m: float = -30e-6,
    gd_search_max_m: float = 30e-6,
) -> P2VMCalibration:
    """Compute the pixel-to-visibility matrix."""
    log.info("--- Step: P2VM ---")

    if not isinstance(phase_correction, bool):
        raise ValueError("phase_correction must be bool")
    search_range = (gd_search_min_m, gd_search_max_m)
    validate_gd_range(search_range)
    if (3 * len(spectra.wavelength_m)) // 4 - len(spectra.wavelength_m) // 4 < 2:
        raise ValueError("P2VM requires at least two central-band wavelength samples")
    spec_tel = spectra.telescope_spectra
    spec_bsl = spectra.baseline_spectra
    phase_reference = spectra.phase_reference
    bsl_to_reg = spectra.baseline_regions
    bsl_to_tel = spectra.baseline_telescopes
    wl_grid = spectra.wavelength_m

    reg_to_bsl = {reg: bsl for bsl, regs in bsl_to_reg.items() for reg in regs}

    n_frame = spec_bsl[0].shape[1]
    n_wave = spec_bsl[0].shape[2]

    transmission = np.zeros((N_OUTPUTS, N_TELESCOPES, n_wave))
    coherence_amplitude = np.zeros((N_OUTPUTS, N_BASELINES, n_wave))
    coherence_phase_rad = np.zeros((N_OUTPUTS, N_BASELINES, n_wave))
    cmat = np.zeros((N_OUTPUTS, N_BASELINES, n_wave))
    fit_rms = np.zeros_like(cmat)
    ellipse_phase = []
    ellipse_amplitude = []
    opd_per_baseline = np.zeros((N_BASELINES, n_frame), dtype=float)
    gd_per_baseline = np.zeros((N_BASELINES, n_frame), dtype=float)

    # --- A. Process Baselines (Interference Terms) ---
    for bsl in range(N_BASELINES):
        log.info(f"Processing baseline {bsl+1}/{N_BASELINES}...")

        spec = spec_bsl[bsl]
        regs = bsl_to_reg[bsl]
        # Keep explicit envelope weights for the planned weighted ellipse fit.
        envs = np.ones((n_frame, n_wave), dtype=float)

        # 1. Fit Ellipses to extract raw Phase & Visibility
        log.debug("    Fitting ellipses for phase extraction...")
        phase, visamp = fit_ellipse_linear(spec[regs], envs)
        visdata = visamp * np.exp(1j * phase)
        ellipse_phase.append(phase)
        ellipse_amplitude.append(visamp)

        # use the central 50% of the wavelength range for OPD and GD estimation
        w_start = n_wave // 4
        w_end = (n_wave * 3) // 4

        opd = np.mean((phase * wl_grid)[:, w_start:w_end] / 2 / np.pi, axis=1)
        gd = compute_gdelay(
            visdata[:, w_start:w_end], wl_grid[w_start:w_end], search_range=search_range
        )

        # Assume OPD=0 <-> GD=0; remove OPD zero-point
        _, opd0 = np.polyfit(gd, opd, 1)
        opd -= opd0
        opd_per_baseline[bsl] = opd
        gd_per_baseline[bsl] = gd
        log.debug(f"    Removing zero-point {opd0=:.2f} m...")

        # 2. Fit P2VM Coefficients
        # Reconstruct phase for fitting
        phi = 2 * np.pi * opd[:, None] / wl_grid[None, :]

        # Solve Linear System: spec = a*cos(phi) + b*sin(phi) + c
        mat = np.ones((len(phi), 3))
        log.debug("    Fitting P2VM coefficients...")
        for iw in range(n_wave):
            # Construct design matrix for this wavelength
            mat[:, 1] = np.sin(phi[:, iw])
            mat[:, 2] = np.cos(phi[:, iw])
            mat_inv = np.linalg.pinv(mat)

            # Fit coefficients
            coeff = np.einsum("if,of->oi", mat_inv, spec[..., iw])
            a, b, c = coeff[:, 2], coeff[:, 1], coeff[:, 0]

            coherence_amplitude[:, bsl, iw] = np.sqrt(a**2 + b**2)
            coherence_phase_rad[:, bsl, iw] = np.arctan2(b, a)
            cmat[:, bsl, iw] = c
            residual = spec[..., iw] - coeff @ mat.T
            fit_rms[:, bsl, iw] = np.sqrt(np.mean(residual**2, axis=1))

    # --- B. Process Telescopes (Photometric Terms) ---
    for tel in range(N_TELESCOPES):
        log.info(f"Processing telescope {tel + 1}/{N_TELESCOPES}...")
        # Median transmission map
        trans_map = np.median(spec_tel[tel], axis=1)
        # Normalize sum to 1
        transmission[:, tel, :] = trans_map / trans_map.sum()

    # --- C. Normalization & Inversion ---
    log.info("P2VM Normalization...")
    log.info("    Phase normalization (A=0)...")
    for bsl in range(N_BASELINES):
        regs = bsl_to_reg[bsl]
        phase_A = coherence_phase_rad[regs[0]]
        coherence_phase_rad[regs] -= phase_A[None, :, :]

    log.info("    Transmission normalization (mean over 24 outputs -> 1)...")
    transmission /= transmission.mean(axis=0)[None, :, :]

    log.info("    Coherence normalization...")
    for bsl in range(N_BASELINES):
        t1, t2 = bsl_to_tel[bsl]
        for iw in range(n_wave):
            t_mat = transmission[:, :, iw]
            c = cmat[:, bsl, iw]

            # Solve for intensities contributing to the DC term
            imat = np.linalg.pinv(t_mat) @ c
            flux_product = imat[t1] * imat[t2]
            if not np.isfinite(flux_product) or flux_product <= 0:
                raise ValueError(
                    f"Invalid photometric normalization for baseline {bsl}, "
                    f"wavelength index {iw}"
                )
            f1f2 = np.sqrt(flux_product)
            coherence_amplitude[:, bsl, iw] /= f1f2

    v2pm_matrix = build_v2pm_matrix(
        transmission,
        coherence_amplitude,
        coherence_phase_rad,
    )

    log.info("V2PM -> P2VM")
    p2vm = np.zeros((N_DATA, N_OUTPUTS, n_wave))
    for iw in range(n_wave):
        p2vm[:, :, iw] = np.linalg.pinv(v2pm_matrix[:, :, iw])

    has_phase_reference = phase_reference.shape[1] > 0
    if has_phase_reference and phase_correction:
        log.info("Applying phase-reference correction...")
        p2vmred_wave = np.einsum("dow,ofw->fdw", p2vm, phase_reference)
        vis_ft = p2vmred_wave[:, REAL_SLICE, :] + 1j * p2vmred_wave[:, IMAG_SLICE, :]
        coherence_phase_rad = _correct_phase_reference(
            coherence_phase_rad,
            vis_ft,
            wl_grid,
            bsl_to_tel,
            reg_to_bsl,
            search_range=search_range,
        )

        v2pm_matrix = build_v2pm_matrix(
            transmission,
            coherence_amplitude,
            coherence_phase_rad,
        )

        p2vm = np.zeros((N_DATA, N_OUTPUTS, n_wave))
        for iw in range(n_wave):
            p2vm[:, :, iw] = np.linalg.pinv(v2pm_matrix[:, :, iw])
    elif phase_correction:
        log.info("Skipping phase correction: no phase-reference cube was provided.")

    product = P2VMCalibration(
        p2vm=p2vm,
        wavelength_m=wl_grid,
        baseline_regions=bsl_to_reg,
        baseline_telescopes=bsl_to_tel,
        transmission=transmission,
        coherence_amplitude=coherence_amplitude,
        coherence_phase_rad=coherence_phase_rad,
        opd_m=opd_per_baseline,
        group_delay_m=gd_per_baseline,
        ellipse_phase_rad=np.asarray(ellipse_phase),
        ellipse_amplitude=np.asarray(ellipse_amplitude),
        dc_term_adu=cmat,
        fit_rms_adu=fit_rms,
        provenance=Provenance(
            producer="planetes.p2vm",
            parameters={
                "phase_correction": phase_correction,
                "phase_reference_applied": bool(
                    has_phase_reference and phase_correction
                ),
                "gd_search_min_m": gd_search_min_m,
                "gd_search_max_m": gd_search_max_m,
            },
            unverified=spectra.provenance.unverified,
            inputs=spectra.provenance.inputs,
            dependencies=spectra.provenance.dependencies,
        ),
    )
    log.info("--- Step: P2VM [DONE] ---")

    return product


def build_v2pm_matrix(
    transmission: NDArray,
    coherence_amplitude: NDArray,
    coherence_phase_rad: NDArray,
) -> NDArray:
    """Expand the V2PM calibration terms into the matrix used for inversion."""
    matrix = np.zeros((N_OUTPUTS, N_DATA, transmission.shape[-1]))
    matrix[:, :N_TELESCOPES] = transmission
    matrix[:, REAL_SLICE] = coherence_amplitude * np.cos(coherence_phase_rad)
    matrix[:, IMAG_SLICE] = coherence_amplitude * np.sin(coherence_phase_rad)
    matrix /= N_OUTPUTS
    return matrix


def _correct_phase_reference(
    phase: NDArray,
    visdata: NDArray,
    wavelength: NDArray,
    baseline_telescopes: Mapping[int, tuple[int, int]],
    region_to_baseline: Mapping[int, int],
    *,
    search_range=(-30e-6, 30e-6),
) -> NDArray:
    """Apply the phase-reference solution to the calibrated V2PM phases."""
    phase = phase.copy()
    opl = np.zeros((len(visdata), N_TELESCOPES))
    wave0 = len(wavelength) // 2

    for telescope in range(1, N_TELESCOPES):
        group_delay = compute_gdelay(
            visdata[:, telescope - 1, :], wavelength, search_range=search_range
        )
        phasor = np.exp(-1j * 2 * np.pi * group_delay[:, None] / wavelength)
        opl[:, telescope] = (
            np.angle(visdata[:, telescope - 1, wave0] * phasor[:, wave0])
            + 2 * np.pi * group_delay / wavelength[wave0]
        )
    opl -= opl.mean(axis=0)[None, :]

    visphase = np.zeros((N_BASELINES, len(wavelength)), dtype=complex)
    for baseline in range(N_BASELINES):
        telescope_1, telescope_2 = baseline_telescopes[baseline]
        opl_difference = opl[:, telescope_2] - opl[:, telescope_1]
        visphase[baseline] = (
            visdata[:, baseline]
            * np.exp(-1j * opl_difference[:, None] * wavelength[wave0] / wavelength)
        ).sum(axis=0)

    reference = np.ones((N_TELESCOPES, len(wavelength)), dtype=complex)
    group_delay = compute_gdelay(visphase, wavelength, search_range=search_range)
    for telescope in range(N_TELESCOPES - 1):
        reference[telescope + 1] = np.exp(
            2j * np.pi * group_delay[telescope] / wavelength
        )

    for baseline in range(N_BASELINES):
        telescope_1, telescope_2 = baseline_telescopes[baseline]
        visphase[baseline] *= reference[telescope_1] * reference[telescope_2].conj()

    for region in range(N_OUTPUTS):
        baseline = region_to_baseline[region]
        phasor = np.exp(1j * phase[region, baseline]) * visphase[baseline].conj()
        phase[region, baseline] = np.angle(phasor)
    return phase


# @njit(parallel=True, fastmath=True, cache=True)
def fit_ellipse_linear(spec: NDArray, envs: NDArray) -> tuple[NDArray, NDArray]:
    """
    Estimate phase and visibility amplitude from ABCD outputs via ellipse fitting.

    Args:
        spec (NDArray): ABCD flux with shape `(4, n_frame, n_wave)`.
        envs (NDArray): Envelope weights with shape `(n_frame, n_wave)`.

    Returns:
        tuple[NDArray, NDArray]:
            Phase in radians with shape `(n_frame, n_wave)`, and visibility
            amplitude with the same shape.
    """
    _, n_frames, n_wave = spec.shape

    phase = np.zeros((n_frames, n_wave))
    visamp = np.zeros((n_frames, n_wave))

    for iwl in range(n_wave):
        A, C, B, D = spec[:, :, iwl]

        X, Y = C - A, D - B

        # For numerical stability, center and scale the data
        X = X - np.mean(X)
        Y = Y - np.mean(Y)
        std_x, std_y = np.std(X), np.std(Y)
        if std_x > 1e-9:
            X /= std_x
        if std_y > 1e-9:
            Y /= std_y

        # Design matrix for: c0*X^2 + c1*XY + c2*X + c3*Y + c4*Y^2 = 1
        coeffs = np.empty((5, n_frames))
        coeffs[0, :] = X**2
        coeffs[1, :] = X * Y
        coeffs[2, :] = X
        coeffs[3, :] = Y
        coeffs[4, :] = Y**2

        # Linear solve via Pseudo-Inverse
        w = np.linalg.pinv(coeffs.T) @ np.ascontiguousarray(envs[:, iwl])

        b_sq = w[4]
        if b_sq <= 0:
            b_sq = 1e-6

        d = w[1] / (2 * b_sq)
        a_sq = w[0] - b_sq * (d**2)
        if a_sq <= 0:
            a_sq = 1e-6

        c2 = w[3] / (2 * b_sq)
        c1 = (w[2] - 2 * b_sq * d * c2) / (2 * a_sq)

        a, b = np.sqrt(a_sq), np.sqrt(b_sq)
        fac = np.sqrt(1 + (a * c1) ** 2 + (b * c2) ** 2)

        # Transform to unit circle
        X_fit = a * (X + c1) / fac
        Y_fit = b * (Y + d * X + c2) / fac

        phase[:, iwl] = np.unwrap(np.arctan2(Y_fit, X_fit))
        visamp[:, iwl] = np.sqrt(X_fit**2 + Y_fit**2)

    return phase, visamp


def extract_calibration_files(
    flat,
    wave,
    main_dark,
    *,
    telescope_flats,
    baseline_cubes,
    wavelength_m,
    phase_reference=None,
    phase_dark=None,
) -> CalibrationSpectra:
    """Extract calibration files and infer the retained physical output mapping."""

    if not isinstance(main_dark, DarkCalibration) or (
        phase_dark is not None and not isinstance(phase_dark, DarkCalibration)
    ):
        raise ValueError("Formal file tasks require complete DARK calibrations")
    telescope_flats = identify_raw(telescope_flats, "FLAT_RAW")
    baseline_cubes = identify_raw(baseline_cubes, "P2VM_RAW")
    if phase_dark is not None and phase_reference is None:
        raise ValueError("phase_dark requires phase_reference")
    model = prepare_extraction(flat, wave, wavelength_m)
    unknown = []
    for path in [*telescope_flats.values(), *baseline_cubes.values()]:
        unknown.extend(check_raw_calibrations(path, main_dark, flat))
    spec_bsl = np.asarray(
        [extract_file(baseline_cubes[name], model, main_dark) for name in BASELINES]
    )
    spec_tel = np.asarray(
        [extract_file(telescope_flats[name], model, main_dark) for name in TELESCOPES]
    )
    phase_reference_spectra = np.empty(
        (N_OUTPUTS, 0, len(model.wavelength_m)), dtype=np.float64
    )
    dependencies = {
        "flat": fingerprint(flat),
        "wave": fingerprint(wave),
        "main_dark": fingerprint(main_dark),
    }
    inputs = {
        **{f"tel_flat_{k}": str(v) for k, v in telescope_flats.items()},
        **{f"baseline_{k}": str(v) for k, v in baseline_cubes.items()},
    }
    if phase_reference is not None:
        selected_dark = main_dark if phase_dark is None else phase_dark
        unknown.extend(
            check_raw_calibrations(
                phase_reference, selected_dark, flat, category="SINGLE_SCI_RAW"
            )
        )
        phase_reference_spectra = extract_file(phase_reference, model, selected_dark)
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
        model.wavelength_m,
        provenance=Provenance(
            producer="planetes.extraction",
            inputs=inputs,
            dependencies=dependencies,
            unverified=tuple(sorted(set(unknown))),
        ),
    )


def wavelength_grid(low=1.15e-6, high=1.30e-6, channels=151) -> NDArray:
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
