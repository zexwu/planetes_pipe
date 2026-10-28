"""P2VM step: input sequence, spectral calibration, parameters and physical response."""

from collections.abc import Iterable, Mapping
from dataclasses import replace
from pathlib import Path

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
    FlatCalibration,
    P2VMCalibration,
    Provenance,
    WavelengthCalibration,
    fingerprint,
    positive,
    validate_gd_range,
    validate_wavelength,
)
from .reduction import compute_gdelay


def calibrate_p2vm_file(
    flat: FlatCalibration,
    wave: WavelengthCalibration,
    main_dark: DarkCalibration,
    *,
    telescope_flats: Mapping[str, Path] | Iterable[Path],
    baseline_cubes: Mapping[str, Path] | Iterable[Path],
    phase_reference: Path | None = None,
    phase_dark: DarkCalibration | None = None,
    phase_correction: bool = True,
    wavelength_min_m: float = 1.15e-6,
    wavelength_max_m: float = 1.30e-6,
    wavelength_channels: int = 151,
    gd_search_min_m: float = -30e-6,
    gd_search_max_m: float = 30e-6,
) -> P2VMCalibration:
    """Read the calibration sequence and return the full 24-output response/inverse.

    Wavelength and coarse GD bounds use metres. The optional phase reference
    defines the instrumental phase and uses its selected DARK, or ``main_dark``.
    """

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
    """Fit baseline responses, normalize the instrument and invert its V2PM."""
    log.info("--- Step: P2VM ---")

    if not isinstance(phase_correction, bool):
        raise ValueError("phase_correction must be bool")
    search_range = (gd_search_min_m, gd_search_max_m)
    validate_gd_range(search_range)
    wavelength_m = spectra.wavelength_m
    if (3 * len(wavelength_m)) // 4 - len(wavelength_m) // 4 < 2:
        raise ValueError("P2VM requires at least two central-band wavelength samples")

    n_frames, n_wavelengths = spectra.baseline_spectra[0].shape[1:]
    coherence_amplitude = np.zeros((N_OUTPUTS, N_BASELINES, n_wavelengths))
    coherence_phase_rad = np.zeros((N_OUTPUTS, N_BASELINES, n_wavelengths))
    dc_term_adu = np.zeros((N_OUTPUTS, N_BASELINES, n_wavelengths))
    fit_rms_adu = np.zeros_like(dc_term_adu)
    ellipse_phases = []
    ellipse_radii = []
    opd_m = np.zeros((N_BASELINES, n_frames), dtype=float)
    group_delay_m = np.zeros((N_BASELINES, n_frames), dtype=float)

    for baseline in range(N_BASELINES):
        log.info(f"Processing baseline {baseline + 1}/{N_BASELINES}...")
        baseline_spectra = spectra.baseline_spectra[baseline]
        outputs = spectra.baseline_regions[baseline]
        # Keep explicit envelope weights for the planned weighted ellipse fit.
        envelope_weights = np.ones((n_frames, n_wavelengths), dtype=float)
        log.debug("    Fitting ellipses for phase extraction...")
        ellipse_phase_rad, ellipse_radius = fit_ellipse_linear(
            baseline_spectra[outputs], envelope_weights
        )
        ellipse_phases.append(ellipse_phase_rad)
        ellipse_radii.append(ellipse_radius)
        opd_m[baseline], group_delay_m[baseline] = _estimate_scan_opd(
            ellipse_phase_rad, ellipse_radius, wavelength_m, search_range
        )
        (
            coherence_amplitude[:, baseline, :],
            coherence_phase_rad[:, baseline, :],
            dc_term_adu[:, baseline, :],
            fit_rms_adu[:, baseline, :],
        ) = _fit_interferometric_response(
            baseline_spectra, opd_m[baseline], wavelength_m
        )

    transmission = _fit_photometric_transmission(spectra.telescope_spectra)
    _normalize_response(
        transmission,
        coherence_amplitude,
        coherence_phase_rad,
        dc_term_adu,
        spectra.baseline_regions,
        spectra.baseline_telescopes,
    )
    v2pm_matrix = build_v2pm_matrix(
        transmission, coherence_amplitude, coherence_phase_rad
    )
    p2vm = _invert_v2pm(v2pm_matrix)

    has_phase_reference = spectra.phase_reference.shape[1] > 0
    if has_phase_reference and phase_correction:
        log.info("Applying phase-reference correction...")
        reference_channels = np.einsum("dow,ofw->fdw", p2vm, spectra.phase_reference)
        reference_coherent_flux_adu = (
            reference_channels[:, REAL_SLICE, :]
            + 1j * reference_channels[:, IMAG_SLICE, :]
        )
        baseline_by_output = {
            output: baseline
            for baseline, outputs in spectra.baseline_regions.items()
            for output in outputs
        }
        coherence_phase_rad = _correct_phase_reference(
            coherence_phase_rad,
            reference_coherent_flux_adu,
            wavelength_m,
            spectra.baseline_telescopes,
            baseline_by_output,
            search_range=search_range,
        )
        v2pm_matrix = build_v2pm_matrix(
            transmission, coherence_amplitude, coherence_phase_rad
        )
        p2vm = _invert_v2pm(v2pm_matrix)
    elif phase_correction:
        log.info("Skipping phase correction: no phase-reference cube was provided.")

    product = P2VMCalibration(
        p2vm=p2vm,
        wavelength_m=wavelength_m,
        baseline_regions=spectra.baseline_regions,
        baseline_telescopes=spectra.baseline_telescopes,
        transmission=transmission,
        coherence_amplitude=coherence_amplitude,
        coherence_phase_rad=coherence_phase_rad,
        opd_m=opd_m,
        group_delay_m=group_delay_m,
        ellipse_phase_rad=np.asarray(ellipse_phases),
        ellipse_amplitude=np.asarray(ellipse_radii),
        dc_term_adu=dc_term_adu,
        fit_rms_adu=fit_rms_adu,
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


def _estimate_scan_opd(
    ellipse_phase_rad: NDArray,
    ellipse_radius: NDArray,
    wavelength_m: NDArray,
    search_range: tuple[float, float],
) -> tuple[NDArray, NDArray]:
    """Use the central half-band and set the fitted OPD zero point at GD=0."""
    ellipse_phasor = ellipse_radius * np.exp(1j * ellipse_phase_rad)
    n_wavelengths = len(wavelength_m)
    start, end = n_wavelengths // 4, (n_wavelengths * 3) // 4
    opd_m = np.mean(
        (ellipse_phase_rad * wavelength_m)[:, start:end] / 2 / np.pi, axis=1
    )
    group_delay_m = compute_gdelay(
        ellipse_phasor[:, start:end], wavelength_m[start:end], search_range=search_range
    )
    _, opd_zero_m = np.polyfit(group_delay_m, opd_m, 1)
    opd_m -= opd_zero_m
    log.debug(f"    Removing OPD zero point {opd_zero_m:.9g} m...")
    return opd_m, group_delay_m


def _fit_interferometric_response(
    baseline_spectra: NDArray, opd_m: NDArray, wavelength_m: NDArray
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Fit c+a*cos(phi)+b*sin(phi) to all outputs, retaining amplitude/phase/DC/RMS.

    Amplitude, DC and RMS are raw ADU at this stage; phase is in radians.
    The response amplitude is normalized later using the photometric transmission.
    """
    n_wavelengths = len(wavelength_m)
    coherence_amplitude = np.zeros((N_OUTPUTS, n_wavelengths))
    coherence_phase_rad = np.zeros((N_OUTPUTS, n_wavelengths))
    dc_term_adu = np.zeros((N_OUTPUTS, n_wavelengths))
    fit_rms_adu = np.zeros_like(dc_term_adu)
    phase_rad = 2 * np.pi * opd_m[:, None] / wavelength_m[None, :]
    design_matrix = np.ones((len(phase_rad), 3))
    log.debug("    Fitting P2VM coefficients...")
    for wavelength_index in range(n_wavelengths):
        design_matrix[:, 1] = np.sin(phase_rad[:, wavelength_index])
        design_matrix[:, 2] = np.cos(phase_rad[:, wavelength_index])
        inverse = np.linalg.pinv(design_matrix)
        coefficients = np.einsum(
            "if,of->oi", inverse, baseline_spectra[..., wavelength_index]
        )
        a, b, c = coefficients[:, 2], coefficients[:, 1], coefficients[:, 0]
        coherence_amplitude[:, wavelength_index] = np.sqrt(a**2 + b**2)
        coherence_phase_rad[:, wavelength_index] = np.arctan2(b, a)
        dc_term_adu[:, wavelength_index] = c
        residual = (
            baseline_spectra[..., wavelength_index] - coefficients @ design_matrix.T
        )
        fit_rms_adu[:, wavelength_index] = np.sqrt(np.mean(residual**2, axis=1))
    return coherence_amplitude, coherence_phase_rad, dc_term_adu, fit_rms_adu


def _fit_photometric_transmission(telescope_spectra: NDArray) -> NDArray:
    """Normalize each telescope's median spectrum by its total over outputs/wavelengths."""
    transmission = np.zeros((N_OUTPUTS, N_TELESCOPES, telescope_spectra.shape[-1]))
    for telescope in range(N_TELESCOPES):
        log.info(f"Processing telescope {telescope + 1}/{N_TELESCOPES}...")
        median_spectrum = np.median(telescope_spectra[telescope], axis=1)
        transmission[:, telescope, :] = median_spectrum / median_spectrum.sum()
    return transmission


def _normalize_response(
    transmission: NDArray,
    coherence_amplitude: NDArray,
    coherence_phase_rad: NDArray,
    dc_term_adu: NDArray,
    baseline_regions: Mapping[int, list[int]],
    baseline_telescopes: Mapping[int, tuple[int, int]],
) -> None:
    """Normalize phase, transmission and ADU coherence to dimensionless response in-place."""
    log.info("P2VM Normalization...")
    log.info("    Phase normalization (A=0)...")
    for baseline in range(N_BASELINES):
        outputs = baseline_regions[baseline]
        reference_phase_rad = coherence_phase_rad[outputs[0]]
        coherence_phase_rad[outputs] -= reference_phase_rad[None, :, :]

    log.info("    Transmission normalization (mean over 24 outputs -> 1)...")
    transmission /= transmission.mean(axis=0)[None, :, :]

    log.info("    Coherence normalization...")
    for baseline in range(N_BASELINES):
        telescope_1, telescope_2 = baseline_telescopes[baseline]
        for wavelength_index in range(transmission.shape[-1]):
            photometry = (
                np.linalg.pinv(transmission[:, :, wavelength_index])
                @ dc_term_adu[:, baseline, wavelength_index]
            )
            flux_product = photometry[telescope_1] * photometry[telescope_2]
            if not np.isfinite(flux_product) or flux_product <= 0:
                raise ValueError(
                    f"Invalid photometric normalization for baseline {baseline}, "
                    f"wavelength index {wavelength_index}"
                )
            coherence_amplitude[:, baseline, wavelength_index] /= np.sqrt(flux_product)


def _invert_v2pm(v2pm_matrix: NDArray) -> NDArray:
    """Compute the same per-wavelength pseudoinverse before and after phase correction."""
    log.info("V2PM -> P2VM")
    n_wavelengths = v2pm_matrix.shape[-1]
    p2vm = np.zeros((N_DATA, N_OUTPUTS, n_wavelengths))
    for wavelength_index in range(n_wavelengths):
        p2vm[:, :, wavelength_index] = np.linalg.pinv(
            v2pm_matrix[:, :, wavelength_index]
        )
    return p2vm


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
    telescope_phase_rad = np.zeros((len(visdata), N_TELESCOPES))
    reference_channel = len(wavelength) // 2

    for telescope in range(1, N_TELESCOPES):
        group_delay = compute_gdelay(
            visdata[:, telescope - 1, :], wavelength, search_range=search_range
        )
        phasor = np.exp(-1j * 2 * np.pi * group_delay[:, None] / wavelength)
        telescope_phase_rad[:, telescope] = (
            np.angle(
                visdata[:, telescope - 1, reference_channel]
                * phasor[:, reference_channel]
            )
            + 2 * np.pi * group_delay / wavelength[reference_channel]
        )
    telescope_phase_rad -= telescope_phase_rad.mean(axis=0)[None, :]

    integrated_coherent_flux = np.zeros((N_BASELINES, len(wavelength)), dtype=complex)
    for baseline in range(N_BASELINES):
        telescope_1, telescope_2 = baseline_telescopes[baseline]
        baseline_phase_rad = (
            telescope_phase_rad[:, telescope_2] - telescope_phase_rad[:, telescope_1]
        )
        integrated_coherent_flux[baseline] = (
            visdata[:, baseline]
            * np.exp(
                -1j
                * baseline_phase_rad[:, None]
                * wavelength[reference_channel]
                / wavelength
            )
        ).sum(axis=0)

    telescope_phasor = np.ones((N_TELESCOPES, len(wavelength)), dtype=complex)
    group_delay = compute_gdelay(
        integrated_coherent_flux, wavelength, search_range=search_range
    )
    for telescope in range(N_TELESCOPES - 1):
        telescope_phasor[telescope + 1] = np.exp(
            2j * np.pi * group_delay[telescope] / wavelength
        )

    for baseline in range(N_BASELINES):
        telescope_1, telescope_2 = baseline_telescopes[baseline]
        integrated_coherent_flux[baseline] *= (
            telescope_phasor[telescope_1] * telescope_phasor[telescope_2].conj()
        )

    for output in range(N_OUTPUTS):
        baseline = region_to_baseline[output]
        phasor = (
            np.exp(1j * phase[output, baseline])
            * integrated_coherent_flux[baseline].conj()
        )
        phase[output, baseline] = np.angle(phasor)
    return phase


def fit_ellipse_linear(spec: NDArray, envs: NDArray) -> tuple[NDArray, NDArray]:
    """
    Estimate unwrapped phase and normalized ellipse radius from A,C,B,D outputs.

    Args:
        spec (NDArray): ABCD flux with shape `(4, n_frame, n_wave)`.
        envs (NDArray): Envelope weights with shape `(n_frame, n_wave)`.

    Returns:
        tuple[NDArray, NDArray]:
            Phase in radians with shape `(n_frame, n_wave)`, and dimensionless
            ellipse radius with the same shape.
    """
    _, n_frames, n_wave = spec.shape

    ellipse_phase_rad = np.zeros((n_frames, n_wave))
    ellipse_radius = np.zeros((n_frames, n_wave))

    for wavelength_index in range(n_wave):
        A, C, B, D = spec[:, :, wavelength_index]

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
        w = np.linalg.pinv(coeffs.T) @ np.ascontiguousarray(envs[:, wavelength_index])

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

        ellipse_phase_rad[:, wavelength_index] = np.unwrap(np.arctan2(Y_fit, X_fit))
        ellipse_radius[:, wavelength_index] = np.sqrt(X_fit**2 + Y_fit**2)

    return ellipse_phase_rad, ellipse_radius


def extract_calibration_files(
    flat: FlatCalibration,
    wave: WavelengthCalibration,
    main_dark: DarkCalibration,
    *,
    telescope_flats: Mapping[str, Path] | Iterable[Path],
    baseline_cubes: Mapping[str, Path] | Iterable[Path],
    wavelength_m: NDArray,
    phase_reference: Path | None = None,
    phase_dark: DarkCalibration | None = None,
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
    extractor = prepare_extraction(flat, wave, wavelength_m)
    unknown = []
    for category, paths in (
        ("FLAT_RAW", telescope_flats),
        ("P2VM_RAW", baseline_cubes),
    ):
        for path in paths.values():
            unknown.extend(
                check_raw_calibrations(path, main_dark, flat, category=category)
            )
    baseline_spectra = np.asarray(
        [extract_file(baseline_cubes[name], extractor, main_dark) for name in BASELINES]
    )
    telescope_spectra = np.asarray(
        [
            extract_file(telescope_flats[name], extractor, main_dark)
            for name in TELESCOPES
        ]
    )
    phase_reference_spectra = np.empty(
        (N_OUTPUTS, 0, len(extractor.wavelength_m)), dtype=np.float64
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
        phase_reference_spectra = extract_file(
            phase_reference, extractor, selected_dark
        )
        dependencies["phase_dark"] = fingerprint(selected_dark)
        inputs["phase_reference"] = str(phase_reference)
    outputs_by_telescope = {}
    for telescope, flux in enumerate(telescope_spectra):
        mean_flux = flux.mean(axis=(1, 2))
        active = np.argsort(mean_flux)[-N_OUTPUTS // 2 :]
        outputs_by_telescope[telescope] = np.sort(active).tolist()

    outputs_by_modulation = {}
    for baseline, flux in enumerate(baseline_spectra):
        std_flux = flux.std(axis=1).mean(axis=-1)
        active = np.argsort(std_flux)[-4:]
        outputs_by_modulation[baseline] = np.sort(active).tolist()

    baseline_regions = {}
    baseline_telescopes = {}
    for telescope_1 in range(N_TELESCOPES):
        for telescope_2 in range(N_TELESCOPES):
            if telescope_1 >= telescope_2:
                continue
            baseline_name = TELESCOPES[telescope_1] + TELESCOPES[telescope_2]
            baseline = BASELINES.index(baseline_name)
            from_telescopes = set(outputs_by_telescope[telescope_1]) & set(
                outputs_by_telescope[telescope_2]
            )
            from_modulation = set(outputs_by_modulation[baseline])
            if from_telescopes != from_modulation:
                raise ValueError(
                    f"Region mapping mismatch for baseline {baseline_name}: "
                    f"{sorted(from_telescopes)} from flats, "
                    f"{sorted(from_modulation)} from modulation"
                )
            baseline_regions[baseline] = sorted(outputs_by_modulation[baseline])
            baseline_telescopes[baseline] = (telescope_1, telescope_2)

    return CalibrationSpectra(
        telescope_spectra,
        baseline_spectra,
        phase_reference_spectra,
        baseline_regions,
        baseline_telescopes,
        extractor.wavelength_m,
        provenance=Provenance(
            producer="planetes.extraction",
            inputs=inputs,
            dependencies=dependencies,
            unverified=tuple(sorted(set(unknown))),
        ),
    )


def wavelength_grid(
    low: float = 1.15e-6, high: float = 1.30e-6, channels: int = 151
) -> NDArray:
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
