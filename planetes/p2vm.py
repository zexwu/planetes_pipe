"""P2VM calibration: scan response, photometric normalization and phase reference."""

from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from . import log
from .extraction import extract_spectra_from_file, prepare_extraction
from .instrument import (
    BASELINE_PAIRS,
    BASELINES,
    IMAG_SLICE,
    N_BASELINES,
    N_DATA,
    N_OUTPUTS,
    N_TELESCOPES,
    REAL_SLICE,
    TELESCOPES,
    _check_baselines,
)
from .io.raw import check_raw_calibrations, identify_raw_exposures, read_raw_metadata
from .products import (
    P2VMSpectra,
    Dark,
    Flat,
    P2VM,
    Wave,
)
from .reduction import estimate_group_delay


def fit_p2vm(
    spectra: P2VMSpectra,
    *,
    phase_correction: bool = True,
    keep_diagnostics: bool = False,
    gd_search_min_m: float = -30e-6,
    gd_search_max_m: float = 30e-6,
) -> P2VM:
    """Fit the full detector response and return its per-wavelength inverse.

    Scans: (baseline, output, frame, wavelength); flats: (telescope, output,
    frame, wavelength). Wavelength and GD bounds are metres. All 24 outputs
    enter each baseline fit, including cross-talk. Optional scan diagnostics
    change only the retained arrays, not the response or inverse.
    """
    log.info("--- Step: P2VM ---")
    _check_input_axes(spectra)
    wavelength_m = spectra.wavelength_m
    search_range = (gd_search_min_m, gd_search_max_m)

    # Scans: ABCD ellipse -> OPD/GD -> sinusoidal response of all 24 outputs.
    amplitude, phase_rad, dc_adu, diagnostics = _fit_scan_responses(
        spectra, search_range, keep_diagnostics=keep_diagnostics
    )

    # Telescope flats supply the photometric scale for the fitted response.
    transmission = _estimate_transmission(spectra.telescope_spectra)
    _normalize_response_inplace(
        transmission, amplitude, phase_rad, dc_adu, spectra.baseline_outputs
    )
    p2vm = _invert_v2pm(build_v2pm(transmission, amplitude, phase_rad))

    # The instrumental reference changes phases, then requires a new inverse.
    if phase_correction and spectra.phase_reference.shape[1] > 0:
        phase_rad = _apply_phase_reference(
            phase_rad,
            p2vm,
            spectra.phase_reference,
            wavelength_m,
            spectra.baseline_outputs,
            search_range=search_range,
        )
        p2vm = _invert_v2pm(build_v2pm(transmission, amplitude, phase_rad))
    elif phase_correction:
        log.info("Skipping phase correction: no phase-reference cube was provided.")

    product = P2VM(
        matrix=p2vm,
        wavelength_m=wavelength_m,
        baseline_outputs=spectra.baseline_outputs,
        transmission=transmission,
        coherence_amplitude=amplitude,
        coherence_phase_rad=phase_rad,
        **diagnostics,
    )
    log.info("--- Step: P2VM [DONE] ---")
    return product


def _check_input_axes(spectra: P2VMSpectra) -> None:
    """Check the array-fit boundary and the physical output assignment."""
    wavelength_m = spectra.wavelength_m
    _check_baselines(spectra.baseline_outputs)
    if (
        wavelength_m.ndim != 1
        or spectra.telescope_spectra.ndim != 4
        or spectra.telescope_spectra.shape[:2] != (N_TELESCOPES, N_OUTPUTS)
        or spectra.telescope_spectra.shape[-1] != len(wavelength_m)
        or spectra.baseline_spectra.ndim != 4
        or spectra.baseline_spectra.shape[:2] != (N_BASELINES, N_OUTPUTS)
        or spectra.baseline_spectra.shape[-1] != len(wavelength_m)
        or spectra.phase_reference.ndim != 3
        or spectra.phase_reference.shape[0] != N_OUTPUTS
        or spectra.phase_reference.shape[-1] != len(wavelength_m)
    ):
        raise ValueError(
            "P2VM spectra must match telescope/output/baseline/wavelength axes"
        )
    if (3 * len(wavelength_m)) // 4 - len(wavelength_m) // 4 < 2:
        raise ValueError("P2VM requires at least two central-band wavelength samples")


def _fit_scan_responses(
    spectra: P2VMSpectra,
    search_range: tuple[float, float],
    *,
    keep_diagnostics: bool,
) -> tuple[NDArray, NDArray, NDArray, dict[str, NDArray]]:
    """Fit each baseline scan: ellipse -> OPD zero point -> detector response.

    Return amplitude [ADU], phase [rad], DC [ADU], each (output, baseline,
    wavelength), and requested diagnostics. Scan OPD/GD are metres; ellipse
    radius is dimensionless. All output rows, including cross-talk, are fitted.
    """
    wavelength_m = spectra.wavelength_m
    n_frames, n_wavelengths = spectra.baseline_spectra[0].shape[1:]
    coherence_amplitude = np.zeros((N_OUTPUTS, N_BASELINES, n_wavelengths))
    coherence_phase_rad = np.zeros((N_OUTPUTS, N_BASELINES, n_wavelengths))
    dc_term_adu = np.zeros((N_OUTPUTS, N_BASELINES, n_wavelengths))
    fit_rms_adu = np.zeros_like(dc_term_adu)
    ellipse_phases = []
    ellipse_radii = []
    opd_m = np.zeros((N_BASELINES, n_frames))
    group_delay_m = np.zeros((N_BASELINES, n_frames))

    for baseline in range(N_BASELINES):
        log.info(f"Fitting baseline {BASELINES[baseline]}...")
        baseline_spectra = spectra.baseline_spectra[baseline]
        outputs = spectra.baseline_outputs[baseline]
        log.debug("    Fitting ellipses for phase extraction...")
        ellipse_phase_rad, ellipse_radius = fit_abcd_ellipse(baseline_spectra[outputs])
        if keep_diagnostics:
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

    diagnostics = {}
    if keep_diagnostics:
        diagnostics = dict(
            opd_m=opd_m,
            group_delay_m=group_delay_m,
            ellipse_phase_rad=np.asarray(ellipse_phases),
            ellipse_amplitude=np.asarray(ellipse_radii),
            dc_term_adu=dc_term_adu,
            fit_rms_adu=fit_rms_adu,
        )
    return coherence_amplitude, coherence_phase_rad, dc_term_adu, diagnostics


def _estimate_scan_opd(
    ellipse_phase_rad: NDArray,
    ellipse_radius: NDArray,
    wavelength_m: NDArray,
    search_range: tuple[float, float],
) -> tuple[NDArray, NDArray]:
    """Convert ellipse phase/radius (frame, wavelength) to scan OPD/GD (frame,).

    Both paths are metres. Use the central half-band, convert phase to
    phase*lambda/(2*pi), and remove the intercept of its linear fit against GD.
    """
    ellipse_phasor = ellipse_radius * np.exp(1j * ellipse_phase_rad)
    n_wavelengths = len(wavelength_m)
    start, end = n_wavelengths // 4, (n_wavelengths * 3) // 4
    opd_m = np.mean(
        (ellipse_phase_rad * wavelength_m)[:, start:end] / 2 / np.pi, axis=1
    )
    group_delay_m = estimate_group_delay(
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
    With phase=atan2(b,a), the response is c+amplitude*cos(phi-phase).
    Arrays returned here are (output, wavelength); OPD is (frame,) in metres.
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


def _estimate_transmission(telescope_spectra: NDArray) -> NDArray:
    """Normalize each telescope's median spectrum by its total over outputs/wavelengths."""
    transmission = np.zeros((N_OUTPUTS, N_TELESCOPES, telescope_spectra.shape[-1]))
    for telescope in range(N_TELESCOPES):
        log.info(f"Measuring transmission for telescope {TELESCOPES[telescope]}...")
        median_spectrum = np.median(telescope_spectra[telescope], axis=1)
        transmission[:, telescope, :] = median_spectrum / median_spectrum.sum()
    return transmission


def _normalize_response_inplace(
    transmission: NDArray,
    coherence_amplitude: NDArray,
    coherence_phase_rad: NDArray,
    dc_term_adu: NDArray,
    baseline_outputs: Mapping[int, list[int]],
) -> None:
    """Normalize phase, transmission and coherence in place.

    Within each physical output quartet, subtract its A row from *all six*
    response components, retaining that convention for cross-talk as well.
    Set each telescope/wavelength's mean transmission over outputs to one,
    then divide coherence [ADU] by sqrt(P_i*P_j) from the fitted DC terms.
    The resulting amplitude and transmission are dimensionless.
    """
    log.info("P2VM Normalization...")
    log.info("    Phase normalization (A=0)...")
    for baseline in range(N_BASELINES):
        outputs = baseline_outputs[baseline]
        reference_phase_rad = coherence_phase_rad[outputs[0]]
        coherence_phase_rad[outputs] -= reference_phase_rad[None, :, :]

    log.info("    Transmission normalization (mean over 24 outputs -> 1)...")
    transmission /= transmission.mean(axis=0)[None, :, :]

    log.info("    Coherence normalization...")
    for baseline in range(N_BASELINES):
        telescope_1, telescope_2 = BASELINE_PAIRS[baseline]
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
        rank = np.linalg.matrix_rank(v2pm_matrix[:, :, wavelength_index])
        if rank != N_DATA:
            raise ValueError(
                f"V2PM at wavelength index {wavelength_index} has rank {rank}; expected {N_DATA}"
            )
        p2vm[:, :, wavelength_index] = np.linalg.pinv(
            v2pm_matrix[:, :, wavelength_index]
        )
    return p2vm


def build_v2pm(
    transmission: NDArray,
    coherence_amplitude: NDArray,
    coherence_phase_rad: NDArray,
) -> NDArray:
    """Expand dimensionless terms into V2PM (output, data channel, wavelength).

    Columns are T4,T3,T2,T1, then Re(C_43,...,C_21), then Im(C_43,...,C_21).
    Coherent columns are amplitude*cos(phase) and amplitude*sin(phase).
    The complete matrix retains the original division by 24 outputs.
    """
    matrix = np.zeros((N_OUTPUTS, N_DATA, transmission.shape[-1]))
    matrix[:, :N_TELESCOPES] = transmission
    matrix[:, REAL_SLICE] = coherence_amplitude * np.cos(coherence_phase_rad)
    matrix[:, IMAG_SLICE] = coherence_amplitude * np.sin(coherence_phase_rad)
    matrix /= N_OUTPUTS
    return matrix


def _apply_phase_reference(
    phase_rad: NDArray,
    p2vm: NDArray,
    reference_spectra: NDArray,
    wavelength_m: NDArray,
    baseline_outputs: Mapping[int, list[int]],
    *,
    search_range: tuple[float, float],
) -> NDArray:
    """Set instrumental phases using a reference, with T4 as the telescope anchor.

    Reference spectra [ADU]: (output, frame, wavelength). Reconstructed complex
    coherence [ADU]: (frame, baseline, wavelength). Input/output response phases
    [rad]: (output, baseline, wavelength). Only each output's own baseline is
    corrected; off-baseline response phases retain their scan normalization.
    """
    log.info("Applying phase-reference correction...")
    phase_rad = phase_rad.copy()
    channels = np.einsum("dow,ofw->fdw", p2vm, reference_spectra)
    coherent_flux_adu = channels[:, REAL_SLICE, :] + 1j * channels[:, IMAG_SLICE, :]
    reference_channel = len(wavelength_m) // 2
    reference_telescope = TELESCOPES.index("4")
    # 43, 42, 41 identify the other telescopes relative to T4 explicitly.
    reference_baselines = [
        (baseline, second)
        for baseline, (first, second) in enumerate(BASELINE_PAIRS)
        if first == reference_telescope
    ]

    # Reconstruct each frame's telescope phase at the central wavelength.
    telescope_phase_rad = np.zeros((len(coherent_flux_adu), N_TELESCOPES))
    for baseline, telescope in reference_baselines:
        group_delay_m = estimate_group_delay(
            coherent_flux_adu[:, baseline, :], wavelength_m, search_range=search_range
        )
        phasor = np.exp(-1j * 2 * np.pi * group_delay_m[:, None] / wavelength_m)
        telescope_phase_rad[:, telescope] = (
            np.angle(
                coherent_flux_adu[:, baseline, reference_channel]
                * phasor[:, reference_channel]
            )
            + 2 * np.pi * group_delay_m / wavelength_m[reference_channel]
        )
    telescope_phase_rad -= telescope_phase_rad.mean(axis=0)[None, :]

    # Remove temporal phase excursions (scaled as 1/lambda), then sum frames.
    integrated_coherent_flux = np.zeros((N_BASELINES, len(wavelength_m)), dtype=complex)
    for baseline, (telescope_1, telescope_2) in enumerate(BASELINE_PAIRS):
        baseline_phase_rad = (
            telescope_phase_rad[:, telescope_2] - telescope_phase_rad[:, telescope_1]
        )
        integrated_coherent_flux[baseline] = (
            coherent_flux_adu[:, baseline]
            * np.exp(
                -1j
                * baseline_phase_rad[:, None]
                * wavelength_m[reference_channel]
                / wavelength_m
            )
        ).sum(axis=0)

    # Remove the residual spectral GD with telescope phasors anchored to T4.
    telescope_phasor = np.ones((N_TELESCOPES, len(wavelength_m)), dtype=complex)
    group_delay_m = estimate_group_delay(
        integrated_coherent_flux, wavelength_m, search_range=search_range
    )
    for baseline, telescope in reference_baselines:
        telescope_phasor[telescope] = np.exp(
            2j * np.pi * group_delay_m[baseline] / wavelength_m
        )
    for baseline, (telescope_1, telescope_2) in enumerate(BASELINE_PAIRS):
        integrated_coherent_flux[baseline] *= (
            telescope_phasor[telescope_1] * telescope_phasor[telescope_2].conj()
        )

    # Apply the reference phase only to the matching physical output quartet.
    baseline_by_output = {
        output: baseline
        for baseline, outputs in baseline_outputs.items()
        for output in outputs
    }
    for output in range(N_OUTPUTS):
        baseline = baseline_by_output[output]
        phasor = (
            np.exp(1j * phase_rad[output, baseline])
            * integrated_coherent_flux[baseline].conj()
        )
        phase_rad[output, baseline] = np.angle(phasor)
    return phase_rad


def fit_abcd_ellipse(spec: NDArray) -> tuple[NDArray, NDArray]:
    """Fit X=C-A, Y=D-B to an ellipse with equation M @ coefficients = 1.

    Input ADU axes are (output, frame, wavelength), ordered A,C,B,D along
    the four-output axis. Return unwrapped phase [rad] and dimensionless radius, both
    (frame, wavelength). Unwrapping runs along frames. For ideal outputs
    I_q = c + A*cos(phi + [0, pi, pi/2, 3*pi/2]_q), this phase decreases
    with phi; preserve that sign through scan OPD and response fitting.
    """
    _, n_frames, n_wave = spec.shape

    ellipse_phase_rad = np.zeros((n_frames, n_wave))
    ellipse_radius = np.zeros((n_frames, n_wave))
    unit_rhs = np.ones(n_frames)

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

        w = np.linalg.pinv(coeffs.T) @ unit_rhs

        # Retained numerical fallbacks; these do not certify a valid ellipse.
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


def make_p2vm(
    flat: Flat,
    wave: Wave,
    main_dark: Dark,
    *,
    telescope_flats: Mapping[str, Path] | Iterable[Path],
    baseline_cubes: Mapping[str, Path] | Iterable[Path],
    phase_reference: Path | None = None,
    phase_dark: Dark | None = None,
    phase_correction: bool = True,
    keep_diagnostics: bool = False,
    wavelength_min_m: float = 1.15e-6,
    wavelength_max_m: float = 1.30e-6,
    wavelength_channels: int = 151,
    gd_search_min_m: float = -30e-6,
    gd_search_max_m: float = 30e-6,
) -> P2VM:
    """Read the calibration sequence and return the full 24-output response/inverse.

    Wavelength and coarse GD bounds use metres. The optional phase reference
    defines the instrumental phase and uses its selected DARK, or ``main_dark``.
    """

    if phase_dark is not None and phase_reference is None:
        raise ValueError("phase_dark requires phase_reference")
    grid = np.linspace(wavelength_min_m, wavelength_max_m, wavelength_channels)
    spectra = extract_p2vm_spectra(
        flat,
        wave,
        main_dark,
        telescope_flats=telescope_flats,
        baseline_cubes=baseline_cubes,
        wavelength_m=grid,
        phase_reference=phase_reference if phase_correction else None,
        phase_dark=phase_dark if phase_correction else None,
    )
    return fit_p2vm(
        spectra,
        phase_correction=phase_correction,
        keep_diagnostics=keep_diagnostics,
        gd_search_min_m=gd_search_min_m,
        gd_search_max_m=gd_search_max_m,
    )


def extract_p2vm_spectra(
    flat: Flat,
    wave: Wave,
    main_dark: Dark,
    *,
    telescope_flats: Mapping[str, Path] | Iterable[Path],
    baseline_cubes: Mapping[str, Path] | Iterable[Path],
    wavelength_m: NDArray,
    phase_reference: Path | None = None,
    phase_dark: Dark | None = None,
) -> P2VMSpectra:
    """Extract calibration files and infer the retained physical output mapping."""

    telescope_flats = identify_raw_exposures(telescope_flats, "FLAT_RAW")
    baseline_cubes = identify_raw_exposures(baseline_cubes, "P2VM_RAW")
    if phase_dark is not None and phase_reference is None:
        raise ValueError("phase_dark requires phase_reference")
    extractor = prepare_extraction(flat, wave, wavelength_m)
    for exposures in (telescope_flats, baseline_cubes):
        for _, raw in exposures.values():
            check_raw_calibrations(raw, main_dark, flat)
    baseline_spectra = np.asarray(
        [
            extract_spectra_from_file(baseline_cubes[name][0], extractor, main_dark)
            for name in BASELINES
        ]
    )
    telescope_spectra = np.asarray(
        [
            extract_spectra_from_file(telescope_flats[name][0], extractor, main_dark)
            for name in TELESCOPES
        ]
    )
    phase_reference_spectra = np.empty((N_OUTPUTS, 0, len(extractor.wavelength_m)))
    if phase_reference is not None:
        selected_dark = main_dark if phase_dark is None else phase_dark
        raw = read_raw_metadata(phase_reference, category="SINGLE_SCI_RAW")
        check_raw_calibrations(raw, selected_dark, flat)
        phase_reference_spectra = extract_spectra_from_file(
            phase_reference, extractor, selected_dark
        )
    return P2VMSpectra(
        telescope_spectra,
        baseline_spectra,
        phase_reference_spectra,
        _identify_output_mapping(telescope_spectra, baseline_spectra),
        extractor.wavelength_m,
    )


def _identify_output_mapping(
    telescope_spectra: NDArray, baseline_spectra: NDArray
) -> dict[int, list[int]]:
    """Require telescope illumination and baseline modulation to identify the same outputs."""
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

    baseline_outputs = {}
    for baseline, (telescope_1, telescope_2) in enumerate(BASELINE_PAIRS):
        from_telescopes = set(outputs_by_telescope[telescope_1]) & set(
            outputs_by_telescope[telescope_2]
        )
        from_modulation = set(outputs_by_modulation[baseline])
        if from_telescopes != from_modulation:
            raise ValueError(
                f"Output mapping mismatch for baseline {BASELINES[baseline]}: "
                f"{sorted(from_telescopes)} from flats, "
                f"{sorted(from_modulation)} from modulation"
            )
        baseline_outputs[baseline] = sorted(outputs_by_modulation[baseline])
    return baseline_outputs
