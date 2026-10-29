"""P2VM calibration: scan response, photometric normalization and phase reference."""

from collections.abc import Mapping
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
from .io.raw import check_raw_calibrations, read_raw_metadata
from .products import P2VMSpectra, Dark, Flat, P2VM, Wave
from .reduction import estimate_group_delay


def fit_p2vm(
    spectra: P2VMSpectra,
    *,
    phase_reference: NDArray | None = None,
    keep_diagnostics: bool = False,
    gd_search_min_m: float = -30e-6,
    gd_search_max_m: float = 30e-6,
) -> P2VM:
    """Fit the full detector response and return its per-wavelength inverse.

    Scans: (baseline, output, frame, wavelength); flats: (telescope, output,
    frame, wavelength). Wavelength and GD bounds are metres. All 24 outputs
    enter each baseline fit, including cross-talk. Supply extracted reference
    spectra (output, frame, wavelength) to apply phase correction, or None.
    Optional scan diagnostics change only the retained arrays, not the response
    or inverse.
    """
    log.info("--- Step: P2VM ---")
    _check_input_axes(spectra)
    wavelength_m = spectra.wavelength_m
    if phase_reference is not None and (
        phase_reference.ndim != 3
        or phase_reference.shape[0] != N_OUTPUTS
        or phase_reference.shape[-1] != len(wavelength_m)
        or phase_reference.shape[1] == 0
    ):
        raise ValueError("Phase reference must match output/frame/wavelength axes")
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
    if phase_reference is not None:
        phase_rad = _apply_phase_reference(
            phase_rad,
            p2vm,
            phase_reference,
            wavelength_m,
            spectra.baseline_outputs,
            search_range=search_range,
        )
        p2vm = _invert_v2pm(build_v2pm(transmission, amplitude, phase_rad))

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
    ):
        raise ValueError(
            "P2VM spectra must match telescope/output/baseline/wavelength axes"
        )
    if (3 * len(wavelength_m)) // 4 - len(wavelength_m) // 4 < 2:
        raise ValueError("P2VM requires at least two central-band wavelength samples")


def _fit_scan_responses(
    spectra: P2VMSpectra, search_range: tuple[float, float], *, keep_diagnostics: bool
) -> tuple[NDArray, NDArray, NDArray, dict[str, NDArray]]:
    """Fit each baseline scan: ellipse -> OPD zero point -> detector response.

    Return amplitude [ADU], phase [rad], DC [ADU], each (output, baseline,
    wavelength), and requested diagnostics. Scan OPD/GD are metres; ellipse
    radius is dimensionless. All output rows, including cross-talk, are fitted.
    """
    wavelength_m = spectra.wavelength_m
    n_frames, n_wave = spectra.baseline_spectra[0].shape[1:]
    amplitude = np.zeros((N_OUTPUTS, N_BASELINES, n_wave))
    phase_rad = np.zeros((N_OUTPUTS, N_BASELINES, n_wave))
    dc_adu = np.zeros((N_OUTPUTS, N_BASELINES, n_wave))
    rms_adu = np.zeros_like(dc_adu)
    scan_phases = []
    scan_radii = []
    opd_m = np.zeros((N_BASELINES, n_frames))
    gd_m = np.zeros((N_BASELINES, n_frames))

    for baseline in range(N_BASELINES):
        log.info(f"Fitting baseline {BASELINES[baseline]}...")
        baseline_spectra = spectra.baseline_spectra[baseline]
        outputs = spectra.baseline_outputs[baseline]
        log.debug("    Fitting ellipses for phase extraction...")
        scan_phase_rad, radius = fit_abcd_ellipse(baseline_spectra[outputs])
        if keep_diagnostics:
            scan_phases.append(scan_phase_rad)
            scan_radii.append(radius)
        opd_m[baseline], gd_m[baseline] = _estimate_scan_opd(
            scan_phase_rad, radius, wavelength_m, search_range
        )
        (
            amplitude[:, baseline, :],
            phase_rad[:, baseline, :],
            dc_adu[:, baseline, :],
            rms_adu[:, baseline, :],
        ) = _fit_interferometric_response(
            baseline_spectra, opd_m[baseline], wavelength_m
        )

    diagnostics = {}
    if keep_diagnostics:
        diagnostics = dict(
            opd_m=opd_m,
            group_delay_m=gd_m,
            ellipse_phase_rad=np.asarray(scan_phases),
            ellipse_amplitude=np.asarray(scan_radii),
            dc_term_adu=dc_adu,
            fit_rms_adu=rms_adu,
        )
    return amplitude, phase_rad, dc_adu, diagnostics


def _estimate_scan_opd(
    phase_rad: NDArray,
    radius: NDArray,
    wavelength_m: NDArray,
    search_range: tuple[float, float],
) -> tuple[NDArray, NDArray]:
    """Convert ellipse phase/radius (frame, wavelength) to scan OPD/GD (frame,).

    Both paths are metres. Use the central half-band, convert phase to
    phase*lambda/(2*pi), and remove the intercept of its linear fit against GD.
    """
    phasor = radius * np.exp(1j * phase_rad)
    n_wave = len(wavelength_m)
    start, end = n_wave // 4, (n_wave * 3) // 4
    opd_m = np.mean((phase_rad * wavelength_m)[:, start:end] / 2 / np.pi, axis=1)
    gd_m = estimate_group_delay(
        phasor[:, start:end], wavelength_m[start:end], search_range=search_range
    )
    _, opd_zero_m = np.polyfit(gd_m, opd_m, 1)
    opd_m -= opd_zero_m
    log.debug(f"    Removing OPD zero point {opd_zero_m:.9g} m...")
    return opd_m, gd_m


def _fit_interferometric_response(
    baseline_spectra: NDArray, opd_m: NDArray, wavelength_m: NDArray
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Fit c+a*cos(phi)+b*sin(phi) to all outputs, retaining amplitude/phase/DC/RMS.

    Amplitude, DC and RMS are raw ADU at this stage; phase is in radians.
    With phase=atan2(b,a), the response is c+amplitude*cos(phi-phase).
    Arrays returned here are (output, wavelength); OPD is (frame,) in metres.
    The response amplitude is normalized later using the photometric transmission.
    """
    n_wave = len(wavelength_m)
    amplitude = np.zeros((N_OUTPUTS, n_wave))
    phase_rad = np.zeros((N_OUTPUTS, n_wave))
    dc_adu = np.zeros((N_OUTPUTS, n_wave))
    rms_adu = np.zeros_like(dc_adu)
    scan_phase_rad = 2 * np.pi * opd_m[:, None] / wavelength_m[None, :]
    design = np.ones((len(scan_phase_rad), 3))
    log.debug("    Fitting P2VM coefficients...")
    for wave_index in range(n_wave):
        design[:, 1] = np.sin(scan_phase_rad[:, wave_index])
        design[:, 2] = np.cos(scan_phase_rad[:, wave_index])
        inverse = np.linalg.pinv(design)
        coefficients = np.einsum(
            "if,of->oi", inverse, baseline_spectra[..., wave_index]
        )
        a, b, c = coefficients[:, 2], coefficients[:, 1], coefficients[:, 0]
        amplitude[:, wave_index] = np.sqrt(a**2 + b**2)
        phase_rad[:, wave_index] = np.arctan2(b, a)
        dc_adu[:, wave_index] = c
        residual = baseline_spectra[..., wave_index] - coefficients @ design.T
        rms_adu[:, wave_index] = np.sqrt(np.mean(residual**2, axis=1))
    return amplitude, phase_rad, dc_adu, rms_adu


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
    amplitude: NDArray,
    phase_rad: NDArray,
    dc_adu: NDArray,
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
        a_phase_rad = phase_rad[outputs[0]]
        phase_rad[outputs] -= a_phase_rad[None, :, :]

    log.info("    Transmission normalization (mean over 24 outputs -> 1)...")
    transmission /= transmission.mean(axis=0)[None, :, :]

    log.info("    Coherence normalization...")
    for baseline in range(N_BASELINES):
        telescope_1, telescope_2 = BASELINE_PAIRS[baseline]
        for wave_index in range(transmission.shape[-1]):
            photometry = (
                np.linalg.pinv(transmission[:, :, wave_index])
                @ dc_adu[:, baseline, wave_index]
            )
            flux_product = photometry[telescope_1] * photometry[telescope_2]
            if not np.isfinite(flux_product) or flux_product <= 0:
                raise ValueError(
                    f"Invalid photometric normalization for baseline {baseline}, "
                    f"wavelength index {wave_index}"
                )
            amplitude[:, baseline, wave_index] /= np.sqrt(flux_product)


def _invert_v2pm(v2pm_matrix: NDArray) -> NDArray:
    """Compute the same per-wavelength pseudoinverse before and after phase correction."""
    log.info("V2PM -> P2VM")
    n_wave = v2pm_matrix.shape[-1]
    p2vm = np.zeros((N_DATA, N_OUTPUTS, n_wave))
    for wave_index in range(n_wave):
        rank = np.linalg.matrix_rank(v2pm_matrix[:, :, wave_index])
        if rank != N_DATA:
            raise ValueError(
                f"V2PM at wavelength index {wave_index} has rank {rank}; expected {N_DATA}"
            )
        p2vm[:, :, wave_index] = np.linalg.pinv(v2pm_matrix[:, :, wave_index])
    return p2vm


def build_v2pm(
    transmission: NDArray, coherence_amplitude: NDArray, coherence_phase_rad: NDArray
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
    coherent_adu = channels[:, REAL_SLICE, :] + 1j * channels[:, IMAG_SLICE, :]
    center = len(wavelength_m) // 2
    anchor = TELESCOPES.index("4")
    # 43, 42, 41 identify the other telescopes relative to T4 explicitly.
    anchor_baselines = [
        (baseline, second)
        for baseline, (first, second) in enumerate(BASELINE_PAIRS)
        if first == anchor
    ]

    # Reconstruct each frame's telescope phase at the central wavelength.
    phase_by_tel_rad = np.zeros((len(coherent_adu), N_TELESCOPES))
    for baseline, telescope in anchor_baselines:
        gd_m = estimate_group_delay(
            coherent_adu[:, baseline, :], wavelength_m, search_range=search_range
        )
        phasor = np.exp(-1j * 2 * np.pi * gd_m[:, None] / wavelength_m)
        phase_by_tel_rad[:, telescope] = (
            np.angle(coherent_adu[:, baseline, center] * phasor[:, center])
            + 2 * np.pi * gd_m / wavelength_m[center]
        )
    phase_by_tel_rad -= phase_by_tel_rad.mean(axis=0)[None, :]

    # Remove temporal phase excursions (scaled as 1/lambda), then sum frames.
    coherent_sum_adu = np.zeros((N_BASELINES, len(wavelength_m)), dtype=complex)
    for baseline, (telescope_1, telescope_2) in enumerate(BASELINE_PAIRS):
        delta_phase_rad = (
            phase_by_tel_rad[:, telescope_2] - phase_by_tel_rad[:, telescope_1]
        )
        coherent_sum_adu[baseline] = (
            coherent_adu[:, baseline]
            * np.exp(
                -1j * delta_phase_rad[:, None] * wavelength_m[center] / wavelength_m
            )
        ).sum(axis=0)

    # Remove the residual spectral GD with telescope phasors anchored to T4.
    phasor_by_tel = np.ones((N_TELESCOPES, len(wavelength_m)), dtype=complex)
    gd_m = estimate_group_delay(
        coherent_sum_adu, wavelength_m, search_range=search_range
    )
    for baseline, telescope in anchor_baselines:
        phasor_by_tel[telescope] = np.exp(2j * np.pi * gd_m[baseline] / wavelength_m)
    for baseline, (telescope_1, telescope_2) in enumerate(BASELINE_PAIRS):
        coherent_sum_adu[baseline] *= (
            phasor_by_tel[telescope_1] * phasor_by_tel[telescope_2].conj()
        )

    # Apply the reference phase only to the matching physical output quartet.
    output_baselines = {
        output: baseline
        for baseline, outputs in baseline_outputs.items()
        for output in outputs
    }
    for output in range(N_OUTPUTS):
        baseline = output_baselines[output]
        phasor = (
            np.exp(1j * phase_rad[output, baseline]) * coherent_sum_adu[baseline].conj()
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

    phase_rad = np.zeros((n_frames, n_wave))
    radius = np.zeros((n_frames, n_wave))
    unit_rhs = np.ones(n_frames)

    for wave_index in range(n_wave):
        A, C, B, D = spec[:, :, wave_index]

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

        phase_rad[:, wave_index] = np.unwrap(np.arctan2(Y_fit, X_fit))
        radius[:, wave_index] = np.sqrt(X_fit**2 + Y_fit**2)

    return phase_rad, radius


def make_p2vm(
    flat: Flat,
    wave: Wave,
    main_dark: Dark,
    *,
    telescope_flats: Mapping[str, Path],
    baseline_cubes: Mapping[str, Path],
    phase_reference: NDArray | None = None,
    keep_diagnostics: bool = False,
    wavelength_min_m: float = 1.15e-6,
    wavelength_max_m: float = 1.30e-6,
    wavelength_channels: int = 151,
    gd_search_min_m: float = -30e-6,
    gd_search_max_m: float = 30e-6,
) -> P2VM:
    """Read the calibration sequence and return the full 24-output response/inverse.

    Wavelength and coarse GD bounds use metres. The optional phase reference is
    already extracted, dark-subtracted spectra on this exact wavelength grid,
    with axes (output, frame, wavelength). Its file and DARK are chosen by the caller.
    """

    grid = np.linspace(wavelength_min_m, wavelength_max_m, wavelength_channels)
    spectra = extract_p2vm_spectra(
        flat,
        wave,
        main_dark,
        telescope_flats=telescope_flats,
        baseline_cubes=baseline_cubes,
        wavelength_m=grid,
    )
    return fit_p2vm(
        spectra,
        phase_reference=phase_reference,
        keep_diagnostics=keep_diagnostics,
        gd_search_min_m=gd_search_min_m,
        gd_search_max_m=gd_search_max_m,
    )


def extract_p2vm_spectra(
    flat: Flat,
    wave: Wave,
    main_dark: Dark,
    *,
    telescope_flats: Mapping[str, Path],
    baseline_cubes: Mapping[str, Path],
    wavelength_m: NDArray,
) -> P2VMSpectra:
    """Extract explicit telescope/baseline file maps and identify detector outputs."""

    if set(telescope_flats) != set(TELESCOPES):
        raise ValueError("Telescope flats require a path for each of 4,3,2,1")
    if set(baseline_cubes) != set(BASELINES):
        raise ValueError("Baseline scans require a path for each of 43,42,41,32,31,21")
    if len({Path(path) for path in telescope_flats.values()}) != N_TELESCOPES:
        raise ValueError("Each telescope requires a different flat file")
    if len({Path(path) for path in baseline_cubes.values()}) != N_BASELINES:
        raise ValueError("Each baseline requires a different scan file")
    extractor = prepare_extraction(flat, wave, wavelength_m)

    baseline_spectra = []
    for baseline in BASELINES:
        path = baseline_cubes[baseline]
        raw = read_raw_metadata(path, category="P2VM_RAW")
        for telescope, state in enumerate(raw.shutters, 1):
            if state is not None and state != (str(telescope) in baseline):
                raise ValueError(
                    f"Scan for {baseline} conflicts with recorded shutters: {path}"
                )
        check_raw_calibrations(raw, main_dark, flat)
        baseline_spectra.append(extract_spectra_from_file(path, extractor, main_dark))
    baseline_spectra = np.asarray(baseline_spectra)

    telescope_spectra = []
    for name in TELESCOPES:
        path = telescope_flats[name]
        raw = read_raw_metadata(path, category="FLAT_RAW")
        for telescope, state in enumerate(raw.shutters, 1):
            if state is not None and state != (str(telescope) in name):
                raise ValueError(
                    f"Flat for T{name} conflicts with recorded shutters: {path}"
                )
        check_raw_calibrations(raw, main_dark, flat)
        telescope_spectra.append(extract_spectra_from_file(path, extractor, main_dark))
    telescope_spectra = np.asarray(telescope_spectra)
    return P2VMSpectra(
        telescope_spectra,
        baseline_spectra,
        _identify_output_mapping(telescope_spectra, baseline_spectra),
        extractor.wavelength_m,
    )


def _identify_output_mapping(
    telescope_spectra: NDArray, baseline_spectra: NDArray
) -> dict[int, list[int]]:
    """Require telescope illumination and baseline modulation to identify the same outputs."""
    illuminated = {}
    for telescope, flux in enumerate(telescope_spectra):
        mean_flux = flux.mean(axis=(1, 2))
        active = np.argsort(mean_flux)[-N_OUTPUTS // 2 :]
        illuminated[telescope] = np.sort(active).tolist()

    modulated = {}
    for baseline, flux in enumerate(baseline_spectra):
        std_flux = flux.std(axis=1).mean(axis=-1)
        active = np.argsort(std_flux)[-4:]
        modulated[baseline] = np.sort(active).tolist()

    baseline_outputs = {}
    for baseline, (telescope_1, telescope_2) in enumerate(BASELINE_PAIRS):
        from_flats = set(illuminated[telescope_1]) & set(illuminated[telescope_2])
        from_scan = set(modulated[baseline])
        if from_flats != from_scan:
            raise ValueError(
                f"Output mapping mismatch for baseline {BASELINES[baseline]}: "
                f"{sorted(from_flats)} from flats, "
                f"{sorted(from_scan)} from modulation"
            )
        baseline_outputs[baseline] = sorted(modulated[baseline])
    return baseline_outputs
