"""P2VM calibration: scan response, photometric normalization and phase reference."""
from planetes_pipe.model.raw import Raw
from collections.abc import Mapping

import numpy as np
from numpy.typing import NDArray

from .. import log
from planetes_pipe.processing.common import order_raws
from planetes_pipe.processing.extraction import extract_spectra, prepare_extraction
from planetes_pipe.model.instrument import layout, _check_baselines, check_setup
from planetes_pipe.model.products import P2VMSpectra, Dark, Flat, P2VM, Wave
from planetes_pipe.model.quality import P2VMQC
from planetes_pipe.processing.group_delay import estimate_group_delay


def fit_p2vm(
    spectra: P2VMSpectra,
    *,
    phase_reference: NDArray | None = None,
    scan_opd_m: NDArray | None = None,
    keep_diagnostics: bool = False,
    gd_search_min_m: float = -30e-6,
    gd_search_max_m: float = 30e-6,
) -> P2VM:
    """Fit the full detector response and invert it at each wavelength.

    Parameters
    ----------
    spectra : P2VMSpectra
        ADU telescope spectra (4, 24, frame, wavelength), baseline scans
        (6, 24, frame, wavelength), physical output mapping and wavelength grid [m].
        Telescope order is 4,3,2,1; baseline order is 43,42,41,32,31,21.
    phase_reference : ndarray or None, optional
        Already dark-subtracted reference spectra (24, frame, wavelength) in ADU,
        on the same grid. None skips instrumental phase correction. File and DARK
        selection belong to the caller; this is not astronomical calibration.
    scan_opd_m : ndarray or None, optional
        Recorded mechanical positions converted to an OPD guide [m], shaped
        (6 baselines, frame). It chooses integer phase cycles only; measured
        ellipse phase still supplies fine motion. None uses ordinary unwrap.
    keep_diagnostics : bool, optional
        Retain scan OPD/GD, ellipse phase/radius, DC and residual RMS when True.
        This changes which arrays are returned, not the fit or inverse.
    gd_search_min_m, gd_search_max_m : float, optional
        Coarse GD initialization interval in metres, default +/-30e-6.
        Newton refinement is unconstrained; see estimate_group_delay.

    Returns
    -------
    p2vm : P2VM
        Inverse (16, 24, wavelength), dimensionless transmission and coherence
        amplitude, response phases [rad], grid [m] and physical output mapping.
        Requested scan diagnostics keep their units documented in P2VM.

    Notes
    -----
    Fit all 24 output responses to each baseline, including cross-talk. Derive
    photometric transmission, normalize phases/amplitudes, then invert V2PM.
    A supplied reference adjusts instrumental phases and triggers a second
    inversion. Each wavelength's V2PM must have numerical rank 16.
    """
    log.info("--- Step: P2VM ---")
    _check_input_axes(spectra)
    wavelength_m = spectra.wavelength_m
    if scan_opd_m is not None:
        scan_opd_m = np.asarray(scan_opd_m, dtype=float)
        if (scan_opd_m.shape != (layout.n_baselines, spectra.baseline_spectra.shape[2])
                or not np.isfinite(scan_opd_m).all()):
            raise ValueError("scan_opd_m must be finite with shape (baseline, frame)")
    if phase_reference is not None and (
        phase_reference.ndim != 3
        or phase_reference.shape[0] != layout.n_outputs
        or phase_reference.shape[-1] != len(wavelength_m)
        or phase_reference.shape[1] == 0
    ):
        raise ValueError("Phase reference must match output/frame/wavelength axes")
    search_range = (gd_search_min_m, gd_search_max_m)

    # Scans: ABCD ellipse -> OPD/GD -> sinusoidal response of all 24 outputs.
    amplitude, phase_rad, dc_adu, diagnostics = _fit_scan_responses(
        spectra, search_range, keep_diagnostics=keep_diagnostics, scan_opd_m=scan_opd_m
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
        or spectra.telescope_spectra.shape[:2] != (layout.n_telescopes, layout.n_outputs)
        or spectra.telescope_spectra.shape[-1] != len(wavelength_m)
        or spectra.baseline_spectra.ndim != 4
        or spectra.baseline_spectra.shape[:2] != (layout.n_baselines, layout.n_outputs)
        or spectra.baseline_spectra.shape[-1] != len(wavelength_m)
    ):
        raise ValueError(
            "P2VM spectra must match telescope/output/baseline/wavelength axes"
        )
    if (3 * len(wavelength_m)) // 4 - len(wavelength_m) // 4 < 2:
        raise ValueError("P2VM requires at least two central-band wavelength samples")


def _fit_scan_responses(
    spectra: P2VMSpectra, search_range: tuple[float, float], *, keep_diagnostics: bool,
    scan_opd_m: NDArray | None = None,
) -> tuple[NDArray, NDArray, NDArray, dict[str, NDArray]]:
    """Fit each baseline scan from ABCD phase through OPD to all-output response.

    Parameters
    ----------
    spectra : P2VMSpectra
        ADU scans (6 baselines, 24 outputs, frame, wavelength), physical output
        assignments and wavelength grid [m]. Telescope spectra are unused here.
    search_range : tuple of float
        Coarse GD initialization interval in metres; Newton is unconstrained.
    keep_diagnostics : bool
        Retain the scan/intermediate arrays when True; fitting is unchanged.

    Returns
    -------
    amplitude, phase_rad, dc_adu : ndarray, shape (24, 6, wavelength)
        Fitted response amplitude [ADU], phase [rad] and constant term [ADU].
        Amplitudes are still unnormalized at this stage; cross-talk is retained.
    diagnostics : dict of str to ndarray
        Empty unless requested. OPD/GD use metres, ellipse phase uses radians,
        ellipse radius is dimensionless, and DC/residual RMS use ADU.
    """
    wavelength_m = spectra.wavelength_m
    n_frames, n_wave = spectra.baseline_spectra[0].shape[1:]
    amplitude = np.zeros((layout.n_outputs, layout.n_baselines, n_wave))
    phase_rad = np.zeros((layout.n_outputs, layout.n_baselines, n_wave))
    dc_adu = np.zeros((layout.n_outputs, layout.n_baselines, n_wave))
    rms_adu = np.zeros_like(dc_adu)
    scan_phases = []
    scan_radii = []
    opd_m = np.zeros((layout.n_baselines, n_frames))
    gd_m = np.zeros((layout.n_baselines, n_frames))

    for baseline in range(layout.n_baselines):
        log.info(f"Fitting baseline {layout.baselines[baseline]}...")
        baseline_spectra = spectra.baseline_spectra[baseline]
        outputs = spectra.baseline_outputs[baseline]
        log.debug("    Fitting ellipses for phase extraction...")
        scan_phase_rad, radius = fit_abcd_ellipse(baseline_spectra[outputs])
        if scan_opd_m is not None:
            scan_phase_rad = unwrap_scan_phase(scan_phase_rad, scan_opd_m[baseline], wavelength_m)
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
    """Convert the ellipse phase to scan OPD and determine its zero point from GD.

    Parameters
    ----------
    phase_rad, radius : ndarray, shape (frame, wavelength)
        Unwrapped ellipse phase [rad] and dimensionless unit-circle radius.
    wavelength_m : ndarray, shape (wavelength,)
        Scan wavelength grid in metres.
    search_range : tuple of float
        Coarse GD initialization interval in metres.

    Returns
    -------
    opd_m, gd_m : ndarray, shape (frame,)
        Optical path and group delay in metres, using the ellipse phase sign.

    Notes
    -----
    Use the central half-band to average phase * wavelength / (2*pi) and to
    estimate GD from radius * exp(1j*phase). Fit OPD versus GD and subtract only
    the intercept from OPD; the fitted slope is not applied as a rescaling.
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
    """Fit c + a*cos(phi) + b*sin(phi) to every detector output in one baseline scan.

    Parameters
    ----------
    baseline_spectra : ndarray, shape (24 outputs, frame, wavelength)
        Dark-subtracted, extracted ADU spectra, including off-baseline outputs.
    opd_m : ndarray, shape (frame,)
        Scan optical paths in metres with the chosen zero point and ellipse sign.
    wavelength_m : ndarray, shape (wavelength,)
        Extracted wavelength grid in metres; phi = 2*pi*OPD/wavelength.

    Returns
    -------
    amplitude : ndarray, shape (24, wavelength)
        sqrt(a*a + b*b) in ADU; photometric normalization occurs later.
    phase_rad : ndarray, shape (24, wavelength)
        atan2(b, a) [rad], so the model is c + amplitude*cos(phi - phase).
    dc_adu, rms_adu : ndarray, shape (24, wavelength)
        Constant term and unweighted population RMS of fit residuals, in ADU.

    Notes
    -----
    The design columns are ordered (1, sin(phi), cos(phi)); use an unweighted
    pseudoinverse independently at each wavelength. No channels are rejected here.
    """
    n_wave = len(wavelength_m)
    amplitude = np.zeros((layout.n_outputs, n_wave))
    phase_rad = np.zeros((layout.n_outputs, n_wave))
    dc_adu = np.zeros((layout.n_outputs, n_wave))
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
    """Estimate relative telescope illumination from frame-median flat spectra.

    Parameters
    ----------
    telescope_spectra : ndarray, shape (4 telescopes, 24 outputs, frame, wavelength)
        Extracted ADU flat spectra in T4,T3,T2,T1 order.

    Returns
    -------
    transmission : ndarray, shape (24 outputs, 4 telescopes, wavelength)
        Dimensionless median spectrum divided by its telescope's total over all
        outputs and wavelengths. The later response normalization sets the mean
        over outputs to one separately at each telescope/wavelength.
    """
    transmission = np.zeros((layout.n_outputs, layout.n_telescopes, telescope_spectra.shape[-1]))
    for telescope in range(layout.n_telescopes):
        log.info(f"Measuring transmission for telescope {layout.telescopes[telescope]}...")
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
    """Set response phase origins and normalize transmission and coherent amplitude.

    Parameters
    ----------
    transmission : ndarray, shape (24 outputs, 4 telescopes, wavelength)
        Relative telescope response; modified in place to mean one over outputs.
    amplitude : ndarray, shape (24 outputs, 6 baselines, wavelength)
        Fitted ADU amplitude; modified in place to a dimensionless response.
    phase_rad : ndarray, shape (24 outputs, 6 baselines, wavelength)
        Fitted phase in radians; modified in place to the A-output phase convention.
    dc_adu : ndarray, shape (24 outputs, 6 baselines, wavelength)
        Constant fitted response in ADU, used to solve baseline photometry.
        Not modified by this function.
    baseline_outputs : mapping of int to list of int
        Baseline-axis index 0..5 -> four detector-output indices in A,C,B,D order.

    Notes
    -----
    For each output quartet, subtract its A row from all six baseline response
    components, including cross-talk. Solve transmission against DC by pseudoinverse,
    then divide amplitude by sqrt(P_i*P_j). A nonpositive/nonfinite photometric
    product raises ValueError. Returns None; the first three arrays are updated.
    """
    log.info("P2VM Normalization...")
    log.info("    Phase normalization (A=0)...")
    for baseline in range(layout.n_baselines):
        outputs = baseline_outputs[baseline]
        a_phase_rad = phase_rad[outputs[0]]
        phase_rad[outputs] -= a_phase_rad[None, :, :]

    log.info("    Transmission normalization (mean over 24 outputs -> 1)...")
    transmission /= transmission.mean(axis=0)[None, :, :]

    log.info("    Coherence normalization...")
    for baseline in range(layout.n_baselines):
        telescope_1, telescope_2 = layout.baseline_pairs[baseline]
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
    """Compute the wavelength-wise inverse needed to reconstruct 16 data channels.

    Parameters
    ----------
    v2pm_matrix : ndarray, shape (24 outputs, 16 data channels, wavelength)
        Dimensionless forward response matrix.

    Returns
    -------
    p2vm : ndarray, shape (16 data channels, 24 outputs, wavelength)
        NumPy pseudoinverse at each wavelength.

    Notes
    -----
    Require numerical rank 16 using NumPy's default matrix_rank criterion;
    otherwise raise ValueError identifying the wavelength. No condition-number
    threshold or alternate regularization is introduced.
    """
    log.info("V2PM -> P2VM")
    n_wave = v2pm_matrix.shape[-1]
    p2vm = np.zeros((layout.n_data, layout.n_outputs, n_wave))
    for wave_index in range(n_wave):
        rank = np.linalg.matrix_rank(v2pm_matrix[:, :, wave_index])
        if rank != layout.n_data:
            raise ValueError(
                f"V2PM at wavelength index {wave_index} has rank {rank}; expected {layout.n_data}"
            )
        p2vm[:, :, wave_index] = np.linalg.pinv(v2pm_matrix[:, :, wave_index])
    return p2vm


def build_v2pm(
    transmission: NDArray, coherence_amplitude: NDArray, coherence_phase_rad: NDArray
) -> NDArray:
    """Expand physical response terms into the forward visibility-to-pixel matrix.

    Parameters
    ----------
    transmission : ndarray, shape (24 outputs, 4 telescopes, wavelength)
        Dimensionless photometric transmission, normalized to mean one over outputs.
    coherence_amplitude : ndarray, shape (24 outputs, 6 baselines, wavelength)
        Dimensionless interferometric response after photometric normalization.
    coherence_phase_rad : ndarray, shape (24 outputs, 6 baselines, wavelength)
        Response phase in radians, with convention amplitude * cos(phi - phase).

    Returns
    -------
    v2pm : ndarray, shape (24 outputs, 16 data channels, wavelength)
        Dimensionless forward matrix. Columns are T4,T3,T2,T1, then Re(C_ij),
        then Im(C_ij), each coherent group ordered 43,42,41,32,31,21.

    Notes
    -----
    Coherent columns are amplitude*cos(phase) and amplitude*sin(phase).
    Divide the entire matrix by 24 to preserve the instrumental flux convention.
    All cross-talk terms remain present; no inversion is performed here.
    """
    matrix = np.zeros((layout.n_outputs, layout.n_data, transmission.shape[-1]))
    matrix[:, :layout.n_telescopes] = transmission
    matrix[:, layout.real_slice] = coherence_amplitude * np.cos(coherence_phase_rad)
    matrix[:, layout.imag_slice] = coherence_amplitude * np.sin(coherence_phase_rad)
    matrix /= layout.n_outputs
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
    """Correct instrumental response phases using a reference anchored to telescope T4.

    Parameters
    ----------
    phase_rad : ndarray, shape (24 outputs, 6 baselines, wavelength)
        Current response phases in radians; copied before modification.
    p2vm : ndarray, shape (16 data channels, 24 outputs, wavelength)
        Initial response inverse used to reconstruct reference coherent flux.
    reference_spectra : ndarray, shape (24 outputs, frame, wavelength)
        Reference spectra in ADU, already extracted with their selected dark.
    wavelength_m : ndarray, shape (wavelength,)
        Reference and response wavelength grid in metres.
    baseline_outputs : mapping of int to list of int
        Baseline-axis index 0..5 -> four output indices in A,C,B,D order.
    search_range : tuple of float
        Coarse GD initialization interval in metres.

    Returns
    -------
    corrected_phase_rad : ndarray, shape (24 outputs, 6 baselines, wavelength)
        New phases [rad]. Only each output's own physical baseline is corrected;
        cross-talk phase components keep their scan normalization.

    Notes
    -----
    Recover telescope phases from baselines 43,42,41, remove their temporal mean,
    scale as 1/lambda and coherently SUM the corrected reference frames. Remove
    the residual chromatic GD with T4-anchored telescope phasors, then subtract
    the integrated reference phase. The caller rebuilds and reinverts V2PM.
    """
    log.info("Applying phase-reference correction...")
    phase_rad = phase_rad.copy()
    channels = np.einsum("dow,ofw->fdw", p2vm, reference_spectra)
    coherent_adu = channels[:, layout.real_slice, :] + 1j * channels[:, layout.imag_slice, :]
    center = len(wavelength_m) // 2
    anchor = layout.telescopes.index("4")
    # 43, 42, 41 identify the other telescopes relative to T4 explicitly.
    anchor_baselines = [
        (baseline, second)
        for baseline, (first, second) in enumerate(layout.baseline_pairs)
        if first == anchor
    ]

    # Reconstruct each frame's telescope phase at the central wavelength.
    phase_by_tel_rad = np.zeros((len(coherent_adu), layout.n_telescopes))
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
    coherent_sum_adu = np.zeros((layout.n_baselines, len(wavelength_m)), dtype=complex)
    for baseline, (telescope_1, telescope_2) in enumerate(layout.baseline_pairs):
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
    phasor_by_tel = np.ones((layout.n_telescopes, len(wavelength_m)), dtype=complex)
    gd_m = estimate_group_delay(
        coherent_sum_adu, wavelength_m, search_range=search_range
    )
    for baseline, telescope in anchor_baselines:
        phasor_by_tel[telescope] = np.exp(2j * np.pi * gd_m[baseline] / wavelength_m)
    for baseline, (telescope_1, telescope_2) in enumerate(layout.baseline_pairs):
        coherent_sum_adu[baseline] *= (
            phasor_by_tel[telescope_1] * phasor_by_tel[telescope_2].conj()
        )

    # Apply the reference phase only to the matching physical output quartet.
    output_baselines = {
        output: baseline
        for baseline, outputs in baseline_outputs.items()
        for output in outputs
    }
    for output in range(layout.n_outputs):
        baseline = output_baselines[output]
        phasor = (
            np.exp(1j * phase_rad[output, baseline]) * coherent_sum_adu[baseline].conj()
        )
        phase_rad[output, baseline] = np.angle(phasor)
    return phase_rad


def fit_abcd_ellipse(spec: NDArray) -> tuple[NDArray, NDArray]:
    """Recover scan phase and normalized radius from four ABCD intensity outputs.

    Parameters
    ----------
    spec : ndarray, shape (4 outputs, frame, wavelength)
        ADU intensities ordered A,C,B,D. The fit uses X=C-A and Y=D-B,
        centers/scales them, and solves a quadratic ellipse with unit right-hand side.

    Returns
    -------
    phase_rad : ndarray, shape (frame, wavelength)
        Unit-circle angle in radians, unwrapped along the frame axis.
    radius : ndarray, shape (frame, wavelength)
        Dimensionless distance from the fitted unit-circle center. This is not
        fringe contrast in ADU; a perfect ellipse has unit radius at any contrast.

    Notes
    -----
    For I_q = c + A*cos(phi + [0, pi, pi/2, 3*pi/2]_q), the recovered phase
    DECREASES with phi. Preserve this sign in scan OPD and response fitting.
    Scaling below 1e-9 is skipped; nonpositive quadratic terms use 1e-6.
    These numerical fallbacks do not certify an ellipse: zero input returns
    zero phase/radius. No scientific channel-rejection rule is imposed here.
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


def wavelength_grid(flat: Flat, wave: Wave) -> NDArray:
    """Common illuminated band, sampled at the median native pixel wavelengths.

    FLAT stores inclusive accepted trace-fit limits. Intersect their WAVE
    coverage, then retain the median output wavelength at each detector column.
    This keeps native pixel sampling without a manual band or channel count.
    """
    if flat.trace_x_range is None:
        raise ValueError("FLAT has no trace-fit limits; regenerate it before making P2VM")
    edges = np.take_along_axis(wave.wavelength_map_m, flat.trace_x_range, axis=1)
    lower = edges.min(axis=1).max()
    upper = edges.max(axis=1).min()
    native = np.median(wave.wavelength_map_m, axis=0)
    grid = np.sort(native[(native >= lower) & (native <= upper)])
    if len(grid) < 2:
        raise ValueError("Fitted traces have no common illuminated wavelength grid")
    return grid


def make_p2vm(
    flat: Flat,
    wave: Wave,
    main_dark: Dark,
    *,
    telescope_flats: list[Raw],
    baseline_cubes: list[Raw],
    phase_reference: NDArray | None = None,
    scan_opd_m: NDArray | None = None,
    keep_diagnostics: bool = False,
    gd_search_min_m: float = -30e-6,
    gd_search_max_m: float = 30e-6,
) -> P2VM:
    """Fit an instrumental P2VM on the common illuminated native wavelength grid.

    Parameters
    ----------
    flat : Flat
        Detector response and sparse extraction geometry.
    wave : Wave
        Wavelength map in metres on the FLAT detector grid.
    main_dark : Dark
        Mean dark in ADU used for both telescope-flat and baseline-scan sequences.
        Known exposure and detector settings must match each input.
    telescope_flats : list of Raw
        Four raw ADU cubes in any order, identified by recorded shutters.
    baseline_cubes : list of Raw
        Six raw ADU scan cubes in any order, identified by recorded shutters.
    phase_reference : ndarray or None, optional
        Extracted ADU spectra (24 outputs, frame, wavelength) on the exact target
        grid. The caller selects its independent DARK and extracts it beforehand.
        None skips the correction; a path is not accepted here.
    scan_opd_m : ndarray or None, optional
        Recorded OPD guide [m], shaped (baseline, frame), in BASELINES order.
        The caller explicitly converts delay-line positions and chooses signs.
    keep_diagnostics : bool, optional
        Retain scan diagnostics in the returned product; default False.
    gd_search_min_m, gd_search_max_m : float, optional
        Coarse GD initialization interval [m]; defaults -30e-6, +30e-6.
        Subsequent Newton refinement is unconstrained.

    Returns
    -------
    p2vm : P2VM
        Response terms and inverse on the saved target grid. Saving is explicit.

    Notes
    -----
    Frame counts must agree within the four telescope files and, separately,
    within the six baseline files. The two groups may have different frame counts.
    """

    grid = wavelength_grid(flat, wave)
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
        scan_opd_m=scan_opd_m,
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
    telescope_flats: list[Raw],
    baseline_cubes: list[Raw],
    wavelength_m: NDArray,
) -> P2VMSpectra:
    """Extract calibration spectra and identify the outputs belonging to each baseline.

    Parameters
    ----------
    flat : Flat
        Detector response and sparse extraction geometry.
    wave : Wave
        Wavelength map [m] on the same detector grid.
    main_dark : Dark
        Mean image [ADU] selected for every telescope and baseline input.
    telescope_flats : list of Raw
        Four raw cubes in any order, identified by recorded shutters.
    baseline_cubes : list of Raw
        Six raw scan cubes in any order, identified by recorded shutters.
    wavelength_m : ndarray, shape (wavelength,)
        Increasing target wavelengths [m], covered by all detector outputs.

    Returns
    -------
    spectra : P2VMSpectra
        Telescope spectra (4, 24, frame, wavelength) and baseline spectra
        (6, 24, frame, wavelength) in ADU, plus the grid and output mapping.
        Frame counts must agree within each group of files.

    Notes
    -----
    Recorded shutters determine the physical input identities. Known detector
    metadata must match the calibrations. Output identification requires agreement
    between telescope illumination and baseline modulation; it does not infer identity from names.
    """

    extractor = prepare_extraction(flat, wave, wavelength_m)

    baseline_spectra = []
    for raw in order_raws(baseline_cubes, layout.baselines):
        check_setup(raw.setup, main_dark.setup, exposure=True, role="DARK")
        check_setup(raw.setup, flat.setup, exposure=False, role="FLAT")
        baseline_spectra.append(extract_spectra(raw, extractor, main_dark))
    baseline_spectra = np.asarray(baseline_spectra)

    telescope_spectra = []
    for raw in order_raws(telescope_flats, layout.telescopes):
        check_setup(raw.setup, main_dark.setup, exposure=True, role="DARK")
        check_setup(raw.setup, flat.setup, exposure=False, role="FLAT")
        telescope_spectra.append(extract_spectra(raw, extractor, main_dark))
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
    """Associate each physical baseline with four outputs using two independent signals.

    Parameters
    ----------
    telescope_spectra : ndarray, shape (4, 24, frame, wavelength)
        Telescope-flat ADU spectra in T4,T3,T2,T1 order.
    baseline_spectra : ndarray, shape (6, 24, frame, wavelength)
        ADU scans in baseline order 43,42,41,32,31,21.

    Returns
    -------
    baseline_outputs : dict of int to list of int
        Baseline-axis index 0..5 -> four increasing detector-output indices.
        Their physical A,C,B,D order is an instrument-layout assumption.

    Notes
    -----
    Select the 12 highest mean-flux outputs for each telescope and the four
    highest temporal-modulation outputs for each baseline. Require the two
    illuminated telescope sets' intersection to equal the modulated set, otherwise
    raise ValueError. Ranking alone does not establish a signal-to-noise threshold.
    """
    illuminated = {}
    for telescope, flux in enumerate(telescope_spectra):
        mean_flux = flux.mean(axis=(1, 2))
        active = np.argsort(mean_flux)[-layout.n_outputs // 2 :]
        illuminated[telescope] = np.sort(active).tolist()

    modulated = {}
    for baseline, flux in enumerate(baseline_spectra):
        std_flux = flux.std(axis=1).mean(axis=-1)
        active = np.argsort(std_flux)[-4:]
        modulated[baseline] = np.sort(active).tolist()

    baseline_outputs = {}
    for baseline, (telescope_1, telescope_2) in enumerate(layout.baseline_pairs):
        from_flats = set(illuminated[telescope_1]) & set(illuminated[telescope_2])
        from_scan = set(modulated[baseline])
        if from_flats != from_scan:
            raise ValueError(
                f"Output mapping mismatch for baseline {layout.baselines[baseline]}: "
                f"{sorted(from_flats)} from flats, "
                f"{sorted(from_scan)} from modulation"
            )
        baseline_outputs[baseline] = sorted(modulated[baseline])
    return baseline_outputs


def measure_quality(product: P2VM) -> P2VMQC:
    """Measure coherence normalized by the two beams, without changing the fit."""
    values = [None] * len(layout.baselines)  # B43,B42,B41,B32,B31,B21
    for baseline, outputs in product.baseline_outputs.items():
        t1, t2 = layout.baseline_pairs[baseline]
        first = product.transmission[outputs, t1, :]
        second = product.transmission[outputs, t2, :]
        if (first > 0).all() and (second > 0).all():
            ratio = product.coherence_amplitude[outputs, baseline, :] / np.sqrt(first) / np.sqrt(second) / 2
            values[baseline] = float(ratio.mean())
    return P2VMQC(*values)


def diagnostics(product: P2VM) -> dict:
    """Largest wavelength-wise inverse condition number; not a registered ESO QC."""
    return {"p2vm_condition_max": float(np.max(np.linalg.cond(np.moveaxis(product.matrix, -1, 0))))}


def unwrap_scan_phase(phase_rad: NDArray, scan_opd_m: NDArray, wavelength_m: NDArray) -> NDArray:
    """Resolve integer cycles using an OPD guide; preserve measured fine phase.

    Anchor at the middle frame and refine five times using the central half of
    the wavelength grid. No frame/channel rejection, weights or fine-phase
    corrections are introduced by this operation.
    """
    anchor = len(phase_rad) // 2
    delta = phase_rad - phase_rad[anchor]
    opd = scan_opd_m - scan_opd_m[anchor]
    middle = slice(len(wavelength_m) // 4, 3 * len(wavelength_m) // 4)
    for _ in range(5):
        cycles = np.rint((2 * np.pi * opd[:, None] / wavelength_m - delta) / (2 * np.pi))
        corrected = delta + 2 * np.pi * cycles
        opd = np.mean((corrected * wavelength_m / (2 * np.pi))[:, middle], axis=1)
    return corrected + phase_rad[anchor]
