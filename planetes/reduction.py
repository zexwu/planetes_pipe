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
    check_setup,
)
from .io.hdu import as_compute_array
from .io.keywords import RAW_IMAGE
from .io.raw import read_raw_metadata
from .products import Dark, Flat, P2VM, P2VMRed, Wave


def reduce_science(
    path: Path,
    dark: Dark,
    flat: Flat,
    wave: Wave,
    p2vm: P2VM,
    *,
    hdu: int | str = RAW_IMAGE,
    gd_search_min_m: float = -100e-6,
    gd_search_max_m: float = 100e-6,
) -> P2VMRed:
    """Extract all science frames and compute instrumental interferometric observables.

    Parameters
    ----------
    path : Path
        Raw SINGLE_SCI_RAW ADU cube, shaped (frame, y, x).
    dark : Dark
        Mean dark in ADU selected for this science exposure and detector setup.
    flat : Flat
        Detector response and extraction geometry on the science detector grid.
    wave : Wave
        Detector wavelength map in metres, compatible with the selected FLAT.
    p2vm : P2VM
        Instrumental inverse and its saved target wavelength grid in metres.
    hdu : int or str, optional
        Raw image HDU; use 0 explicitly for primary-HDU bench cubes.
    gd_search_min_m, gd_search_max_m : float, optional
        Coarse GD initialization interval in metres, default [-100e-6, 100e-6).
        Subsequent Newton updates are not constrained to this interval.

    Returns
    -------
    reduced : P2VMRed
        Per-frame photometry/coherent flux [ADU], visibility amplitude, invalid
        flags, GD [m] and differential phase [deg], on the saved P2VM grid.

    Notes
    -----
    No temporal averaging, transfer-function calibration or uncertainty propagation
    is performed. All frames are retained; saving is explicit.
    """

    raw = read_raw_metadata(path, category="SINGLE_SCI_RAW", hdu=hdu)
    check_setup(raw.setup, dark.setup, exposure=True, role="DARK")
    check_setup(raw.setup, flat.setup, exposure=False, role="FLAT")
    extractor = prepare_extraction(flat, wave, p2vm.wavelength_m)
    spectra = extract_spectra_from_file(path, extractor, dark, hdu=hdu)
    return reduce_spectra(
        spectra, p2vm, gd_search_min_m=gd_search_min_m, gd_search_max_m=gd_search_max_m
    )


def reduce_spectra(
    spectra: NDArray,
    p2vm: P2VM,
    *,
    gd_search_min_m: float = -100e-6,
    gd_search_max_m: float = 100e-6,
    include_phase: bool = True,
) -> P2VMRed:
    """Apply the P2VM inverse and normalize the reconstructed coherent flux.

    Parameters
    ----------
    spectra : ndarray, shape (24 outputs, frame, wavelength)
        Already dark-subtracted and resampled spectra in ADU, on exactly the
        wavelength grid stored in p2vm. Axis lengths are checked; an array alone
        cannot establish wavelength-grid identity.
    p2vm : P2VM
        Inverse shaped (16 data channels, 24 outputs, wavelength), ordered as
        T4,T3,T2,T1, six real coherent channels, then six imaginary channels.
    gd_search_min_m, gd_search_max_m : float, optional
        Coarse GD initialization interval [min, max) in metres. Newton refinement
        can leave the interval. Ignored when include_phase is False.
    include_phase : bool, optional
        Compute GD and differential phase when True. False returns both as None
        for live previews; these incomplete products cannot be saved as P2VMRED.

    Returns
    -------
    reduced : P2VMRed
        Photometry (frame, 4, wavelength) and complex coherence
        (frame, 6, wavelength) in ADU; dimensionless VISAMP and boolean flags on
        the coherence axes. GD is (frame, 6) in metres; phase is in degrees.

    Notes
    -----
    Baseline order is 43,42,41,32,31,21. VISAMP = abs(C_ij)/sqrt(P_i*P_j).
    Flags describe this normalization's validity, not an S/N or GD-quality cut.
    GD and phase are calculated from all coherent channels without flag masking.
    Differential phase removes GD, rotates by the mean phasor, subtracts the
    arithmetic mean spectral angle and wraps to [-180, 180) degrees.
    """
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
    channels = np.einsum("box,ofx->fbx", p2vm.matrix, spectra)
    flux_adu = channels[:, :N_TELESCOPES, :]
    coherent_adu = channels[:, REAL_SLICE, :] + 1j * channels[:, IMAG_SLICE, :]
    visamp, flags = normalize_coherent_flux(coherent_adu, flux_adu)

    if not include_phase:
        return P2VMRed(
            flux_adu,
            coherent_adu,
            visamp,
            flags,
            p2vm.wavelength_m,
            p2vm.baseline_outputs,
        )

    gd_m = np.zeros((len(coherent_adu), N_BASELINES))
    log.info("Computing group delay for science spectra...")
    for baseline in range(N_BASELINES):
        gd_m[:, baseline] = estimate_group_delay(
            coherent_adu[:, baseline],
            p2vm.wavelength_m,
            search_range=(gd_search_min_m, gd_search_max_m),
        )

    phasor = coherent_adu * np.exp(
        -2j * np.pi * gd_m[:, :, None] / p2vm.wavelength_m[None, :]
    )
    phasor *= phasor.mean(axis=2)[:, :, None].conj()
    phase_deg = np.angle(phasor, deg=True)
    phase_deg -= phase_deg.mean(axis=2)[:, :, None]
    phase_deg = (phase_deg + 180) % 360 - 180

    product = P2VMRed(
        photometric_flux_adu=flux_adu,
        coherent_flux_adu=coherent_adu,
        visibility_amplitude=visamp,
        flags=flags,
        wavelength_m=p2vm.wavelength_m,
        group_delay_m=gd_m,
        differential_phase_deg=phase_deg,
        baseline_outputs=p2vm.baseline_outputs,
    )
    log.info("Reduction for science spectra completed.")

    log.info("--- Step: REDUCE science spectra [DONE] ---")
    return product


def normalize_coherent_flux(
    coherent_flux: NDArray, flux: NDArray
) -> tuple[NDArray, NDArray]:
    """Normalize coherent amplitudes by the fixed baseline photometric products.

    Parameters
    ----------
    coherent_flux : complex ndarray, shape (frame, 6, wavelength)
        Coherent flux in ADU, in baseline order 43,42,41,32,31,21.
    flux : ndarray, shape (frame, 4, wavelength)
        Photometry in ADU, in telescope order T4,T3,T2,T1.

    Returns
    -------
    visamp : ndarray, shape (frame, 6, wavelength)
        abs(coherent_flux) / sqrt(P_i * P_j), or NaN for invalid samples.
    flags : ndarray of bool, shape (frame, 6, wavelength)
        True when the photometric product is nonpositive/nonfinite or the complex
        coherent flux is nonfinite. This tests the product, not each flux separately.

    Notes
    -----
    No clipping at visibility one and no statistical rejection are applied.
    """
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
    """Locate the coherent-sum maximum and refine its optical delay with Newton steps.

    Parameters
    ----------
    visdata : complex ndarray, shape (sample, wavelength)
        Coherent spectra or dimensionless phasors. Each row is fitted independently;
        all wavelength channels enter equally, without masking or noise weights.
    wavelength : ndarray, shape (wavelength,)
        Wavelengths in metres, matching the final axis of visdata.
    search_range : tuple of float, optional
        Coarse delay interval [lower, upper) in metres; default +/-30 micrometres.
        The fixed grid spacing is 1e-8 m. This is an initialization interval,
        not a bound on the returned delay.
    n_newton : int, optional
        Number of unconstrained Newton updates of squared coherent-sum amplitude;
        default 5. A zero Hessian leaves that sample's current delay unchanged.

    Returns
    -------
    group_delay_m : ndarray, shape (sample,)
        Delay maximizing abs(sum(C(lambda) * exp(-2j*pi*GD/lambda))) locally,
        in metres, with this explicit phase/sign convention.

    Notes
    -----
    This estimator does not certify a detected fringe. An all-zero spectrum
    returns the lower search edge, and nonfinite inputs can yield nonfinite
    results. Interpret these values together with signal validity diagnostics.
    """
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
    """Prepare calibrations once for repeated reduction of raw, unbinned frames.

    Parameters
    ----------
    flat : Flat
        Detector response and sparse extraction geometry.
    wave : Wave
        Wavelength map in metres on the same detector grid.
    p2vm : P2VM
        Response inverse and the target wavelength grid used for every frame.

    Notes
    -----
    Construction prepares read-only extraction arrays; it does not load files.
    Pass the selected DARK to each reduce_frame call. Array-only live frames do
    not carry exposure metadata, so the caller must ensure calibration matching.
    """

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
        """Reduce one raw detector frame using the prepared extraction and P2VM.

        Parameters
        ----------
        raw : ndarray, shape (y, x)
            Unbinned ADU image; converted to calculation precision at this boundary.
        dark : Dark
            Selected mean image in ADU on exactly the prepared detector grid.
        include_phase : bool, optional
            False by default for previews. True also computes GD and differential phase.
        gd_search_min_m, gd_search_max_m : float, optional
            Coarse GD interval in metres, used only with include_phase=True.
            Newton refinement may leave these initialization bounds.

        Returns
        -------
        reduced : P2VMRed
            Instrumental observables with a length-one frame axis. GD and differential
            phase are None unless requested. The raw frame is not modified.
        """
        cube = as_compute_array(raw)[None, :, :]
        spectra = self.extractor.extract(cube, dark)
        return reduce_spectra(
            spectra,
            self.p2vm,
            include_phase=include_phase,
            gd_search_min_m=gd_search_min_m,
            gd_search_max_m=gd_search_max_m,
        )
