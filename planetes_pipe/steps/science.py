"""SINGLE_SCI_P2VMRED: per-frame instrumental fluxes, visibility, group delay and phase.

    from planetes_pipe.steps import science
    out = science.run(science_raw, dark, flat, wave, p2vm, output_dir="science")
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from astropy.io import fits as afits
from numpy.typing import NDArray

from planetes_pipe.instrument import layout
from planetes_pipe.io import products
from planetes_pipe.io.keywords import DIT, MJD_OBS
from planetes_pipe.io.raw import Raw, check_calibrations, read_raw, record_saturation
from planetes_pipe.logs import step
from planetes_pipe.steps.dark import Dark, read_dark
from planetes_pipe.steps.extraction import (
    SpectralExtractor,
    estimate_group_delay,
    extract_spectra,
    prepare_extraction,
)
from planetes_pipe.steps.flat import Flat, read_flat
from planetes_pipe.steps.p2vm import P2VM, read_p2vm
from planetes_pipe.steps.wave import Wave, read_wave

log = logging.getLogger(__name__)


@dataclass(slots=True)
class P2VMRed:
    """Per-frame P2VM reduction, NOT an averaged VIS/OIFITS product.

    GD and differential phase may be omitted for live previews; a saved
    P2VMRED product requires both.
    """

    photometric_flux_adu: NDArray  # (frame, telescope, wavelength)
    coherent_flux_adu: NDArray  # complex (frame, baseline, wavelength)
    visibility_amplitude: NDArray  # (frame, baseline, wavelength)
    flags: NDArray  # bool (frame, baseline, wavelength)
    wavelength_m: NDArray
    baseline_outputs: dict[int, list[int]]  # baseline index -> A,C,B,D outputs
    group_delay_m: NDArray | None = None  # (frame, baseline)
    differential_phase_deg: NDArray | None = None  # (frame, baseline, wavelength)


def reduce_science(
    raw: Raw,
    dark: Dark,
    flat: Flat,
    wave: Wave,
    p2vm: P2VM,
    *,
    gd_search_min_m: float = -100e-6,
    gd_search_max_m: float = 100e-6,
) -> P2VMRed:
    """Extract all science frames and compute instrumental interferometric observables.

    Parameters
    ----------
    raw : Raw
        Raw SINGLE_SCI_RAW ADU cube, shaped (frame, y, x).
    dark : Dark
        Mean dark in ADU with the science exposure's detector window and DIT.
    flat : Flat
        Detector response and extraction geometry on the science detector grid.
    wave : Wave
        Detector wavelength map in metres, compatible with the selected FLAT.
    p2vm : P2VM
        Instrumental inverse and its saved target wavelength grid in metres.
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
    is performed. All frames are retained; run() writes the product.
    """

    check_calibrations(raw, dark=dark, flat=flat)
    extractor = prepare_extraction(flat, wave, p2vm.wavelength_m)
    spectra = extract_spectra(raw, extractor, dark)
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
    if (
        p2vm.wavelength_m.ndim != 1
        or p2vm.matrix.shape != (layout.n_data, layout.n_outputs, len(p2vm.wavelength_m))
        or spectra.ndim != 3
        or spectra.shape[0] != layout.n_outputs
        or spectra.shape[-1] != len(p2vm.wavelength_m)
    ):
        raise ValueError("Science spectra and P2VM output/wavelength dimensions differ")
    log.info("applying the P2VM to %d frames", spectra.shape[1])
    channels = np.einsum("box,ofx->fbx", p2vm.matrix, spectra)
    flux_adu = channels[:, :layout.n_telescopes, :]
    coherent_adu = channels[:, layout.real_slice, :] + 1j * channels[:, layout.imag_slice, :]
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

    gd_m = np.zeros((len(coherent_adu), layout.n_baselines))
    for baseline in range(layout.n_baselines):
        gd_m[:, baseline] = estimate_group_delay(
            coherent_adu[:, baseline],
            p2vm.wavelength_m,
            search_range=(gd_search_min_m, gd_search_max_m),
        )
    log.info("median group delay: %s", ", ".join(
        f"{name} {np.median(gd_m[:, b]) * 1e6:+.1f} um" for b, name in enumerate(layout.baselines)))

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

    for baseline, (telescope_1, telescope_2) in enumerate(layout.baseline_pairs):
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
    Construction prepares read-only extraction arrays; it does not load products.
    Pass the selected DARK to each reduce_frame call. NDArray-only live frames do
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
        return layout.n_baselines

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
        cube = np.asarray(raw, dtype=np.float64)[None, :, :]
        spectra = self.extractor.extract(cube, dark)
        return reduce_spectra(
            spectra,
            self.p2vm,
            include_phase=include_phase,
            gd_search_min_m=gd_search_min_m,
            gd_search_max_m=gd_search_max_m,
        )


INSNAME = "PLANETES_SC"


def _time_columns(n_frames: int, n_rows_per_frame: int, start_mjd, dit_s) -> list:
    """TIME [us since the first frame], MJD and INT_TIME [s], as GRAVITY's P2VMRED tables.

    TIME assumes one frame per DIT, so it is an estimate: the camera records no
    per-frame timestamps. Values are NaN when the raw header lacks MJD-OBS or DIT.
    """
    dit = np.nan if dit_s is None else float(dit_s)
    start = np.nan if start_mjd is None else float(start_mjd)
    time_us = np.repeat(np.arange(n_frames) * dit * 1e6, n_rows_per_frame)
    return [products.column("TIME", time_us, "us"),
            products.column("MJD", start + time_us / 86400e6, "d"),
            products.column("INT_TIME", np.full(len(time_us), dit), "s")]


def _oi_table(columns: list, name: str) -> afits.BinTableHDU:
    table = afits.BinTableHDU.from_columns(columns, name=name)
    table.header["INSNAME"] = (INSNAME, "Instrument name")
    return table


def write_reduced(path: Path, product: P2VMRed, *, header=None, start_mjd=None, dit_s=None) -> None:
    """Save per-frame observables (requires GD and differential phase); never overwrite.

    OI_WAVELENGTH, OI_FLUX and OI_VIS are named and arranged as in GRAVITY's P2VMRED,
    but are not valid OIFITS: errors, target and array data are not produced. Table
    rows are frame-major: frame 1 telescopes (or baselines), frame 2, ...
    """
    if product.group_delay_m is None or product.differential_phase_deg is None:
        raise ValueError("Saved P2VMRED requires GD and differential phase")
    nf, nt, nw = product.photometric_flux_adu.shape
    nb = layout.n_baselines
    stations = np.asarray(layout.telescopes, dtype=int)
    pairs = np.asarray([[int(layout.telescopes[t]) for t in pair] for pair in layout.baseline_pairs])
    wavelength = [products.column("EFF_WAVE", product.wavelength_m, "m")]
    flux = [
        products.column("FRAME", np.repeat(np.arange(1, nf + 1), nt)),
        *_time_columns(nf, nt, start_mjd, dit_s),
        products.column("STA_INDEX", np.tile(stations, nf)),
        products.column("FLUX", product.photometric_flux_adu.reshape(-1, nw), "adu"),
    ]
    vis = [
        products.column("FRAME", np.repeat(np.arange(1, nf + 1), nb)),
        *_time_columns(nf, nb, start_mjd, dit_s),
        products.column("STA_INDEX", np.tile(pairs, (nf, 1))),
        products.column("VISDATA", product.coherent_flux_adu.reshape(-1, nw), "adu"),
        products.column("VISAMP", product.visibility_amplitude.reshape(-1, nw)),
        products.column("FLAG", product.flags.reshape(-1, nw)),
        products.column("GDELAY", product.group_delay_m.reshape(-1), "m"),
        products.column("DIFFPHASE", product.differential_phase_deg.reshape(-1, nw), "deg"),
    ]
    extensions = [
        products.detector_table(product.baseline_outputs),
        _oi_table(wavelength, "OI_WAVELENGTH"),
        _oi_table(flux, "OI_FLUX"),
        _oi_table(vis, "OI_VIS"),
    ]
    products.save_product(path, "SINGLE_SCI_P2VMRED", extensions, header=header)


def read_reduced(path: Path) -> P2VMRed:
    """Read a checksummed SINGLE_SCI_P2VMRED; rows are frame-major as written.

    Products written before 0.30 use PLANETES_WAVELENGTH/WAVELENGTH,
    PLANETES_FLUX/FLUXDATA and PLANETES_VIS; they are read as well.
    """
    with products.open_product(path, "SINGLE_SCI_P2VMRED") as hdus:
        new = "OI_VIS" in hdus
        wave_hdu, wave_column = (("OI_WAVELENGTH", "EFF_WAVE") if new
                                 else ("PLANETES_WAVELENGTH", "WAVELENGTH"))
        flux_hdu, flux_column = ("OI_FLUX", "FLUX") if new else ("PLANETES_FLUX", "FLUXDATA")
        vis_hdu = "OI_VIS" if new else "PLANETES_VIS"
        wavelength = products.read_column(hdus, wave_hdu, wave_column)
        nt, nb, nw = layout.n_telescopes, layout.n_baselines, len(wavelength)
        nf = len(hdus[flux_hdu].data) // nt
        flux = products.read_column(hdus, flux_hdu, flux_column).reshape(nf, nt, nw)
        vis = {name: products.read_column(hdus, vis_hdu, name).reshape(nf, nb, -1)
               for name in ("VISDATA", "VISAMP", "FLAG", "GDELAY", "DIFFPHASE")}
        return P2VMRed(
            photometric_flux_adu=flux,
            coherent_flux_adu=vis["VISDATA"],
            visibility_amplitude=vis["VISAMP"],
            flags=vis["FLAG"],
            wavelength_m=wavelength,
            baseline_outputs=products.baseline_outputs(hdus["IMAGING_DETECTOR_SC"].data["REGNAME"]),
            group_delay_m=vis["GDELAY"].reshape(nf, nb),
            differential_phase_deg=vis["DIFFPHASE"],
        )


@step("SCIENCE")
def run(science_file, dark_file, flat_file, wave_file, p2vm_file, *, output_dir,
        gd_search_min_m=-100e-6, gd_search_max_m=100e-6) -> dict[str, Path]:
    """planetes_reduce: one SINGLE_SCI_RAW + DARK, FLAT, WAVE, P2VM -> SINGLE_SCI_P2VMRED."""
    raw = read_raw(science_file, "SINGLE_SCI_RAW")
    dark, flat, wave = read_dark(dark_file), read_flat(flat_file), read_wave(wave_file)
    p2vm = read_p2vm(p2vm_file)
    options = dict(gd_search_min_m=gd_search_min_m, gd_search_max_m=gd_search_max_m)
    product = reduce_science(raw, dark, flat, wave, p2vm, **options)

    inputs = {"SINGLE_SCI_RAW": [science_file], "DARK": [dark_file], "FLAT": [flat_file],
              "WAVE": [wave_file], "P2VM": [p2vm_file]}
    header = products.provenance("planetes_reduce", science_file, inputs, **options)
    record_saturation(header, {"SCI": raw}, flat.extracted_pixels())
    path = products.product_path(output_dir, science_file, "SINGLE_SCI_P2VMRED")
    write_reduced(path, product, header=header, start_mjd=MJD_OBS.get(raw.header),
                  dit_s=DIT.get(raw.header))
    return {"SINGLE_SCI_P2VMRED": path}
