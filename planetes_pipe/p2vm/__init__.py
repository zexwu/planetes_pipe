"""P2VM: instrumental pixel-to-visibility response from telescope flats and baseline scans.

This file is the step (spectra, wavelength grid, phase reference, QC, FITS, run());
the matrix fit itself is in fit.py.

    from planetes_pipe import p2vm
    out = p2vm.run(dark, flat, wave, flat_raws, scan_raws, output_dir="calibration")
"""

from collections.abc import Mapping
from pathlib import Path

import numpy as np
from astropy.io import fits as afits
from numpy.typing import NDArray

from planetes_pipe import files
from planetes_pipe.dark import Dark, read_dark
from planetes_pipe.extraction import extract_spectra, prepare_extraction
from planetes_pipe.flat import Flat, read_flat
from planetes_pipe.instrument import Raw, check_setup, layout, order_raws, shutter_identity
from planetes_pipe.p2vm.fit import P2VM, P2VMSpectra, fit_p2vm
from planetes_pipe.wave import Wave, read_wave

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


def extract_phase_reference(raw: Raw, dark: Dark, flat: Flat, wave: Wave) -> NDArray:
    """Extract an instrumental phase-reference exposure onto the P2VM target grid.

    Parameters
    ----------
    raw : Raw
        Phase-reference ADU cube with all shutters open.
    dark : Dark
        Mean dark in ADU selected for this exposure; its own DARK when the
        exposure differs from the calibration scans.
    flat, wave : Flat, Wave
        The same calibrations later passed to make_p2vm.

    Returns
    -------
    spectra : ndarray, shape (24 outputs, frame, wavelength)
        Dark-subtracted ADU spectra, ready for make_p2vm(phase_reference=...).
    """
    check_setup(raw.setup, dark.setup, exposure=True, role="DARK")
    check_setup(raw.setup, flat.setup, exposure=False, role="FLAT")
    extractor = prepare_extraction(flat, wave, wavelength_grid(flat, wave))
    return extract_spectra(raw, extractor, dark)


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
        grid, from extract_phase_reference() with its own DARK. None skips the
        correction.
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
        Response terms and inverse on the target grid. Not written to disk.

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


def qc(product: P2VM) -> dict:
    """Mean coherence normalized by the two beams, per baseline; the fit is unchanged."""
    values = {}
    for baseline, outputs in product.baseline_outputs.items():
        t1, t2 = layout.baseline_pairs[baseline]
        first = product.transmission[outputs, t1, :]
        second = product.transmission[outputs, t2, :]
        if (first > 0).all() and (second > 0).all():
            ratio = (product.coherence_amplitude[outputs, baseline, :]
                     / np.sqrt(first) / np.sqrt(second) / 2)
            key = f"ESO QC P2VM_COHERENCE_SC{layout.baselines[baseline]}"
            values[key] = float(ratio.mean())
    return values


# Optional image extensions: field, extension name, unit.
_IMAGES = [
    ("matrix", "PLANETES_P2VM_INVERSE", ""),
    ("opd_m", "PLANETES_OPD", "m"),
    ("group_delay_m", "PLANETES_GD", "m"),
    ("ellipse_phase_rad", "PLANETES_ELLIPSE_PHASE", "rad"),
    ("ellipse_amplitude", "PLANETES_ELLIPSE_AMPLITUDE", ""),
    ("fit_rms_adu", "PLANETES_P2VM_FIT_RMS", "adu"),
]


def write_p2vm(path: Path, product: P2VM, *, header=None) -> None:
    """Save a P2VM with its QC; never overwrite."""
    detector = files.detector_table(product.baseline_outputs)
    columns = [
        files.column("REGNAME", detector.data["REGNAME"]),
        files.column("TRANSMISSION", product.transmission),
        files.column("COHERENCE", product.coherence_amplitude),
        files.column("PHASE", product.coherence_phase_rad, "rad"),
    ]
    if product.dc_term_adu is not None:
        columns.append(files.column("C_MATRIX", product.dc_term_adu, "adu"))
    extensions = [
        detector,
        files.table("PLANETES_WAVELENGTH", [files.column("WAVELENGTH", product.wavelength_m, "m")]),
        files.table("P2VM_SC", columns),
    ]
    extensions += [files.image_hdu(ext, getattr(product, name), unit)
                   for name, ext, unit in _IMAGES if getattr(product, name) is not None]
    files.save_product(path, "P2VM", extensions, header=header, qc=qc(product))


def read_p2vm(path: Path) -> P2VM:
    """Read a checksummed P2VM at computation precision."""
    with files.open_product(path, "P2VM") as hdus:
        labels = hdus["IMAGING_DETECTOR_SC"].data["REGNAME"]
        values = {
            "wavelength_m": files.read_column(hdus, "PLANETES_WAVELENGTH", "WAVELENGTH"),
            "baseline_outputs": files.baseline_outputs(labels),
            "transmission": files.read_column(hdus, "P2VM_SC", "TRANSMISSION"),
            "coherence_amplitude": files.read_column(hdus, "P2VM_SC", "COHERENCE"),
            "coherence_phase_rad": files.read_column(hdus, "P2VM_SC", "PHASE"),
        }
        if "C_MATRIX" in hdus["P2VM_SC"].columns.names:
            values["dc_term_adu"] = files.read_column(hdus, "P2VM_SC", "C_MATRIX")
        for name, ext, unit in _IMAGES:
            if ext in hdus:
                values[name] = files.read_image(hdus, ext)
        return P2VM(**values)


def read_scan_opd(
    baseline_cubes: list[Path], dl_gain: Mapping[str, float]
) -> NDArray:
    """Read per-frame DL_SCAN positions and explicitly convert mechanical mm to OPD m."""
    if set(dl_gain) != set(layout.baselines):
        raise ValueError("dl_gain requires all six physical baselines")
    paths = {shutter_identity(files.read_shutters(afits.getheader(path))): path
             for path in baseline_cubes}
    positions = []
    for baseline in layout.baselines:
        table, header = afits.getdata(paths[baseline], "DL_SCAN", header=True)
        if header["MOVINGDL"] != int(baseline[1]):
            raise ValueError(f"DL gain assumes the second arm moves: {baseline}")
        counts = table["NFRAMES"]
        if not np.array_equal(table["FIRST_FRAME"], np.cumsum(counts) - counts):
            raise ValueError(f"Noncontiguous DL_SCAN frames: {baseline}")
        position = np.repeat((table["BEFORE_MM"] + table["AFTER_MM"]) / 2, counts)
        positions.append(position * 1e-3 * dl_gain[baseline])
    return np.asarray(positions)


def run(dark, flat, wave, flat_raws, p2vm_raws, *, output_dir,
        phase_reference=None, phase_dark=None, keep_diagnostics=False,
        gd_search_min_m=-30e-6, gd_search_max_m=30e-6, dl_gain=None) -> dict[str, Path]:
    """Recipe planetes_p2vm: 4 FLAT_RAW + 6 P2VM_RAW scans + DARK, FLAT, WAVE -> P2VM.

    phase_reference is an optional all-shutters-open raw exposure; phase_dark is
    its DARK when its exposure differs from the scans. dl_gain maps each baseline
    ('43', ...) to its delay-line mm -> OPD m gain; None ignores DL_SCAN.
    """
    dark_product, flat_product, wave_product = read_dark(dark), read_flat(flat), read_wave(wave)
    inputs = {"FLAT_RAW": flat_raws, "P2VM_RAW": p2vm_raws,
              "DARK": [dark], "FLAT": [flat], "WAVE": [wave]}
    reference_spectra = None
    if phase_reference is not None:
        selected = dark_product if phase_dark is None else read_dark(phase_dark)
        reference_spectra = extract_phase_reference(
            files.read_raw(phase_reference, "SINGLE_SCI_RAW"), selected, flat_product,
            wave_product)
        inputs["PHASE_REFERENCE"] = [phase_reference]
        if phase_dark is not None:
            inputs["PHASE_DARK_CALIB"] = [phase_dark]
    scans = [files.read_raw(path, "P2VM_RAW") for path in p2vm_raws]
    product = make_p2vm(
        flat_product, wave_product, dark_product,
        telescope_flats=[files.read_raw(path, "FLAT_RAW") for path in flat_raws],
        baseline_cubes=scans,
        phase_reference=reference_spectra,
        scan_opd_m=None if dl_gain is None else read_scan_opd(p2vm_raws, dl_gain),
        keep_diagnostics=keep_diagnostics,
        gd_search_min_m=gd_search_min_m, gd_search_max_m=gd_search_max_m,
    )
    # Name and inherit the header from the 43 scan, independent of input order.
    reference = next(p for p, raw in zip(p2vm_raws, scans) if shutter_identity(raw.shutters) == "43")
    header = files.provenance("planetes_p2vm", reference, inputs,
                             keep_diagnostics=keep_diagnostics, gd_search_min_m=gd_search_min_m,
                             gd_search_max_m=gd_search_max_m, dl_gain=dl_gain)
    out = files.output_paths(output_dir, reference, "P2VM")
    write_p2vm(out["P2VM"], product, header=header)
    return out
