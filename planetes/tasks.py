"""Explicit file tasks shared by Python callers and recipes; science stays array-only."""

from dataclasses import replace
from pathlib import Path

import numpy as np
from astropy.io import fits as afits
from numpy.typing import NDArray

from .calibration.dark import (
    BAD_PIXEL_THRESHOLD,
    compute_badpix,
    compute_dark,
    frame_mean,
)
from .calibration.flat import PROFILE_WIDTH, compute_flat
from .calibration.p2vm import compute_p2vm
from .calibration.wavelength import compute_wavelength
from .extraction import prepare_extraction
from .identity import fingerprint
from .instrument import (
    BASELINES,
    N_OUTPUTS,
    N_TELESCOPES,
    TELESCOPES,
    DetectorSetup,
    check_setup,
)
from .io import keywords as kw
from .io.fits import read_product
from .io.raw import (
    check_raw_calibrations,
    identify_raw,
    iter_raw_cube,
    read_raw_metadata,
)
from .products import (
    BadPixelMap,
    CalibrationSpectra,
    DarkCalibration,
    DarkImage,
    FlatCalibration,
    P2VMCalibration,
    Provenance,
    ReducedObservables,
    WavelengthCalibration,
    WavelengthReference,
    validate,
    validate_gd_range,
    wavelength_grid,
)
from .reduction import FrameReducer, reduce_spectra


def calibrate_dark_file(
    path: Path, *, bad_pixel_threshold: float = BAD_PIXEL_THRESHOLD
) -> tuple[DarkCalibration, BadPixelMap]:
    """Return complete DARK and independent BAD from a declared raw dark stack."""
    from dataclasses import replace

    metadata = read_raw_metadata(path, category="DARK_RAW")
    dark = compute_dark(iter_raw_cube(path))
    dark = replace(
        dark,
        setup=metadata.setup,
        provenance=replace(
            dark.provenance, inputs={"dark_raw": str(Path(path).absolute())}
        ),
    )
    return dark, compute_badpix(dark, threshold=bad_pixel_threshold)


def load_dark(path: Path):
    """GUI loader: complete new DARK, or explicitly selected raw/image subtraction data."""
    from dataclasses import replace

    path = Path(path)
    with afits.open(path, memmap=True) as hdus:
        head = hdus[0].header
        if kw.PRO_CATEGORY in head:
            return read_product(path, DarkCalibration)
        role = str(head.get(kw.DPR_TYPE, "")).upper()
        if role and role != "DARK":
            raise ValueError("Expected dark input, FITS declares another raw category")
        index = next((i for i, h in enumerate(hdus) if h.data is not None), None)
        if index is None:
            raise ValueError("Dark FITS has no data")
        data = hdus[index].data
        unit = str(hdus[index].header.get("BUNIT", "adu")).lower()
        if unit != "adu":
            raise ValueError("Dark subtraction expects ADU")
        if data.ndim == 3 and len(data) >= 2:
            result = compute_dark(
                np.asarray(data[i : i + 16], dtype=np.float32)
                for i in range(0, len(data), 16)
            )
            setup = (
                read_raw_metadata(path).setup
                if kw.FORMAT_KEY in head
                else DetectorSetup(
                    width=data.shape[2], height=data.shape[1], legacy=True
                )
            )
            return replace(
                result,
                setup=setup,
                provenance=replace(
                    result.provenance, inputs={"dark_raw": str(path.absolute())}
                ),
            )
        if data.ndim == 3 and len(data) == 1:
            data = data[0]
        if data.ndim != 2:
            raise ValueError("A GUI dark must be a 2-D image or nonempty cube")
        result = DarkImage(
            np.array(data, dtype=np.float32, copy=True),
            provenance=Provenance(
                producer="planetes.dark.load_image",
                inputs={"dark_image": str(path.absolute())},
            ),
        )
    validate(result)
    return result


def calibrate_flat_file(
    telescope_flats,
    dark: DarkCalibration,
    bad: BadPixelMap,
    *,
    profile_width: float = PROFILE_WIDTH,
) -> FlatCalibration:
    """Resolve four FLAT_RAW shutters and return geometry, without saving products."""
    from dataclasses import replace

    telescope_flats = identify_raw(telescope_flats, "FLAT_RAW")
    unknown = []
    maps = []
    setup = read_raw_metadata(telescope_flats[TELESCOPES[0]]).setup
    for name in TELESCOPES:
        unknown.extend(check_raw_calibrations(telescope_flats[name], dark))

        unknown.extend(
            check_setup(
                read_raw_metadata(telescope_flats[name]).setup,
                setup,
                exposure=False,
                role="flat sequence",
            )
        )
        maps.append(frame_mean(iter_raw_cube(telescope_flats[name])) - dark.mean_adu)
    result = compute_flat(dark, bad, maps, profile_width=profile_width)
    return replace(
        result,
        setup=setup,
        provenance=replace(
            result.provenance,
            inputs={
                f"flat_raw_T{name}": str(telescope_flats[name]) for name in TELESCOPES
            },
            unverified=tuple(sorted(set(unknown))),
        ),
    )


def wavelength_reference_path():
    """Return the installed, format-3 static line-list file."""
    from importlib.resources import files
    from pathlib import Path

    return Path(str(files("planetes").joinpath("data/wavelength_lines.fits")))


def load_wavelength_reference(path=None):
    """Load the packaged or explicitly supplied static reference, including origin labels."""

    return read_product(
        wavelength_reference_path() if path is None else path, WavelengthReference
    )


def calibrate_wavelength_file(
    path, dark, flat, *, reference=None
) -> WavelengthCalibration:
    """Read a lamp cube and return a wavelength solution, without writing files."""
    from dataclasses import replace

    unknown = check_raw_calibrations(path, dark, flat, category="WAVELAMP_RAW")
    validate(dark)
    reference = load_wavelength_reference() if reference is None else reference
    validate(reference)
    image = frame_mean(iter_raw_cube(path))
    if image.shape != dark.mean_adu.shape or image.shape != flat.flat_map_adu.shape:
        raise ValueError("Lamp/dark/flat detector shapes differ")
    image = image - dark.mean_adu
    result = compute_wavelength(image, flat, reference.wavelength_um)
    info = Provenance(
        producer="planetes.wavelength",
        inputs={"wavelamp_raw": str(path)},
        unverified=unknown,
        dependencies={
            "flat": fingerprint(flat),
            "dark": fingerprint(dark),
            "line_list": fingerprint(reference),
        },
    )
    return replace(result, provenance=info)


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
    wavelength_min_um=1.15,
    wavelength_max_um=1.30,
    wavelength_channels=151,
    gd_search_min_um=-30.0,
    gd_search_max_um=30.0,
) -> P2VMCalibration:
    """Run the complete file-level P2VM task on explicitly selected input roles."""
    from dataclasses import replace

    if phase_dark is not None and phase_reference is None:
        raise ValueError("phase_dark requires phase_reference")
    if not isinstance(phase_correction, bool):
        raise ValueError("phase_correction must be bool")
    grid = wavelength_grid(wavelength_min_um, wavelength_max_um, wavelength_channels)
    spectra = extract_calibration_files(
        flat,
        wave,
        main_dark,
        telescope_flats=telescope_flats,
        baseline_cubes=baseline_cubes,
        wavelength_um=grid,
        phase_reference=phase_reference if phase_correction else None,
        phase_dark=phase_dark if phase_correction else None,
    )
    result = compute_p2vm(
        spectra,
        phase_correction=phase_correction,
        gd_search_min_um=gd_search_min_um,
        gd_search_max_um=gd_search_max_um,
    )
    params = {
        **result.provenance.parameters,
        "wavelength_min_um": wavelength_min_um,
        "wavelength_max_um": wavelength_max_um,
        "wavelength_channels": wavelength_channels,
        "phase_dark_reused": bool(
            phase_reference is not None and phase_correction and phase_dark is None
        ),
    }
    return replace(result, provenance=replace(result.provenance, parameters=params))


def extract_calibration_files(
    flat,
    wave,
    main_dark,
    *,
    telescope_flats,
    baseline_cubes,
    wavelength_um,
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
    validate(main_dark)
    if phase_dark is not None and phase_reference is None:
        raise ValueError("phase_dark requires phase_reference")
    model = prepare_extraction(flat, wave, wavelength_um)
    spec_bsl = np.asarray(
        [extract_file(baseline_cubes[name], model, main_dark) for name in BASELINES]
    )
    spec_tel = np.asarray(
        [extract_file(telescope_flats[name], model, main_dark) for name in TELESCOPES]
    )
    phase_reference_spectra = np.empty(
        (N_OUTPUTS, 0, len(model.wavelength_um)), dtype=np.float64
    )
    dependencies = {
        "flat": fingerprint(flat),
        "wave": fingerprint(wave),
        "main_dark": fingerprint(main_dark),
    }
    unknown = []
    for path in [*telescope_flats.values(), *baseline_cubes.values()]:
        unknown.extend(check_raw_calibrations(path, main_dark, flat))
    inputs = {
        **{f"tel_flat_{k}": str(v) for k, v in telescope_flats.items()},
        **{f"baseline_{k}": str(v) for k, v in baseline_cubes.items()},
    }
    if phase_reference is not None:
        read_raw_metadata(phase_reference, category="SINGLE_SCI_RAW")
        selected_dark = main_dark if phase_dark is None else phase_dark
        phase_reference_spectra = extract_file(phase_reference, model, selected_dark)
        unknown.extend(check_raw_calibrations(phase_reference, selected_dark, flat))
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
        model.wavelength_um,
        provenance=Provenance(
            producer="planetes.extraction",
            inputs=inputs,
            dependencies=dependencies,
            unverified=tuple(sorted(set(unknown))),
        ),
    )


def extract_file(path: Path, extractor, dark) -> NDArray:
    """Read and extract a whole file, retaining all frames and the 16-frame chunking."""

    check_raw_calibrations(path, dark, flat_setup=extractor.setup)
    return np.concatenate(
        [extractor.extract(chunk, dark) for chunk in iter_raw_cube(path)], axis=1
    )


def reduce_file(
    path: Path,
    dark,
    flat,
    wave,
    p2vm: P2VMCalibration,
    *,
    gd_search_min_um=-100.0,
    gd_search_max_um=100.0,
) -> ReducedObservables:
    """Read the complete raw science cube, apply explicit calibrations and return results."""

    unknown = check_raw_calibrations(path, dark, flat, category="SINGLE_SCI_RAW")
    validate(dark)
    validate_gd_range((gd_search_min_um, gd_search_max_um))
    reducer = FrameReducer(flat, wave, p2vm)
    spectra = extract_file(path, reducer.extractor, dark)
    result = reduce_spectra(
        spectra,
        p2vm,
        gd_search_min_um=gd_search_min_um,
        gd_search_max_um=gd_search_max_um,
    )
    info = Provenance(
        producer="planetes.reduction",
        unverified=unknown,
        parameters={
            "gd_search_min_um": gd_search_min_um,
            "gd_search_max_um": gd_search_max_um,
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


def load_reducer(directory: Path) -> FrameReducer:
    """Read a calibration directory and prepare one array-only live reducer."""
    directory = Path(directory).expanduser().resolve()
    return FrameReducer(
        read_product(directory / "planetes_flat.fits"),
        read_product(directory / "planetes_wave.fits"),
        read_product(directory / "planetes_p2vm.fits"),
        directory,
    )
