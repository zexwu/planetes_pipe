"""The five reduction steps: raw/product files in, product files out.

Each function is one PyEsoRex recipe (planetes_dark, planetes_flat, ...): the
recipes in eso/recipes.py only translate the SOF and CPL parameters into a call
here. Products carry DICD PRO provenance and are never overwritten.

    from planetes_pipe import steps

    cal = steps.dark("dark.fits", output_dir="calibration")
    flat = steps.flat(cal["DARK"], cal["BAD"], flat_raws, output_dir="calibration")
    steps.wave("lamp.fits", lamp_dark, flat["FLAT"], output_dir="calibration")

Each call returns {ESO PRO CATG: written path}. For in-memory work, call the
functions in planetes_pipe.processing directly.
"""

from inspect import signature
from pathlib import Path

from planetes_pipe import io
from planetes_pipe.io.common import CATEGORIES, product_filename
from planetes_pipe.io.provenance import provenance_header
from planetes_pipe.io.raw import read_raw_metadata
from planetes_pipe.model.products import Bad, Dark, Flat, P2VM, P2VMRed, Wave
from planetes_pipe.model.raw import shutter_identity
from planetes_pipe.processing import dark as dark_processing
from planetes_pipe.processing import flat as flat_processing
from planetes_pipe.processing import p2vm as p2vm_processing
from planetes_pipe.processing import reduce as reduce_processing
from planetes_pipe.processing import wave as wave_processing

# Python keyword -> (recipe option name, help). Shared with the CPL recipes.
PARAMETERS = {
    "bad_pixel_threshold": ("bad-dark-threshold",
                            "Positive multiplier in the retained dark bad-pixel tests"),
    "aperture_width_pixels": ("profile-width", "Full extraction aperture in detector pixels"),
    "metrology_trace": ("metrology-trace",
                        "Detect 25 traces and omit the middle metrology trace"),
    "keep_diagnostics": ("keep-diagnostics",
                         "Retain P2VM scan and fit diagnostics in the calibration product"),
    "gd_search_min_m": ("gd-search-min-m",
                        "Coarse GD lower initialization [m]; Newton remains unconstrained"),
    "gd_search_max_m": ("gd-search-max-m",
                        "Coarse GD upper initialization [m]; Newton remains unconstrained"),
    "dl_gain": ("dl-gain", "Delay-line mm to OPD m gain per baseline, e.g. "
                "43:-1,42:-1,41:-1,32:1,31:-1,21:-1; empty disables the DL_SCAN OPD guide"),
}

QUALITY = {
    Dark: dark_processing.measure_quality,
    Bad: dark_processing.measure_quality,
    Flat: flat_processing.measure_quality,
    Wave: wave_processing.measure_quality,
    P2VM: p2vm_processing.measure_quality,
    P2VMRed: reduce_processing.measure_quality,
}


def dark(dark_raw, *, output_dir, bad_pixel_threshold=dark_processing.BAD_PIXEL_THRESHOLD):
    """DARK_RAW -> DARK and BAD."""
    with io.raw.read(dark_raw, category="DARK_RAW") as raw:
        products = dark_processing.make_dark_products(
            raw, bad_pixel_threshold=bad_pixel_threshold
        )
    return _save(dark, products, output_dir, dark_raw, [(dark_raw, "DARK_RAW")],
                 bad_pixel_threshold=bad_pixel_threshold)


def flat(dark, bad, flat_raws, *, output_dir,
         aperture_width_pixels=flat_processing.APERTURE_WIDTH_PIXELS, metrology_trace=False):
    """Four FLAT_RAW (one per telescope shutter) + DARK + BAD -> FLAT."""
    with io.raw.read_many(flat_raws, category="FLAT_RAW") as raws:
        product = flat_processing.make_flat(
            raws, io.dark.read(dark), io.bad.read(bad),
            aperture_width_pixels=aperture_width_pixels, metrology_trace=metrology_trace,
        )
    inputs = [*((path, "FLAT_RAW") for path in flat_raws), (dark, "DARK"), (bad, "BAD")]
    return _save(flat, [product], output_dir, _with_shutters(flat_raws, "FLAT_RAW", "4"),
                 inputs, aperture_width_pixels=aperture_width_pixels,
                 metrology_trace=metrology_trace)


def wave(wavelamp_raw, dark, flat, *, output_dir):
    """WAVELAMP_RAW + its DARK + FLAT -> WAVE."""
    with io.raw.read(wavelamp_raw, category="WAVELAMP_RAW") as raw:
        product = wave_processing.make_wave(raw, io.dark.read(dark), io.flat.read(flat))
    inputs = [(wavelamp_raw, "WAVELAMP_RAW"), (dark, "DARK"), (flat, "FLAT")]
    return _save(wave, [product], output_dir, wavelamp_raw, inputs)


def p2vm(dark, flat, wave, flat_raws, p2vm_raws, *, output_dir,
         phase_reference=None, phase_dark=None, keep_diagnostics=False,
         gd_search_min_m=-30e-6, gd_search_max_m=30e-6, dl_gain=None):
    """Four FLAT_RAW + six P2VM_RAW scans + DARK, FLAT, WAVE -> P2VM.

    phase_reference is an optional all-shutters-open raw exposure; phase_dark is
    its DARK when its exposure differs from the scans. dl_gain maps each baseline
    ('43', ...) to its delay-line mm -> OPD m gain; None ignores DL_SCAN.
    """
    dark_product, flat_product, wave_product = (
        io.dark.read(dark), io.flat.read(flat), io.wave.read(wave)
    )
    inputs = [*((path, "FLAT_RAW") for path in flat_raws),
              *((path, "P2VM_RAW") for path in p2vm_raws),
              (dark, "DARK"), (flat, "FLAT"), (wave, "WAVE")]
    reference = None
    if phase_reference is not None:
        selected = dark_product if phase_dark is None else io.dark.read(phase_dark)
        with io.raw.read(phase_reference, category="SINGLE_SCI_RAW") as raw:
            reference = p2vm_processing.extract_phase_reference(
                raw, selected, flat_product, wave_product
            )
        inputs.append((phase_reference, "PHASE_REFERENCE"))
        if phase_dark is not None:
            inputs.append((phase_dark, "PHASE_DARK_CALIB"))
    opd = None if dl_gain is None else io.p2vm.read_scan_opd(p2vm_raws, dl_gain)
    with (
        io.raw.read_many(flat_raws, category="FLAT_RAW") as flats,
        io.raw.read_many(p2vm_raws, category="P2VM_RAW") as scans,
    ):
        product = p2vm_processing.make_p2vm(
            flat_product, wave_product, dark_product,
            telescope_flats=flats, baseline_cubes=scans,
            phase_reference=reference, scan_opd_m=opd,
            keep_diagnostics=keep_diagnostics,
            gd_search_min_m=gd_search_min_m, gd_search_max_m=gd_search_max_m,
        )
    return _save(p2vm, [product], output_dir, _with_shutters(p2vm_raws, "P2VM_RAW", "43"),
                 inputs, keep_diagnostics=keep_diagnostics, gd_search_min_m=gd_search_min_m,
                 gd_search_max_m=gd_search_max_m, dl_gain=dl_gain)


def reduce(science_raw, dark, flat, wave, p2vm, *, output_dir,
           gd_search_min_m=-100e-6, gd_search_max_m=100e-6):
    """SINGLE_SCI_RAW + DARK, FLAT, WAVE, P2VM -> per-frame SINGLE_SCI_P2VMRED."""
    with io.raw.read(science_raw, category="SINGLE_SCI_RAW") as raw:
        product = reduce_processing.reduce_science(
            raw, io.dark.read(dark), io.flat.read(flat), io.wave.read(wave),
            io.p2vm.read(p2vm),
            gd_search_min_m=gd_search_min_m, gd_search_max_m=gd_search_max_m,
        )
    inputs = [(science_raw, "SINGLE_SCI_RAW"), (dark, "DARK"), (flat, "FLAT"),
              (wave, "WAVE"), (p2vm, "P2VM")]
    return _save(reduce, [product], output_dir, science_raw, inputs,
                 gd_search_min_m=gd_search_min_m, gd_search_max_m=gd_search_max_m)


def _with_shutters(paths, category: str, identity: str) -> Path:
    """The input recorded with the given open shutters, independent of input order."""
    return next(
        Path(path) for path in paths
        if shutter_identity(read_raw_metadata(path, category=category).shutters) == identity
    )


def _save(step, products, output_dir, reference, inputs, **options) -> dict[str, Path]:
    """Write each product with its QC and the provenance of this step call."""
    reference = Path(reference)
    inputs = sorted([(Path(path), tag) for path, tag in inputs],
                    key=lambda item: item[0] != reference)  # Reference raw first.
    defaults = signature(step).parameters
    parameters = [(PARAMETERS[name][0], value, defaults[name].default, PARAMETERS[name][1])
                  for name, value in options.items()]
    written = {}
    for product in products:
        category = CATEGORIES[type(product)]
        path = Path(output_dir) / product_filename(reference, product)
        header = provenance_header(reference, recipe=f"planetes_{step.__name__}",
                                   category=category, inputs=inputs,
                                   parameters=parameters, filename=path.name)
        io.CODECS[type(product)].write(path, product, quality=QUALITY[type(product)](product),
                                       header=header)
        written[category] = path
    return written
