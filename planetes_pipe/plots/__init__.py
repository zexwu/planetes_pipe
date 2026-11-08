"""Diagnostic PDFs: one module per product, as in steps/; portrait A4, as GRAVITY's reports.

    from planetes_pipe import plots
    plots.report("PLANETES.2026-11-04T161749.121_p2vm.fits")   # -> ..._p2vm.pdf next to it

    uv run --locked python -m planetes_pipe.plots reduced/*.fits

Each module (dark, flat, wave, p2vm, science) has sections(product, ...), a list of
titled blocks; pages.write_pdf flows them onto portrait A4 pages. report() picks
the module from the product's PRO.CATG and starts with the product's identity
from its header. Existing PDFs are never overwritten.
"""
from pathlib import Path

from astropy.io import fits

from planetes_pipe import __version__
from planetes_pipe.io.keywords import DATE, DIT, PRO_CATG, PRO_PIPELINE, PRO_RAW_NAME, PRO_RECIPE
from planetes_pipe.plots import dark, flat, p2vm, science, wave
from planetes_pipe.logs import running
from planetes_pipe.plots.pages import summary_section, write_pdf
from planetes_pipe.steps.dark import read_bad, read_dark
from planetes_pipe.steps.flat import read_flat
from planetes_pipe.steps.p2vm import read_p2vm
from planetes_pipe.steps.science import read_reduced
from planetes_pipe.steps.wave import read_wave

TITLES = {"DARK": "PLANETES DARK Report", "BAD": "PLANETES BAD Report",
          "FLAT": "PLANETES FLAT Report", "WAVE": "PLANETES WAVE Report",
          "P2VM": "PLANETES P2VM Quality Control Report",
          "SINGLE_SCI_P2VMRED": "PLANETES Science P2VMRED Report"}


def _identity(header) -> list:
    """Rows of the product summary, from the header's DICD keywords."""
    raws = []
    for n in range(1, 100):
        name = PRO_RAW_NAME.at(n).get(header)
        if name is None:
            break
        raws.append(name)
    rows = [("Category", PRO_CATG.get(header)), ("Recipe", PRO_RECIPE.get(header)),
            ("Pipeline", PRO_PIPELINE.get(header)), ("Written", DATE.get(header)),
            ("DIT [s]", DIT.get(header)), ("Raw inputs", len(raws))]
    if raws:
        rows += [("First raw", raws[0]), ("Last raw", raws[-1])]
    return [(label, value) for label, value in rows if value is not None]


def report(product_file, output=None) -> Path:
    """Write the diagnostic PDF of one product file; output defaults to <product>.pdf."""
    with running("REPORT", "plots.report"):
        return _report(Path(product_file), output)


def _report(path: Path, output) -> Path:
    header = fits.getheader(path)
    category = PRO_CATG.get(header)
    if category == "DARK":
        body = dark.sections(dark=read_dark(path))
    elif category == "BAD":
        body = dark.sections(bad=read_bad(path))
    elif category == "FLAT":
        body = flat.sections(read_flat(path))
    elif category == "WAVE":
        body = wave.sections(read_wave(path), header)
    elif category == "P2VM":
        body = p2vm.sections(read_p2vm(path), header)
    elif category == "SINGLE_SCI_P2VMRED":
        body = science.sections(read_reduced(path))
    else:
        raise ValueError(f"{path.name}: no report for PRO.CATG {category!r}")
    sections = [summary_section(path, header, _identity(header)), *body]
    return write_pdf(Path(output) if output else path.with_suffix(".pdf"), TITLES[category],
                     sections, footer=f"{path.name}    planetes_pipe {__version__}")


def save_calibration_plots(dark_product, bad_product, flat_product, wave_product, p2vm_product,
                           output: Path) -> Path:
    """One PDF of a calibration set: DARK and BAD, FLAT, WAVE and P2VM sections in turn."""
    sections = [*dark.sections(dark=dark_product, bad=bad_product), *flat.sections(flat_product),
                *wave.sections(wave_product), *p2vm.sections(p2vm_product)]
    return write_pdf(output, "PLANETES Calibration Report", sections,
                     footer=f"{Path(output).name}    planetes_pipe {__version__}")


def save_reduction_plots(reduced, output: Path, *, object_label: str = "PLANETES",
                         full: bool = False) -> Path:
    """One PDF of per-frame science diagnostics (see science.sections)."""
    with running("REPORT", "plots.save_reduction_plots"):
        return write_pdf(output, f"{object_label} — Science P2VMRED Report",
                         science.sections(reduced, full=full),
                         footer=f"{Path(output).name}    planetes_pipe {__version__}")
