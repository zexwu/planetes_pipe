"""Diagnostic PDFs: one module per product, as in steps/.

    from planetes_pipe import plots
    plots.report("PLANETES.2026-11-04T161749.121_p2vm.fits")   # -> ..._p2vm.pdf next to it

    uv run --locked python -m planetes_pipe.plots reduced/*.fits

Each module (dark, flat, wave, p2vm, science) has pages(product, ...), which yields
matplotlib Figures; pages.write_pdf collects them. report() picks the module from
the product's PRO.CATG. Existing PDFs are never overwritten.
"""
from pathlib import Path

from astropy.io import fits

from planetes_pipe import log
from planetes_pipe.io.keywords import PRO_CATG
from planetes_pipe.plots import dark, flat, p2vm, science, wave
from planetes_pipe.plots.pages import write_pdf
from planetes_pipe.steps.dark import read_bad, read_dark
from planetes_pipe.steps.flat import read_flat
from planetes_pipe.steps.p2vm import read_p2vm
from planetes_pipe.steps.science import read_reduced
from planetes_pipe.steps.wave import read_wave


def report(product_file, output=None) -> Path:
    """Write the diagnostic PDF of one product file; output defaults to <product>.pdf."""
    path = Path(product_file)
    header = fits.getheader(path)
    category = PRO_CATG.get(header)
    if category == "DARK":
        figures = dark.pages(dark=read_dark(path))
    elif category == "BAD":
        figures = dark.pages(bad=read_bad(path))
    elif category == "FLAT":
        figures = flat.pages(read_flat(path))
    elif category == "WAVE":
        figures = wave.pages(read_wave(path))
    elif category == "P2VM":
        figures = p2vm.pages(read_p2vm(path), header)
    elif category == "SINGLE_SCI_P2VMRED":
        figures = science.pages(read_reduced(path), label=path.stem)
    else:
        raise ValueError(f"{path.name}: no report for PRO.CATG {category!r}")
    return write_pdf(Path(output) if output else path.with_suffix(".pdf"), figures)


def save_calibration_plots(dark_product, bad_product, flat_product, wave_product, p2vm_product,
                           output: Path) -> Path:
    """One PDF of a calibration set: DARK and BAD, FLAT, WAVE and P2VM pages in turn."""
    def figures():
        yield from dark.pages(dark=dark_product, bad=bad_product)
        yield from flat.pages(flat_product)
        yield from wave.pages(wave_product)
        yield from p2vm.pages(p2vm_product)
    return write_pdf(output, figures())


def save_reduction_plots(reduced, output: Path, *, object_label: str = "PLANETES",
                         full: bool = False) -> Path:
    """One PDF of per-frame science diagnostics (see science.pages)."""
    log.info(f"--- Step: PLOTTING {object_label} ---")
    path = write_pdf(output, science.pages(reduced, label=object_label, full=full))
    log.info(f"--- Step: PLOTTING {object_label} [DONE] ---")
    return path
