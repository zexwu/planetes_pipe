"""Build DARK/BAD, FLAT, WAVE and P2VM from one raw calibration directory, with a
diagnostic PDF next to each product (<product>.pdf, as plots.report writes them).

Each <step>.run() is the same code as the PyEsoRex recipe planetes_<step> and
returns {product category: written path}. Raw files are found by header type, and
each dark by matching the exposure it calibrates (same window and DIT, same capture
run, closest in time); the steps check that every calibration matches its raw data.
"""
from pathlib import Path

from astropy.io import fits

from planetes_pipe import configure_logging, plots
from planetes_pipe.io.keywords import TPL_START
from planetes_pipe.io.raw import find_dark, find_raw
from planetes_pipe.steps import dark, flat, p2vm, wave

RAW = Path("../../data/2026-11-06/raw/calibration")  # The reference night (workspace README).
OUTPUT = Path("../../data/2026-11-06/example_calibrations/")  # Never overwritten.
# 2026-11-06 has two calibration runs, TPL.START 16:40:23 and 17:15:27; this reduces the
# second, which the science exposures used. The lamp is a run of its own.
RUN = "2026-11-06T17:15:27"


def in_run(paths):
    return [path for path in paths if TPL_START.get(fits.getheader(path)) == RUN]


def main():
    configure_logging()  # The library is silent unless the script asks.
    lamp_raws = find_raw(RAW, "WAVELAMP_RAW")
    flat_raws = in_run(find_raw(RAW, "FLAT_RAW"))  # Four files; identities come from shutters.
    scan_raws = in_run(find_raw(RAW, "P2VM_RAW"))  # Six baseline scans.

    main_dark = dark.run(find_dark(flat_raws[0], RAW), output_dir=OUTPUT)  # Flats' and scans' DIT.
    lamp_dark = dark.run(find_dark(lamp_raws[0], RAW), output_dir=OUTPUT)  # The lamp's DIT.
    # 25 traces, excluding the middle metrology trace.
    flat_out = flat.run(main_dark["DARK"], main_dark["BAD"], flat_raws, output_dir=OUTPUT,
                        metrology_trace=True)
    wave_out = wave.run(lamp_raws[0], lamp_dark["DARK"], flat_out["FLAT"], output_dir=OUTPUT)
    p2vm_out = p2vm.run(main_dark["DARK"], flat_out["FLAT"], wave_out["WAVE"], flat_raws,
                        scan_raws, output_dir=OUTPUT,  # Scans record their OPD sign.
                        gd_search_min_m=-100e-6, gd_search_max_m=100e-6,
                        wave_min_m=1100e-9, wave_max_m=1270e-9,  # Avoids the SuperK pump, ~1064 nm.
                        keep_diagnostics=True)
    for out in (main_dark, lamp_dark, flat_out, wave_out, p2vm_out):
        for product in out.values():
            plots.report(product)  # Portrait A4 PDF; GRAVITY-style quality page for P2VM.
    print(p2vm_out["P2VM"].resolve())


if __name__ == "__main__":
    main()
