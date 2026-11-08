"""Build DARK/BAD, FLAT, WAVE and P2VM from one raw calibration directory, with a
diagnostic PDF next to each product (<product>.pdf, as plots.report writes them).

Each <step>.run() is the same code as the PyEsoRex recipe planetes_<step> and
returns {product category: written path}. Raw files are found by header type, and
each dark by matching the exposure it calibrates (same window and DIT, same capture
run, closest in time); the steps check that every calibration matches its raw data.
"""
from pathlib import Path

from planetes_pipe import configure_logging, plots
from planetes_pipe.io.raw import find_dark, find_raw
from planetes_pipe.steps import dark, flat, p2vm, wave

RAW = Path("../../captures/2026-11-04/raw/calibration")
OUTPUT = Path("../../captures/2026-11-04/reduced/")  # Existing products are never overwritten.
# OPD sign of each baseline's scan. Scans record it (DL_SCAN OPDGAIN) since 2026-11-05;
# older ones, such as 2026-11-04, need it here or a reversed baseline goes uncorrected.
DL_GAIN = {"43": -1, "42": -1, "41": -1, "32": 1, "31": -1, "21": -1}


def main():
    configure_logging()  # The library is silent unless the script asks.
    lamp_raws = find_raw(RAW, "WAVELAMP_RAW")
    flat_raws = find_raw(RAW, "FLAT_RAW")  # Four files; identities come from shutters.
    scan_raws = find_raw(RAW, "P2VM_RAW")  # Six baseline scans.

    main_dark = dark.run(find_dark(flat_raws[0], RAW), output_dir=OUTPUT)  # Flats' and scans' DIT.
    lamp_dark = dark.run(find_dark(lamp_raws[0], RAW), output_dir=OUTPUT)  # The lamp's DIT.
    # 25 traces, excluding the middle metrology trace.
    flat_out = flat.run(main_dark["DARK"], main_dark["BAD"], flat_raws, output_dir=OUTPUT,
                        metrology_trace=True)
    wave_out = wave.run(lamp_raws[0], lamp_dark["DARK"], flat_out["FLAT"], output_dir=OUTPUT)
    p2vm_out = p2vm.run(main_dark["DARK"], flat_out["FLAT"], wave_out["WAVE"], flat_raws,
                        scan_raws, output_dir=OUTPUT, dl_gain=DL_GAIN,
                        gd_search_min_m=-100e-6, gd_search_max_m=100e-6,
                        wave_min_m=1100e-9, wave_max_m=1270e-9,  # Avoids the SuperK pump, ~1064 nm.
                        keep_diagnostics=True)
    for out in (main_dark, lamp_dark, flat_out, wave_out, p2vm_out):
        for product in out.values():
            plots.report(product)  # Portrait A4 PDF; GRAVITY-style quality page for P2VM.
    print(p2vm_out["P2VM"].resolve())


if __name__ == "__main__":
    main()
