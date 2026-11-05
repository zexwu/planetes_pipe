"""Build DARK/BAD, FLAT, WAVE and P2VM from one raw calibration directory, with a
diagnostic PDF next to each product (<product>.pdf, as plots.report writes them).

Each <step>.run() is the same code as the PyEsoRex recipe planetes_<step> and
returns {product category: written path}. Choose the inputs below; the steps
check that each calibration matches its raw data (DIT, ROI, detector).
"""
from pathlib import Path

from planetes_pipe import configure_logging, plots
from planetes_pipe.io.raw import find_raw
from planetes_pipe.steps import dark, flat, p2vm, wave

RAW = Path("../../captures/2026-11-04/raw/calibration")
DARK_RAW = RAW / "PLANETES.2026-11-04T161628.211.fits"  # Same DIT as flats and scans.
LAMP_DARK_RAW = RAW / "PLANETES.2026-11-04T164022.435.fits"  # Same DIT as the lamp.
OUTPUT = Path("../../captures/2026-11-04/reduced/")  # Existing products are never overwritten.
DL_GAIN = {"43": -1, "42": -1, "41": -1, "32": 1, "31": -1, "21": -1}  # Or None.


def main():
    configure_logging()  # The library is silent unless the script asks.
    lamp_raws = find_raw(RAW, "WAVELAMP_RAW")
    flat_raws = find_raw(RAW, "FLAT_RAW")  # Four files; identities come from shutters.
    scan_raws = find_raw(RAW, "P2VM_RAW")  # Six baseline scans.

    main_dark = dark.run(DARK_RAW, output_dir=OUTPUT)
    lamp_dark = dark.run(LAMP_DARK_RAW, output_dir=OUTPUT)
    # 25 traces, excluding the middle metrology trace.
    flat_out = flat.run(main_dark["DARK"], main_dark["BAD"], flat_raws, output_dir=OUTPUT,
                        metrology_trace=True)
    wave_out = wave.run(lamp_raws[0], lamp_dark["DARK"], flat_out["FLAT"], output_dir=OUTPUT)
    p2vm_out = p2vm.run(main_dark["DARK"], flat_out["FLAT"], wave_out["WAVE"], flat_raws,
                        scan_raws, output_dir=OUTPUT, dl_gain=DL_GAIN,
                        gd_search_min_m=-100e-6, gd_search_max_m=100e-6,
                        keep_diagnostics=True)
    for out in (main_dark, lamp_dark, flat_out, wave_out, p2vm_out):
        for product in out.values():
            plots.report(product)  # Portrait A4 PDF; GRAVITY-style quality page for P2VM.
    print(p2vm_out["P2VM"].resolve())


if __name__ == "__main__":
    main()
