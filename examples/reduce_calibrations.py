"""Build DARK/BAD, FLAT, WAVE and P2VM from one selected raw dataset.

Each <step>.run() is the same code as the PyEsoRex recipe planetes_<step> and
returns {product category: written path}.
"""
from pathlib import Path

from planetes_pipe import dark, flat, p2vm, wave
from planetes_pipe.association import find_calibration_inputs

RAW = Path("D:/camera_captures/2026-10-29/raw/calibration")
OUTPUT = Path("calibration")  # Existing products are never overwritten.
DL_GAIN = {"43": -1, "42": -1, "41": -1, "32": 1, "31": -1, "21": -1}  # Or None.


def main():
    dark_raw, lamp_dark_raw, lamp_raw, flat_raws, scan_raws = find_calibration_inputs(RAW)

    # 1. DARK / BAD, with a separate matching dark for the lamp.
    main_dark = dark.run(dark_raw, output_dir=OUTPUT)
    lamp_dark = main_dark
    if lamp_dark_raw != dark_raw:
        lamp_dark = dark.run(lamp_dark_raw, output_dir=OUTPUT)

    # 2. FLAT: four shutters, excluding the middle metrology trace.
    flat_out = flat.run(main_dark["DARK"], main_dark["BAD"], flat_raws, output_dir=OUTPUT,
                        metrology_trace=True)

    # 3. WAVE: lamp spectrum and its matching dark.
    wave_out = wave.run(lamp_raw, lamp_dark["DARK"], flat_out["FLAT"], output_dir=OUTPUT)

    # 4. P2VM: telescope flats and the six baseline scans.
    p2vm_out = p2vm.run(main_dark["DARK"], flat_out["FLAT"], wave_out["WAVE"], flat_raws,
                        scan_raws, output_dir=OUTPUT, dl_gain=DL_GAIN,
                        gd_search_min_m=-100e-6, gd_search_max_m=100e-6,
                        keep_diagnostics=True)
    print(p2vm_out["P2VM"].resolve())


if __name__ == "__main__":
    main()
