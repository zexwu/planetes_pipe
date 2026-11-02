"""Build DARK/BAD, FLAT, WAVE and P2VM from one selected raw dataset."""
from contextlib import ExitStack
from pathlib import Path

from planetes_pipe import io
from planetes_pipe.io.association import find_calibration_inputs
from planetes_pipe.io.common import product_filename
from planetes_pipe.processing import dark, flat, p2vm, wave

RAW = Path("D:/camera_captures/2026-10-29/raw/calibration")
OUTPUT = Path("calibration")  # Choose a new directory for each run.
DL_GAIN = {"43": -1, "42": -1, "41": -1, "32": 1, "31": -1, "21": -1}


def main():
    dark_path, lamp_dark_path, lamp_path, flats, scans = find_calibration_inputs(RAW)
    OUTPUT.mkdir(parents=True, exist_ok=False)

    # 1. DARK / BAD, with a separate matching dark for the lamp.
    with io.raw.read(dark_path, category="DARK_RAW") as frames:
        dark_cal, bad = dark.make_dark_products(frames)
    io.dark.write(OUTPUT / product_filename(dark_path, dark_cal), dark_cal,
                  quality=dark.measure_quality(dark_cal))
    io.bad.write(OUTPUT / product_filename(dark_path, bad), bad, quality=dark.measure_quality(bad))
    lamp_dark = dark_cal
    if lamp_dark_path != dark_path:
        with io.raw.read(lamp_dark_path, category="DARK_RAW") as frames:
            lamp_dark, _ = dark.make_dark_products(frames)
        io.dark.write(OUTPUT / product_filename(lamp_dark_path, lamp_dark), lamp_dark,
                      quality=dark.measure_quality(lamp_dark))

    # 2. FLAT: four shutters, excluding the middle metrology trace.
    with ExitStack() as files:
        flat_frames = {t: files.enter_context(io.raw.read(path, category="FLAT_RAW"))
                       for t, path in flats.items()}
        flat_cal = flat.make_flat(flat_frames, dark_cal, bad, metrology_trace=True)
    io.flat.write(OUTPUT / product_filename(flats["4"], flat_cal), flat_cal,
                  quality=flat.measure_quality(flat_cal))

    # 3. WAVE: lamp spectrum and its matching dark.
    with io.raw.read(lamp_path, category="WAVELAMP_RAW") as frames:
        wave_cal = wave.make_wave(frames, lamp_dark, flat_cal)
    io.wave.write(OUTPUT / product_filename(lamp_path, wave_cal), wave_cal,
                  quality=wave.measure_quality(wave_cal))

    # 4. P2VM: reopen the consumed flat streams, then read the six scans.
    opd = io.p2vm.read_scan_opd(scans, DL_GAIN) if DL_GAIN is not None else None
    with ExitStack() as files:
        flat_frames = {t: files.enter_context(io.raw.read(path, category="FLAT_RAW"))
                       for t, path in flats.items()}
        scan_frames = {b: files.enter_context(io.raw.read(path, category="P2VM_RAW"))
                       for b, path in scans.items()}
        p2vm_cal = p2vm.make_p2vm(
            flat_cal, wave_cal, dark_cal,
            telescope_flats=flat_frames, baseline_cubes=scan_frames,
            scan_opd_m=opd,
            gd_search_min_m=-100e-6, gd_search_max_m=100e-6,
            keep_diagnostics=True,
        )
    io.p2vm.write(OUTPUT / product_filename(scans["43"], p2vm_cal), p2vm_cal,
                  quality=p2vm.measure_quality(p2vm_cal))
    print(OUTPUT.resolve())


if __name__ == "__main__":
    main()
