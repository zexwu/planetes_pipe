"""Exercise the actual Qt6 viewer with recorded raw frames, without bench hardware."""

from pathlib import Path
import json
import sys
import tempfile
from time import perf_counter

import numpy as np
from astropy.io import fits


def validate_gui(devel: Path, products: Path, science: Path, output: Path):
    """Run with the GUI's interpreter and QT_QPA_PLATFORM=offscreen; save evidence."""
    from pyqtgraph.Qt import QtWidgets
    from planetes.reduction import FrameReducer, reduce_file
    from planetes.dark import load_dark
    from planetes.fits import iter_raw_cube

    sys.path.insert(0, str(devel))
    from camview_p2vm import FakeCamera, P2VMMainWindow

    output.mkdir(parents=True, exist_ok=False)
    calibration_directory = output / "calibrations"
    calibration_directory.mkdir()
    for name in ("flat", "wave", "p2vm"):
        (calibration_directory / f"planetes_{name}.fits").symlink_to(
            products / name / f"planetes_{name}.fits"
        )
    dark = products / "dark_science/planetes_dark.fits"
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    reducer = FrameReducer.from_directory(calibration_directory)
    window = P2VMMainWindow(
        FakeCamera(science),
        reducer,
        source_label="Recorded-frame validation",
        reducer_dark_path=dark,
    )
    window.refreshTimer.stop()
    window.p2vm_enabled.setChecked(True)
    window.remove_dark = True
    window.remove_dark_button.setChecked(True)
    raw = next(iter_raw_cube(science))[0]
    with tempfile.TemporaryDirectory() as temporary:
        one_frame = Path(temporary) / "one_frame.fits"
        fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(raw[None])]).writeto(one_frame)
        expected = reduce_file(
            one_frame, load_dark(dark), reducer.flat, reducer.wave, reducer.p2vm
        )
        window.image_stream._index = 0
        window._refresh()
        np.testing.assert_array_equal(
            window.latest_visamp, expected.visibility_amplitude[0]
        )
        np.testing.assert_array_equal(window.raw_image, raw)
        window.resize(1100, 850)
        window.show()
        app.processEvents()
        window.grab().save(str(output / "viewer.png"))
        window.tabs.setCurrentIndex(window.visibility_tab_index)
        app.processEvents()
        window.grab().save(str(output / "visibility.png"))
        window.tabs.setCurrentIndex(window.camera_tab_index)
        window.binning_combo.setCurrentIndex(1)
        window.image_stream._index = 0
        window._refresh()
        np.testing.assert_array_equal(
            window.latest_visamp, expected.visibility_amplitude[0]
        )
        assert window.image.shape == (raw.shape[0] // 2, raw.shape[1] // 2)
        window.remove_dark = False
        window._refresh()
        assert window.latest_visamp is None
        window._take_dark()
        assert window.memory_dark.mean_adu.shape == raw.shape
        assert not hasattr(window.memory_dark, "temporal_rms_adu")
        window._refresh()
        assert window.latest_visamp is not None, window.reducer_dir_status.text()
        assert "2D" in window.reducer_dir_status.text()
        window.dark_source_combo.setEditText(str(products / "wave/planetes_wave.fits"))
        window._set_dark_source_from_text()
        window._refresh()
        assert window.latest_visamp is None
        assert "dark" in window.reducer_dir_status.text().lower()
    measured_dark = load_dark(dark)
    reducer.reduce_frame(raw, dark=measured_dark)
    times = []
    for _ in range(128):
        start = perf_counter()
        reducer.reduce_frame(raw, dark=measured_dark)
        times.append((perf_counter() - start) * 1000)
    result = {
        "checks": [
            "raw frame / single dark subtraction",
            "display binning invariance",
            "dark-off pause",
            "current-frame DarkImage",
            "wrong-category rejection",
        ],
        "all_passed": True,
        "frames_timed": len(times),
        "raw_frame_shape": list(raw.shape),
        "mean_ms": float(np.mean(times)),
        "median_ms": float(np.median(times)),
        "p95_ms": float(np.percentile(times, 95)),
        "hardware_connected": False,
    }
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    window.close()
    app.quit()
    return result


if __name__ == "__main__":
    project = Path(__file__).resolve().parents[1]
    # Choose fresh output and the completed recipe chain to validate.
    print(
        validate_gui(
            project.parent / "devel",
            project / "reduced/regression_new_run",
            project / "2026_02_05/sci2/sci.fits",
            project / "reduced/gui_validation_new_run",
        )
    )
