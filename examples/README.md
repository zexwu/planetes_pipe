# Python reduction

Run from the `PLANETES_new` / `PLANETES_py313` workspace root.

**Generate calibrations:** set `RAW` and `OUTPUT` in `reduce_calibrations.py`, then:

```powershell
uv run --locked python planetes_pipe/examples/reduce_calibrations.py
```

The script uses raw header types, shutters and exposure times to select one set.
The example detects 25 traces, excludes metrology and uses recorded DL signs.
P2VM selects the common illuminated wavelength band and native sampling automatically.
Adjust them for your dataset. FITS filenames keep the reference raw timestamps; existing files are never overwritten.

**Reduce science:** set `SCIENCE` and `CALIBRATIONS` in `reduce_science.py`, then:

```powershell
uv run --locked python planetes_pipe/examples/reduce_science.py
```

Use a directory containing one FLAT, WAVE and P2VM, and one dark matching the
science exposure. Products are identified by FITS category, not their filenames.
Existing outputs are never overwritten.

The old 2026-02-05 dataset has missing calibration exposure metadata and uses
its own explicit selections in `../../test_data/reduce.py`.
Both scripts call each step's `run()`, the same code as the optional
[PyEsoRex route](../docs/eso/README.md); SOF files are only needed there.
