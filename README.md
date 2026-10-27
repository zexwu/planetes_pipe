# PLANETES pipeline 0.4.1

PLANETES computes instrumental interferometric observables from detector cubes.
Python, PyEsoRex and the Qt6 bench viewer share the same scientific tasks and
single-frame kernels. File tasks return typed objects; saving is explicit in
Python. Raw data are read only, and product writers refuse existing destinations.

## Install

For scientific functions and FITS I/O:

```bash
python -m pip install -e .
```

For the ESO executor, first install CPL following the
[official ESO installation instructions](https://www.eso.org/sci/software/pycpl/pycpl-site/user/installation.html).
Use a separate environment and an explicit CPL prefix:

```bash
UV_PROJECT_ENVIRONMENT=/path/to/new-venv CPLDIR=/path/to/cpl uv sync --locked --extra eso
```

The local validated executor is `reduced/eso_pilot/runtime/venv/bin/pyesorex`:
Python 3.13, CPL 7.4, PyCPL 1.0.4.post1 and PyEsoRex 1.0.4. The original `.venv`
is retained. Qt and PyCPL are optional; the numerical modules import neither.

## Run with PyEsoRex

```bash
pyesorex --recipe-dir=/absolute/path/to/planetes_pipe/planetes/recipes --recipes
mkdir /path/to/new-dark-output
cd /path/to/new-dark-output
pyesorex --recipe-dir=/absolute/path/to/planetes_pipe/planetes/recipes planetes_dark /absolute/path/to/dark.sof
```

Use a new directory per invocation. Fill the absolute paths in `examples/*.sof.in`
before using them. Tags describe categories; repeated FLAT_RAW/P2VM_RAW entries are
identified by physical shutter metadata, independently of order and filenames.

| Recipe | Required SOF roles | Optional roles | Output |
| --- | --- | --- | --- |
| `planetes_dark` | `DARK_RAW` | — | `planetes_dark.fits` + `planetes_bad.fits` |
| `planetes_flat` | `DARK`, `BAD`, `FLAT_RAW` × 4 | — | `planetes_flat.fits` |
| `planetes_wave` | `DARK`, `FLAT`, `WAVELAMP_RAW` | `LINE_LIST` | `planetes_wave.fits` |
| `planetes_p2vm` | `DARK`, `FLAT`, `WAVE`, `FLAT_RAW` × 4, `P2VM_RAW` × 6 | `PHASE_REFERENCE`, `PHASE_DARK_CALIB` | `planetes_p2vm.fits` |
| `planetes_reduce` | `DARK`, `FLAT`, `WAVE`, `P2VM`, `SINGLE_SCI_RAW` | — | `planetes_reduced.fits` |

All formal file tasks require a complete DARK with measured temporal RMS and at
least two frames. BAD is independent and records its actual parent DARK. Generate
separate darks for main calibration, lamp and science where appropriate. Known
exposure/ROI/detector conflicts fail before subtraction; explicitly legacy unknown
fields are recorded as unverified. DarkImage remains a GUI/single-frame object.
A supplied phase reference keeps its SINGLE_SCI_RAW classification and uses the
independently selected phase DARK, or records reuse of the main DARK. A disabled
reference is excluded from both scientific provenance and DFS used inputs.

Telescope order is 4,3,2,1; baseline order is 43,42,41,32,31,21; the four outputs
within a baseline use A,C,B,D. The wave recipe defaults to the bundled static
30-line FITS reference with the original ECSV values and origin labels.

Parameters use the same defaults and validation in Python and ESO. For example:

```bash
pyesorex --recipe-dir=/absolute/path/to/planetes_pipe/planetes/recipes planetes_p2vm \
  --phase-correction=true --wavelength-min-um=1.15 --wavelength-max-um=1.30 \
  --wavelength-channels=151 /absolute/path/to/p2vm.sof
```

See [parameters and FITS contract](docs/PRODUCT_CONTRACT.md) for all settings.
PyEsoRex can save its own `.rc` parameter file; a project YAML is unnecessary.

## Run from Python

```python
from pathlib import Path
from planetes.tasks import (
    calibrate_dark_file, calibrate_flat_file, calibrate_wavelength_file,
    calibrate_p2vm_file, reduce_file,
)
from planetes.io.fits import write_product

raw = Path('/absolute/raw')
main_dark, bad = calibrate_dark_file(raw / 'dark.fits')
wave_dark, _ = calibrate_dark_file(raw / 'wave_dark.fits')
science_dark, _ = calibrate_dark_file(raw / 'sci_dark.fits')
telescopes = [raw / f'tel{n}.fits' for n in ('4', '3', '2', '1')]
baselines = [raw / f'bsl{n}.fits' for n in ('43', '42', '41', '32', '31', '21')]
flat = calibrate_flat_file(telescopes, main_dark, bad)
wave = calibrate_wavelength_file(raw / 'wave.fits', wave_dark, flat)
p2vm = calibrate_p2vm_file(
    flat, wave, main_dark, telescope_flats=telescopes, baseline_cubes=baselines,
    phase_reference=raw / 'sci.fits', phase_dark=science_dark,
)
result = reduce_file(raw / 'sci.fits', science_dark, flat, wave, p2vm)
write_product(Path('/absolute/new-output/planetes_reduced.fits'), result)
```

Save each calibration with the same `write_product` function when needed. Read it
with `read_product(path, ExpectedClass)` from `planetes.io.fits`. For single frames:

```python
from planetes.reduction import FrameReducer
reducer = FrameReducer(flat, wave, p2vm)
observables = reducer.reduce_frame(raw_frame_adu, dark=science_dark)
```

This returns a length-one frame axis, instrumental fluxes, visibility amplitude
and invalid flags. Set `diagnostics=True` for GD and differential phase. Science
and live extraction always use the P2VM's saved wavelength grid; insufficient
coverage or inconsistent flat/wave parents fails before processing.

The actual Qt6 viewer integration, recovery instructions and fake-data launch
are documented in [GUI integration](docs/GUI_INTEGRATION.md).

## Organization and validation

| Location | Responsibility |
| --- | --- |
| `calibration/` | Array-only dark, flat, wavelength and P2VM calculations |
| `extraction.py`, `reduction.py` | Array-only extraction/resampling, instrumental observables and FrameReducer |
| `tasks.py` | File tasks used by Python and recipes; load_reducer/load_dark for GUI setup |
| `products.py` | Named calibration/results, array units/axes and scientific validation |
| `instrument.py` | Fixed telescope/baseline/output conventions and physical DetectorSetup |
| `identity.py` | Scientific fingerprints and private immutable canonical encoding |
| `qc.py`, `plotting.py` | Numerical quality measurements and scientific plots |
| `io/raw.py` | Classified raw FITS and acquisition metadata |
| `io/fits.py` | Product FITS encoder/decoder |
| `io/keywords.py` | FITS categories/extensions and scientific QC -> keyword mapping |
| `io/hdu.py`, `io/publication.py` | Shared FITS primitives and verified non-overwriting publication |
| `recipes/planetes_*.py` | Five discoverable PyEsoRex entry points |
| `recipes/parameters.py`, `recipes/dfs.py` | CPL parameter translation, frames and DFS; all CPL imports stay here |

Start with `tasks.py` to follow a complete file operation. For a numerical method,
open the corresponding calibration module or extraction/reduction. For a FITS
field, open `io/keywords.py` and `io/fits.py`. A recipe should only select inputs,
translate parameters, call the shared task and register the result.

[Change report and diagrams](docs/CHANGE_REPORT.md) explain the migration and
full-data evidence. [Scientific methods](docs/SCIENTIFIC_METHODS.md) describe retained numerical
definitions. Maintained tests are grouped into scientific calculations, calibration
relationships, FITS I/O and optional ESO integration. One-off migration scripts
and logs are kept in Git history and the local archive referenced in the report.

```bash
python -m unittest discover -s tests -v
```

Real recipe tests require PyCPL/PyEsoRex and are skipped when unavailable.
Exact comparisons use the same numerical environment and matching Numba cache
states: the old algorithm itself has a small cold/warm difference. Cache and
fastmath settings are retained.

Format 3 breaks v0.3/schema-2 FITS and Python APIs. Regenerate products from
standardized raw copies; historical raw are never rewritten by the reader.
[The product contract](docs/PRODUCT_CONTRACT.md) describes the strict new-acquisition
metadata and the explicit legacy policy. [QC definitions](docs/QC_PARAMETERS.md)
map applicable measurements to GRAVITY **1.9.6 source definitions**, with units,
statistical domains and unavailable reasons. Necessary PLANETES fields are named
explicitly; this does not make PLANETES output a GRAVITY-compatible product.

Outputs remain instrumental ADU, visibility amplitude, GD and differential phase.
ESO dictionary approval, EDPS association, uncertainties, transfer functions,
final OIFITS and Phase 3 acceptance remain subsequent work.

Version 0.4.1 reorganizes Python imports and names without changing format 3 or
scientific results; existing 0.4 format-3 products remain readable. Use file tasks
from `planetes.tasks`, raw I/O from `planetes.io.raw`, and product I/O from
`planetes.io.fits`. Old `planetes.eso`, `planetes.fits`, `planetes.models` and
root-level calibration imports are removed. The clearer numerical names are
`compute_wavelength`, `resample_spectra`, and `reduce_spectra`; the Python dark-task
argument is `bad_pixel_threshold`. CPL names/aliases and scientific defaults stay
unchanged. The private format-3 BAD provenance key `bad_threshold` is retained to
preserve existing scientific fingerprints.
