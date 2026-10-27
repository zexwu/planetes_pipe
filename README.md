# PLANETES pipeline 0.3

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
pyesorex --recipe-dir=/absolute/path/to/planetes_pipe/planetes/eso --recipes
mkdir /path/to/new-dark-output
cd /path/to/new-dark-output
pyesorex --recipe-dir=/absolute/path/to/planetes_pipe/planetes/eso planetes_dark /absolute/path/to/dark.sof
```

Use a new directory per invocation. Fill the absolute paths in `examples/*.sof.in`
before using them. Tags describe input roles; each accepts exactly one file.

| Recipe | Required SOF roles | Optional roles | Output |
| --- | --- | --- | --- |
| `planetes_dark` | `DARK_RAW` | — | `planetes_dark.fits` |
| `planetes_flat` | `DARK_CALIB`, four `TEL_FLAT_*` | — | `planetes_flat.fits` |
| `planetes_wave` | `DARK_CALIB`, `PLANETES_FLAT`, `WAVE_RAW` | `PLANETES_LINE_LIST` | `planetes_wave.fits` |
| `planetes_p2vm` | `DARK_CALIB`, `PLANETES_FLAT`, `PLANETES_WAVE`, four `TEL_FLAT_*`, six `P2VM_RAW_*` | `PHASE_REFERENCE`, `PHASE_DARK_CALIB` | `planetes_p2vm.fits` |
| `planetes_reduce` | `DARK_CALIB`, `SCIENCE_RAW`, `PLANETES_FLAT`, `PLANETES_WAVE`, `PLANETES_P2VM` | — | `planetes_reduced.fits` |

`DARK_CALIB` is a schema-2 dark product. Flat requires full temporal statistics;
wave/P2VM/reduce also accept a subtraction-only `PLANETES_DARK_IMAGE`. Generate
separate dark products for the main calibration, lamp and science where appropriate.
A supplied phase reference uses `PHASE_DARK_CALIB`, or explicitly records reuse of
the main dark if that role is absent. Without a reference, phase correction is skipped.
When correction is disabled, the unused reference is excluded from DFS used inputs.

Telescope order is 4,3,2,1; baseline order is 43,42,41,32,31,21; the four outputs
within a baseline use A,C,B,D. The wave recipe defaults to the bundled static
30-line FITS reference with the original ECSV values and origin labels.

Parameters use the same defaults and validation in Python and ESO. For example:

```bash
pyesorex --recipe-dir=/absolute/path/to/planetes_pipe/planetes/eso planetes_p2vm \
  --phase-correction=true --wavelength-min-um=1.15 --wavelength-max-um=1.30 \
  --wavelength-channels=151 /absolute/path/to/p2vm.sof
```

See [parameters and FITS contract](docs/PRODUCT_CONTRACT.md) for all settings.
PyEsoRex can save its own `.rc` parameter file; a project YAML is unnecessary.

## Run from Python

```python
from pathlib import Path
from planetes.dark import calibrate_dark_file
from planetes.flat import calibrate_flat_file
from planetes.wavelength import calibrate_wavelength_file
from planetes.p2vm import calibrate_p2vm_file
from planetes.reduction import reduce_file
from planetes.fits import write_product

raw = Path('/absolute/raw')
main_dark = calibrate_dark_file(raw / 'dark.fits')
wave_dark = calibrate_dark_file(raw / 'wave_dark.fits')
science_dark = calibrate_dark_file(raw / 'sci_dark.fits')
telescopes = {n: raw / f'tel{n}.fits' for n in ('4', '3', '2', '1')}
baselines = {n: raw / f'bsl{n}.fits' for n in ('43', '42', '41', '32', '31', '21')}
flat = calibrate_flat_file(telescopes, main_dark)
wave = calibrate_wavelength_file(raw / 'wave.fits', wave_dark, flat)
p2vm = calibrate_p2vm_file(
    flat, wave, main_dark, telescope_flats=telescopes, baseline_cubes=baselines,
    phase_reference=raw / 'sci.fits', phase_dark=science_dark,
)
result = reduce_file(raw / 'sci.fits', science_dark, flat, wave, p2vm)
write_product(Path('/absolute/new-output/planetes_reduced.fits'), result)
```

Save each calibration with the same `write_product` function when needed. Read it
with `read_product(path, ExpectedClass)` from `planetes.fits`. For single frames:

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

| Module | Responsibility |
| --- | --- |
| `models.py` | Scientific objects, units, axes, parent fingerprints |
| `fits.py` | Raw reading, schema-2 encoding, verified non-overwriting publication |
| `dark.py`, `flat.py`, `wavelength.py`, `p2vm.py`, `reduction.py` | File tasks and their ordinary array calculations |
| `extraction.py` | Shared subtraction, extraction and resampling kernels |
| `instrument.py` | Fixed telescope, baseline and output conventions |
| `qc.py`, `plotting.py` | Descriptive QC and explicitly requested diagnostics |
| `eso/` | SOF roles, parameters, task calls and CPL DFS registration |

[Change report and diagrams](docs/CHANGE_REPORT.md) explain the migration and
full-data evidence. [Scientific methods](document.md) describe retained numerical
definitions. `validation/` contains reproducible regression and GUI checks.

```bash
python -m unittest discover -s tests -v
```

Real recipe tests require PyCPL/PyEsoRex and are skipped when unavailable.
Exact comparisons use the same numerical environment and matching Numba cache
states: the old algorithm itself has a small cold/warm difference. Cache and
fastmath settings are retained.

Schema 2 breaks v0.2 FITS/Python APIs. Regenerate calibrations from raw data;
existing products are not automatically converted. Output is instrumental ADU,
visibility amplitude, GD and differential phase. ESO observation metadata,
EDPS association, uncertainty propagation, transfer functions, OIFITS and
Phase 3 acceptance remain separate work. This release does not claim completion
of every requirement in the supplied ESO documents.
