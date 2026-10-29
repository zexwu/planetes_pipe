# PLANETES pipeline 0.13.0

PLANETES computes instrumental interferometric observables from detector cubes.
Python and PyEsoRex share the same scientific steps and single-frame kernels.
File entry points return typed objects; saving is explicit in Python. Raw data are read only, and product writers refuse existing destinations.
Qt6 applications are maintained separately; this repository contains no GUI patches.

P2VM calculation: [flow diagram](docs/p2vm_flow.png) ·
[phase and normalization conventions](docs/SCIENTIFIC_METHODS.md#p2vm-p2vmpy) ·
[change report](docs/CHANGE_REPORT.md).

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
SOF categories are unchanged. For bench files without shutter readback, use the
explicit Python identity mappings below.

| Recipe | Required SOF roles | Optional roles | Output |
| --- | --- | --- | --- |
| `planetes_dark` | `DARK_RAW` | — | `planetes_dark.fits` + `planetes_bad.fits` |
| `planetes_flat` | `DARK`, `BAD`, `FLAT_RAW` × 4 | — | `planetes_flat.fits` |
| `planetes_wave` | `DARK`, `FLAT`, `WAVELAMP_RAW` | `LINE_LIST` | `planetes_wave.fits` |
| `planetes_p2vm` | `DARK`, `FLAT`, `WAVE`, `FLAT_RAW` × 4, `P2VM_RAW` × 6 | `PHASE_REFERENCE`, `PHASE_DARK_CALIB` | `planetes_p2vm.fits` |
| `planetes_reduce` | `DARK`, `FLAT`, `WAVE`, `P2VM`, `SINGLE_SCI_RAW` | — | `planetes_reduced.fits` |

`Dark` always contains `mean_adu`; `temporal_rms_adu` and `n_frames` are None
when unknown. One measured frame has `n_frames=1` and no measured RMS; a standalone
2-D dark image has unknown frame count. Python subtraction/reduction only needs the
mean image. `detect_bad_pixels` requires temporal RMS, and `make_dark_products`
therefore needs a multi-frame stack to return both Dark and Bad. Existing complete
DARK FITS products require RMS and a measured frame count (at least two).
Select appropriate darks for main calibration, lamp and science. Known exposure,
ROI and detector contradictions fail before subtraction; missing fields stay unknown.
A supplied phase reference keeps its SINGLE_SCI_RAW classification and uses the
independently selected phase DARK, or records reuse of the main DARK. A disabled
reference is excluded from the calculation and DFS used inputs.

Telescope order is 4,3,2,1; baseline order is 43,42,41,32,31,21; the four outputs
within a baseline use A,C,B,D. The wave recipe defaults to the bundled static
30-line FITS reference with the original ECSV values and origin labels.

Parameters use the same scientific defaults in Python and ESO. For example:

```bash
pyesorex --recipe-dir=/absolute/path/to/planetes_pipe/planetes/recipes planetes_p2vm \
  --phase-correction=true --wavelength-min-m=1.15e-6 --wavelength-max-m=1.30e-6 \
  --wavelength-channels=151 /absolute/path/to/p2vm.sof
```

See [parameters and FITS contract](docs/PRODUCT_CONTRACT.md) for all settings.
PyEsoRex can save its own `.rc` parameter file; a project YAML is unnecessary.

## Run from Python

[examples/reduce.py](examples/reduce.py) is a complete linear script: edit its raw
and output directories, then run it. It explicitly selects separate main, lamp
and science DARKs and saves eight calibration/instrumental products. Its P2VM call
has no phase reference. The example below shows how to select and extract a
reference with its matching DARK when available.

```python
from pathlib import Path
from planetes.dark import make_dark_products
from planetes.flat import make_flat
from planetes.wavelength import make_wave, wavelength_reference_path
from planetes.p2vm import make_p2vm
from planetes.reduction import reduce_science
from planetes.io.fits import read_product, write_product
from planetes.products import WavelengthReference

raw = Path('/absolute/raw')
main_dark, bad = make_dark_products(raw / 'dark.fits')
wave_dark, _ = make_dark_products(raw / 'wave_dark.fits')
science_dark, _ = make_dark_products(raw / 'sci_dark.fits')
telescopes = {n: raw / f'tel{n}.fits' for n in ('4', '3', '2', '1')}
baselines = {n: raw / f'bsl{n}.fits' for n in ('43', '42', '41', '32', '31', '21')}
flat = make_flat(telescopes, main_dark, bad)
lines = read_product(wavelength_reference_path(), WavelengthReference)
wave = make_wave(raw / 'wave.fits', wave_dark, flat, reference=lines)
p2vm = make_p2vm(
    flat, wave, main_dark, telescope_flats=telescopes, baseline_cubes=baselines,
)
p2vmred = reduce_science(raw / 'sci.fits', science_dark, flat, wave, p2vm)
write_product(Path('/absolute/new-output/planetes_reduced.fits'), p2vmred)
```

`make_wave` always receives an explicit line list. The WAVE recipe selects the
packaged list when no LINE_LIST is supplied and records it as a DFS input.

For instrumental phase correction, select the reference file and DARK in the
calling script, then extract the reference on the same grid as the scans:

```python
import numpy as np
from planetes.extraction import extract_spectra_from_file, prepare_extraction
from planetes.io.raw import check_raw_calibrations, read_raw_metadata
from planetes.p2vm import extract_p2vm_spectra, fit_p2vm

grid = np.linspace(1.15e-6, 1.30e-6, 151)
scans = extract_p2vm_spectra(
    flat, wave, main_dark, telescope_flats=telescopes,
    baseline_cubes=baselines, wavelength_m=grid,
)
reference_path = raw / 'phase_reference.fits'
reference_dark = science_dark  # Explicitly choose the dark for this exposure.
metadata = read_raw_metadata(reference_path, category='SINGLE_SCI_RAW')
check_raw_calibrations(metadata, reference_dark, flat)
extractor = prepare_extraction(flat, wave, grid)
reference = extract_spectra_from_file(reference_path, extractor, reference_dark)
p2vm = fit_p2vm(scans, phase_reference=reference)
```

`phase_reference=None` means no correction. Both `fit_p2vm` and `make_p2vm`
accept extracted reference spectra `(output, frame, wavelength)`, never a path.
The recipe's `--phase-correction` remains a file-selection option; the numerical
functions have no such switch or implicit DARK fallback.

Unversioned bench FITS may store their cube in the primary or an image extension
and omit unfinished acquisition keywords. Known exposure, ROI and shutter
contradictions still fail; missing metadata stays unknown. The dictionaries above
state physical identities explicitly: filenames and list order never assign them.

The reader selects `IMAGING_DATA_SC` when present, otherwise the unique 3-D image.
Multiple unnamed cubes require `hdu=...` in the low-level reader. A known non-ADU
unit is rejected; missing BUNIT follows the bench ADU convention. `read_raw_header()`
returns the unchanged primary header. `RawMetadata` contains only category, four
shutter states (each may be None), and detector setup. Format-3 files remain
readable; only relevant physical fields are read from their private metadata.
Acquisition uses `CaptureMetadata`, `build_raw_header` and `write_raw` from
`planetes.io.acquisition`; host timing and configuration sources stay there.

Save each calibration with the same `write_product` function when needed. Read it
with `read_product(path, ExpectedClass)` from `planetes.io.fits`. QC is retained by
default; `write_product(path, product, include_qc=False)` saves without computing
QC. Formal recipes explicitly retain QC. The FITS encoder only serializes supplied
measurements; `qc.measure_quality(product)` returns a dictionary of available
statistics. Unknown values are omitted, without a separate missing-status table.
Writers generate checksums and publish through temporary files without overwriting
existing paths; they do not automatically read back and decode their own outputs.
Product readers still check checksums, units and physical identities. FITS
round-trip comparisons belong to the existing development tests.
See [the retained QC set](docs/QC_PARAMETERS.md).
For single frames:

```python
from planetes.reduction import FrameReducer
reducer = FrameReducer(flat, wave, p2vm)
observables = reducer.reduce_frame(raw_frame_adu, dark=science_dark)
```

This returns a `P2VMRed` with a length-one frame axis, instrumental fluxes,
visibility amplitude and invalid flags. GD and differential phase are `None` unless
`include_phase=True`; both are required when saving P2VMRED. Science and live
extraction use the saved P2VM grid. Invalid physical mappings, detector dimensions
and wavelength coverage fail at the calculation/extraction boundary.

## Numerical precision

The I/O boundary owns precision: floating arrays are written as **float32**
(**complex64** for complex signals), and loaded as **float64/complex128** for
calculation. Integer pixels, indices and flags retain their integer/bool types.
Existing raw files are never rewritten. FITS headers keep their standard scalar
representation. `io/hdu.py` contains the policy; numerical algorithms contain no
precision casts. Array-level Python callers supply calculation arrays, using
`as_compute_array` from that module once at their input boundary if needed.

Raw reading still processes 16 frames at a time. Dark statistics merge chunk
means and centered sums of squares, retaining population RMS without subtracting
large second moments. Wavelength coverage tolerates only the float32 encoding
error bound (`2**-24` relative); larger coverage gaps are rejected.

Float32 calibration storage introduces measurable rounding through the chain;
see the controlled comparison in [the change report](docs/CHANGE_REPORT.md).

## Organization

Each scientific step owns its file entry point, parameters, array calculation and
local helper functions. Read a workflow through `make_dark_products`, `make_flat`,
`make_wave`, `make_p2vm` and `reduce_science`. Array operations are named for their
science: `measure_dark`, `detect_bad_pixels`, `build_flat`,
`fit_wavelength_solution`, `fit_p2vm` and `reduce_spectra`. Saving stays explicit.

| Location | Responsibility |
| --- | --- |
| `dark.py` | DARK/BAD inputs, temporal statistics and bad-pixel parameters |
| `flat.py` | Four telescope inputs, aperture parameter, response and trace fitting |
| `wavelength.py` | Lamp/reference inputs, line matching and wavelength surface fitting |
| `p2vm.py` | Calibration sequence, target grid, output mapping and P2VM response fitting |
| `reduction.py` | Science inputs, instrumental observables, GD and live FrameReducer |
| `extraction.py` | Shared pixel extraction, wavelength resampling and chunked file extraction |
| `products.py` | Plain `Dark`, `Bad`, `Flat`, `Wave`, `P2VM`, `P2VMRed` data; no file provenance or dependency graph |
| `instrument.py` | Detector settings and fixed physical telescope/baseline conventions |
| `qc.py`, `plotting.py` | Descriptive quality measurements and scientific plots |
| `io/raw.py`, `io/fits.py` | Physical raw fields, product FITS layout and reading |
| `io/acquisition.py` | Capture metadata, raw header construction and raw writing |
| `io/hdu.py` | Storage/computation precision, native byte order and shared FITS primitives |
| `io/keywords.py`, `io/publication.py` | FITS names and non-overwriting publication |
| `recipes/planetes_*.py` | SOF roles → step call → CPL DFS output registration |
| `recipes/parameters.py`, `recipes/dfs.py` | Shared CPL translation only; every CPL import stays in recipes/ |

FLAT/P2VM Python file inputs are explicit `Mapping[str, Path]` values:
`telescope_flats` has keys `4,3,2,1`; `baseline_cubes` has keys
`43,42,41,32,31,21`. Their values remain paths throughout the workflow.
In PyEsoRex, repeated SOF entries are mapped from shutter metadata at the recipe
boundary. Each recipe then reads its DARK/FLAT/WAVE/P2VM products explicitly.

There is one implementation per step. Python and PyEsoRex use those same
functions. No task dispatcher, separate calibration layer or identity module is
needed. Scientific parameters and defaults belong to the step functions. Each
corresponding recipe declares the CPL names/aliases and SOF multiplicities; shared
adapters only translate those declarations. Importing or running a step does not
require CPL or Qt.

Wavelength, OPD and GD use **metres throughout Python and FITS**. Flux stays ADU,
coherence phases use radians and differential phase uses degrees. Valid Python
pixel indices are zero-based; the FITS boundary writes the GRAVITY one-based
coordinates. GRAVITY's existing `... UM SC` QC cards are emitted alongside their
metre equivalents at the file boundary.

[Code style](docs/CODE_STYLE.md) puts direct data flow and fixed input forms before
local names and table alignment.
[Simplification notes](docs/SIMPLIFICATION.md) explain the product interfaces.
[Change report and diagrams](docs/CHANGE_REPORT.md) record the migration and
full-data evidence. [Scientific methods](docs/SCIENTIFIC_METHODS.md) describe retained numerical
definitions. During development, 12 focused tests cover core scientific calculations,
bench P2VM inputs, FITS round trips/non-overwriting writes and one optional PyEsoRex
P2VM-to-science chain. One-off migration scripts
and logs are kept in Git history and the local archive referenced in the report.

```bash
python -m unittest discover -s tests -v
```

Real recipe tests require PyCPL/PyEsoRex and are skipped when unavailable.
Numerical comparisons use the same environment and matching Numba cache
states: the old algorithm itself has a small cold/warm difference. Cache and
fastmath settings are retained.

Version 0.7 introduced plain `Dark`, `Flat`, `Wave`, `P2VM` and `P2VMRed`
dataclasses. Replace previous `*Calibration`, `VisibilityObservables` and
`ReducedObservables` imports directly; there are no legacy aliases. Array
computations keep their existing units and definitions. General validators,
SHA-256 product fingerprints, parent dependency graphs and `Provenance` are
removed. Python callers select their calibration files explicitly; ESO DFS
records the actual recipe inputs and parameters.

Version 0.8.0 adopts the workflow/array API above, `P2VM.matrix`,
`baseline_outputs` and the internal `P2VMSpectra` type. Use
`aperture_width_pixels` for the full extraction aperture; `include_phase=False`
skips group delay and differential phase in live previews. `trace_spectrum` always
returns centers, position errors and Gaussian sigma. The existing CPL parameter
name/alias `planetes.calib.profile-width` / `--profile-width` remains unchanged.

Version 0.8.1 merges DarkImage into `Dark` and shortens BadPixelMap to `Bad`.
The old classes and DarkInput union are removed without compatibility aliases.
Mean-only darks remain usable without fabricating zero RMS; QC reports available
mean statistics and omits unmeasured temporal RMS.

Version 0.9.0 centralizes fixed telescope pairs in `instrument.BASELINE_PAIRS`;
products no longer store `baseline_telescopes`. Actual `baseline_outputs` remain
in the products. P2VM scan/ellipse/DC/residual diagnostics are omitted by default;
use `keep_diagnostics=True` in Python or `--keep-diagnostics=true` in PyEsoRex
to retain them. This does not alter calibration or reduction.

Version 0.13.0 removes `load_dark`, `load_wavelength_reference` and `load_reducer`.
Read calibrations with `read_product(path, ExpectedClass)` and construct
`FrameReducer(flat, wave, p2vm)` explicitly. `make_wave` requires `reference`.
`P2VMSpectra` contains only scans, telescope flats, output identities and wavelength;
pass optional reference spectra separately to `fit_p2vm`. Qt6 patches, their
manifest and deployment documentation are removed from this repository.

Version 0.12.0 requires explicit file mappings in Python FLAT/P2VM workflows.
Replace former raw-file lists with `{identity: path}` mappings; SOF syntax is
unchanged. The former `identify_raw_exposures`, `load_recipe_inputs` and
`save_product` helpers are removed. Recipes use `read_sof` (path lists plus DFS
frames), explicit `read_product` calls, and the single `save_products` entry.
Raw format-3 metadata keeps its existing interpretation with one explicit
header-first physical-field selection and saved source flags.

Version 0.11.0 makes the P2VM fit follow the physical stages above. Call
`fit_abcd_ellipse(spec)` without the removed unit `envs` argument; the existing
pseudoinverse, thresholds and phase order are retained.

New products use FITS format **6**; formats **4/5** remain readable. Full structured
raw files stay at format **3**. Product data layout and units remain documented in
[the product contract](docs/PRODUCT_CONTRACT.md); optional measurements retain
[the QC definitions](docs/QC_PARAMETERS.md).

The current product is per-frame instrumental **P2VMRED**. Averaged VIS/VIS2/T3
and astronomical transfer-function calibration are later scientific stages; no
placeholder modules or output classes are added for them. Instrumental phase
references and astronomical calibrators have separate physical roles.

Outputs remain instrumental ADU, visibility amplitude, GD and differential phase.
ESO dictionary approval, EDPS association, uncertainties, transfer functions,
final OIFITS and Phase 3 acceptance remain subsequent work.
