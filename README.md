# PLANETES pipeline 0.15.0

PLANETES computes instrumental interferometric observables from detector cubes.
Python and PyEsoRex share the same scientific steps and single-frame kernels.
File entry points return typed objects; saving is explicit in Python. Raw data are read only, and product writers refuse existing destinations.
Qt6 applications are maintained separately; this repository contains no GUI patches.

P2VM calculation: [flow diagram](docs/p2vm_flow.png) ·
[phase and normalization conventions](docs/SCIENTIFIC_METHODS.md#p2vm-p2vmpy) ·
[change report](docs/CHANGE_REPORT.md).

## Python package and bench compatibility

The distribution is `planetes-pipe`, imported as `planetes_pipe`. The separate
`catkit2-planetes` hardware project owns `planetes`; both must be able to coexist
in one bench process. The ESO recipe names remain `planetes_dark`, `planetes_flat`,
`planetes_wave`, `planetes_p2vm` and `planetes_reduce`.

When changing public functions, product formats/names or package metadata, update
and validate the sibling `../bench` ([planetes_bench](https://github.com/zexwu/planetes_bench))
in the same change. Run its compatibility
tests using its own locked environment, including offline GUI loading and the
comparison of live reduction with the same pipeline/calibrations. Select products
by `ESO PRO CATG`, not a fixed filename. Check the dataset's calling scripts too.

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

The ESO recipe tests use Python 3.13, CPL 7.4, PyCPL 1.0.4.post1 and PyEsoRex
1.0.4. Qt and PyCPL are optional; the numerical modules import neither.

## Run with PyEsoRex

```bash
pyesorex --recipe-dir=/absolute/path/to/planetes_pipe/planetes_pipe/recipes --recipes
mkdir /path/to/new-dark-output
cd /path/to/new-dark-output
pyesorex --recipe-dir=/absolute/path/to/planetes_pipe/planetes_pipe/recipes planetes_dark /absolute/path/to/dark.sof
```

Use a new directory per invocation. Fill the absolute paths in `examples/*.sof.in`
before using them. Tags describe categories; repeated FLAT_RAW/P2VM_RAW entries are
identified by physical shutter metadata, independently of order and filenames.
SOF categories are unchanged. Legacy files without shutter readback require
explicitly assigned Python identity mappings. The supplied example uses the
standardized bench raw files with physical shutter headers.

| Recipe | Required SOF roles | Optional roles | Output with `--static-name=true` |
| --- | --- | --- | --- |
| `planetes_dark` | `DARK_RAW` | — | `planetes_dark.fits` + `planetes_bad.fits` |
| `planetes_flat` | `DARK`, `BAD`, `FLAT_RAW` × 4 | — | `planetes_flat.fits` |
| `planetes_wave` | `DARK`, `FLAT`, `WAVELAMP_RAW` | `LINE_LIST` | `planetes_wave.fits` |
| `planetes_p2vm` | `DARK`, `FLAT`, `WAVE`, `FLAT_RAW` × 4, `P2VM_RAW` × 6 | `PHASE_REFERENCE`, `PHASE_DARK_CALIB` | `planetes_p2vm.fits` |
| `planetes_reduce` | `DARK`, `FLAT`, `WAVE`, `P2VM`, `SINGLE_SCI_RAW` | — | `planetes_reduced.fits` |

Default product names follow GRAVITY's reference-raw convention:
`<raw stem>_<PRO CATG lowercased without underscores>.fits`. The raw stem is
kept exactly, including timestamps, fractional seconds and explicit `UNKNOWN`
markers. For example:

```text
PLANETES.2026-02-05T210609.fits
→ PLANETES.2026-02-05T210609_singlescip2vmred.fits
PLANETES.2026-02-05TUNKNOWN.0006.fits
→ PLANETES.2026-02-05TUNKNOWN.0006_p2vm.fits
```

DARK/BAD use their DARK_RAW input, WAVE uses WAVELAMP_RAW, and science uses
SINGLE_SCI_RAW. FLAT explicitly uses the T4 flat and P2VM uses the 43-baseline
scan; changing SOF order therefore does not change their product names.
The same raw supplies the inherited DFS header, including observation-time
cards when present. `--static-name=true` selects the fixed names in the table above. No observation
time or timezone is invented from the current clock, file mtime or UNKNOWN text.
The SOF templates use dynamic names derived from their example raw filenames.

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

Scientific messages use CPL logging during each recipe invocation, respecting
PyEsoRex's independent `--msg-level` and `--log-level` settings. The previous
Python logger configuration is restored when the recipe returns or raises.
Standalone scripts can call `planetes_pipe.configure_logging()` for console progress.

Parameters use the same scientific defaults in Python and ESO. For example:

```bash
pyesorex --recipe-dir=/absolute/path/to/planetes_pipe/planetes_pipe/recipes planetes_p2vm \
  --phase-correction=false --wavelength-min-m=1.15e-6 --wavelength-max-m=1.30e-6 \
  --wavelength-channels=151 /absolute/path/to/p2vm.sof
```

See [parameters and FITS contract](docs/PRODUCT_CONTRACT.md) for all settings.
PyEsoRex can save its own `.rc` parameter file; a project YAML is unnecessary.

## Run from Python

[examples/reduce.py](examples/reduce.py) directly reduces the standardized
`../2026_02_05/raw/calibration` and `raw/science` dataset:

```bash
.venv/bin/python examples/reduce.py
```

The script derives the sibling dataset directory from its own location. Change
`night` and the five explicitly selected DARK/lamp/science paths for another
night. FLAT and P2VM input identities come from DPR type and physical shutter
headers, never from timestamp/import-index text. Three DARKs are selected
separately; the main calibration DARK has unknown DIT in this dataset.

Outputs go to a **new** `../2026_02_05/reduced_example/` directory, with the raw
stem preserved. Re-running requires selecting another output directory. The
script saves seven calibrations, reads the saved DARK/FLAT/WAVE/P2VM back, then
writes per-frame science observables. This matches the bench's use of serialized
calibrations. No instrumental phase reference is selected for this dataset.

The SOF templates use these actual raw names and the calibration products from
`reduced_example`. Replace `/absolute/PLANETES_new` with your workspace root;
point calibration paths at your chosen recipe outputs if needed. `dark.sof.in`,
`lamp_dark.sof.in` and `science_dark.sof.in` explicitly distinguish the three
DARK inputs. The P2VM SOF contains no fictitious phase-reference exposure.

`make_wave` always receives an explicit line list. The WAVE recipe selects the
packaged list when no LINE_LIST is supplied and records it as a DFS input.

For a separate dataset with an instrumental phase reference, the following
continues from the example's variables in an interactive session. Explicitly
select that reference file and its DARK, then extract it on the scan grid:

```python
import numpy as np
from planetes_pipe.extraction import extract_spectra_from_file, prepare_extraction
from planetes_pipe.instrument import check_setup
from planetes_pipe.io.raw import read_raw_metadata
from planetes_pipe.p2vm import extract_p2vm_spectra, fit_p2vm

grid = np.linspace(1.15e-6, 1.30e-6, 151)
scans = extract_p2vm_spectra(
    flat, wave, main_dark, telescope_flats=telescope_flats,
    baseline_cubes=baseline_cubes, wavelength_m=grid,
)
reference_path = Path('/absolute/path/to/selected_reference.fits')
reference_dark = science_dark  # Explicitly choose the dark for this exposure.
metadata = read_raw_metadata(reference_path, category='SINGLE_SCI_RAW')
check_setup(metadata.setup, reference_dark.setup, exposure=True, role="DARK")
check_setup(metadata.setup, flat.setup, exposure=False, role="FLAT")
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
contradictions still fail; missing metadata stays unknown. Explicit Python mappings
can supply physical identities for such legacy cubes; filenames and list order
never assign them.

Raw readers and all file steps default to `IMAGING_DATA_SC`. For a primary-HDU
bench cube, explicitly pass `hdu=0`; another named/numbered image HDU can be selected
the same way. There is no automatic HDU search. `read_raw_metadata` also requires
an explicit `category`, then checks any declared DPR type against it. A known
non-ADU unit is rejected; missing BUNIT follows the bench ADU convention. Use
Astropy `fits.getheader(path)` to inspect the original header.
`RawMetadata` contains only category, four shutter states (each may be None), and
detector setup. Physical fields come from the header and cube dimensions. No JSON
metadata is read or written. Raw acquisition/writing belongs to the acquisition
project; it is not part of this package.

Read calibrations with `read_product(path, ExpectedClass)`. Save them with
`write_product(path, product)` or explicitly supplied QC:

```python
quality = measure_quality(product)
path = output_directory / product_filename(reference_raw, product)
write_product(path, product, quality=quality)
```

`product_filename` is shared with the recipes and preserves the selected raw
stem. `write_product` still uses exactly the path supplied by the caller; it
neither renames explicit destinations nor chooses a reference raw implicitly.

The writer only encodes its inputs. Recipes explicitly measure the existing QC
set before calling the DFS writer. `qc.measure_quality(product)` returns available
statistics; unknown values are omitted. No flags or scientific masks are changed.
Writers generate checksums and publish through temporary files without overwriting
existing paths. Each file is published independently; a later failure does not
remove earlier complete outputs. Writers do not read back their own outputs.
Product readers still check checksums, units and physical identities. FITS
round-trip comparisons belong to the existing development tests.
See [the retained QC set](docs/QC_PARAMETERS.md).
For single frames:

```python
from planetes_pipe.reduction import FrameReducer
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
| `io/hdu.py` | Precision, native byte order, checksums and single-file staging |
| `io/keywords.py` | Explicit FITS names, categories and QC keyword mappings |
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
definitions. The 17 focused tests cover core scientific calculations, a synthetic
raw DARK→FLAT→WAVE chain with known traces and dispersion, rejection of mismatched
DARK dimensions, bench P2VM inputs, FITS round trips/non-overwriting writes and
timestamp/UNKNOWN filename preservation and optional PyEsoRex
P2VM-to-science/logging and dynamic/static naming checks. One-off migration scripts
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

Version 0.11.0 makes the P2VM fit follow the physical stages above. Call
`fit_abcd_ellipse(spec)` without the removed unit `envs` argument; the existing
pseudoinverse, thresholds and phase order are retained.

Version 0.14.0 removes JSON persistence, `PLANETES_META`, `CaptureMetadata`,
exposure-source/legacy states and the header-versus-JSON reconciliation code.
Physical fields have one location: FITS headers. Raw files no longer receive a
private format version; existing raw headers and detector cubes can still be read
without consulting any extra metadata extension.

Version 0.15.0 removes private product-format and BAD-bit marker gates. Product
reading depends on category, actual HDUs/columns, units and physical identities,
so a private version number alone no longer blocks a readable file. There is no
JSON compatibility path. `write_product` takes explicit `quality` instead of
`include_qc`; raw file steps take explicit `hdu` when the standard extension is
not used. Raw writing and multi-file rollback have been removed from the package.
Product data layout and units remain documented in
[the product contract](docs/PRODUCT_CONTRACT.md); optional measurements retain
[the QC definitions](docs/QC_PARAMETERS.md).

The current product is per-frame instrumental **P2VMRED**. Averaged VIS/VIS2/T3
and astronomical transfer-function calibration are later scientific stages; no
placeholder modules or output classes are added for them. Instrumental phase
references and astronomical calibrators have separate physical roles.

Outputs remain instrumental ADU, visibility amplitude, GD and differential phase.
ESO dictionary approval, EDPS association, uncertainties, transfer functions,
final OIFITS and Phase 3 acceptance remain subsequent work.
