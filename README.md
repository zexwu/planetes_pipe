# PLANETES data reduction

Version 0.2.0 runs detector calibration, wavelength calibration, P2VM construction,
and instrumental science reduction as four PyCPL recipes. Files are selected by
explicit tags in a set-of-frames (`.sof`) file. Recipe parameters are managed by
PyEsoRex. There is no runtime YAML configuration or custom `planetes` command.
Raw FITS and existing scientific products are never overwritten.

The numerical definitions retain the original 4-telescope, 6-baseline,
24-output implementation. The resulting science FITS contains instrumental
observables; it is not a transfer-function-calibrated OIFITS/Phase 3 product.

## Install

Install CPL and PyCPL/PyEsoRex using the
[official ESO instructions](https://www.eso.org/sci/software/pycpl/pycpl-site/user/installation.html).
With CPL installed at a dedicated prefix, use the lockfile to reproduce the
validated dependency versions in a separate environment:

```bash
UV_PROJECT_ENVIRONMENT=/path/to/new-venv CPLDIR=/path/to/cpl \
    uv sync --locked --extra eso
```

For scientific functions and FITS I/O without the execution framework, install
with `pip install -e .`. The scientific modules do not import PyCPL.

The locally installed executor from the initial trial remains available at
`reduced/eso_pilot/runtime/venv/bin/pyesorex`. Its environment uses Python 3.13,
CPL 7.4, PyCPL 1.0.4.post1, and PyEsoRex 1.0.4. CPL's optional FFT support is
not enabled; the retained P2VM code uses NumPy. The original `.venv` was not
reinstalled by this migration.

## Run

List the recipes:

```bash
pyesorex --recipe-dir=/absolute/path/to/planetes_pipe/planetes/eso --recipes
```

Use a separate output directory for each recipe invocation. Run the executor
inside that directory so its default log filename remains short; CPL limits the
length of its log filename, including any explicit log directory.

```bash
mkdir -p /path/to/new-flat-output
cd /path/to/new-flat-output
pyesorex --recipe-dir=/absolute/path/to/planetes_pipe/planetes/eso \
    planetes_flat /absolute/path/to/flat.sof
```

All file paths in the SOF templates must be replaced with actual absolute paths.
Input order is irrelevant. Every supported tag accepts exactly one file;
duplicate, missing, or unknown tags are rejected.

| Recipe | Required tags | Optional tags | Product |
| --- | --- | --- | --- |
| `planetes_flat` | `DARK_RAW`, `TEL_FLAT_4`, `TEL_FLAT_3`, `TEL_FLAT_2`, `TEL_FLAT_1` | none | `planetes_flat.fits` |
| `planetes_wave` | `WAVE_RAW`, `WAVE_DARK`, `PLANETES_FLAT` | `PLANETES_LINE_LIST` | `planetes_wave.fits` |
| `planetes_p2vm` | `PLANETES_FLAT`, `PLANETES_WAVE`, four `TEL_FLAT_*`, six `P2VM_RAW_*` | `PHASE_REFERENCE`, `PHASE_DARK` | `planetes_p2vm.fits` |
| `planetes_reduce` | `SCIENCE_RAW`, `PLANETES_FLAT`, `PLANETES_WAVE`, `PLANETES_P2VM` | `SCIENCE_DARK` | `planetes_reduced.fits` |

The telescope order is `4,3,2,1`; baseline order is `43,42,41,32,31,21`.
Four outputs within each baseline retain A,C,B,D ordering. The SOF examples in
`examples/` show the exact input roles. A `.sof.in` is a template, not a runnable
input until its paths have been filled in.

Run wave after flat, P2VM after flat and wave, and science reduction after all
three calibration products exist. The wave recipe automatically includes the
packaged static line-list FITS in its provenance unless an explicit
`PLANETES_LINE_LIST` is provided. The original ECSV and per-line origin labels
are retained beside it; the 30 wavelength values are unchanged.

The one existing science-mode switch is a standard recipe parameter:

```bash
pyesorex --recipe-dir=/absolute/path/to/planetes_pipe/planetes/eso \
    planetes_p2vm --phase-correction=true /absolute/path/to/p2vm.sof
```

Its canonical name is `planetes.p2vm.phase_correction`, default `true`.
PyEsoRex also supports saving parameters in a recipe `.rc` file. If no phase
reference is supplied, the retained algorithm skips that correction and the
output records that it was not applied. A science dark overrides the master
dark when supplied; a phase-reference dark is independently selected.

## Code organization

```text
planetes/eso/          PyCPL input validation, parameters, DFS provenance, products
planetes/processing.py full-cube reads and spectral-extraction orchestration
planetes/algorithms/   array/object calculations, independent of file paths and runners
planetes/io/           versioned FITS schemas and the static wavelength reference
planetes/product.py    in-memory scientific dataclasses
planetes/plotting.py   diagnostics from a ReducedProduct
planetes/instrument.py fixed telescope, baseline, and output conventions
bench/                experimental realtime code, outside the installed package
```

Recipe adapters call ordinary scientific functions and shared I/O functions.
The common ESO helpers cover tag selection and product registration; they do not
implement a second workflow engine. The processing layer preserves 16-frame
chunking and full-cube reading. Intermediate calibration spectra remain in memory.

## FITS products

Calibration and instrumental science products have an empty primary data array
and named extensions. Every HDU has `CHECKSUM` and `DATASUM`. Readers check the
schema version, shapes, units, instrument ordering, baseline identities, and
checksums. Native-endian arrays are returned for Numba. The baseline table uses
explicit zero-based indices; physical beam identities are recorded in the header.

Complex coherent flux is stored as separate real and imaginary ADU extensions,
then reconstructed by component assignment. Boolean science flags are stored as
0/1 bytes and restored as boolean arrays. Neither masks nor numerical values
are recomputed by the serializer.

CPL DFS utilities record the recipe, version, parameters, inputs, and product
signature. Descriptive QC measurements are documented in `docs/QC_PARAMETERS.md`.
They do not add rejection thresholds or change the science selection.

Product categories, the payload schema, and the QC definitions still require ESO
review. Original laboratory FITS headers are read as supplied, with roles made
explicit by SOF tags. Exposure-time units, time systems, and the additional ESO
observation metadata must be agreed with the acquisition software before archive
integration. The version does not supply EDPS association rules, propagated
science uncertainties, on-sky transfer-function calibration, closure-phase
products, or Phase 3 ingestion validation.

## Diagnostic plots

The existing plot calculations consume an object instead of a configuration:

```python
from pathlib import Path
from planetes.io.products import read_product
from planetes.product import ReducedProduct
from planetes.plotting import plot_reduced

reduced = read_product(Path('/path/to/planetes_reduced.fits'), ReducedProduct)
plot_reduced(reduced, Path('/path/to/new-diagnostic.pdf'), object_label='science')
```

Plots are created only when explicitly called, and existing diagnostic files are
not overwritten. Use `full=True` for the additional sampled pages.

## Validation and recovery

```bash
reduced/eso_pilot/runtime/venv/bin/python -m unittest discover -s tests -v
```

Tests that execute recipes require the optional real PyCPL/PyEsoRex runtime.
They are skipped if that runtime is not installed. Numerical-core and FITS tests
can also run in the original environment.

The retained Numba Gaussian fit shows a small cold-compilation versus warm-cache
difference in the original implementation as well as this version. Byte-level
comparisons must use the same environment and matching cache states; the report
includes the controlled measurements. Cache and fastmath settings were not changed.

See `docs/CHANGE_REPORT.md` for the original-code backups, file-by-file changes,
real-data regression, and flowcharts. Existing old NPZ products are left in place
and are not silently converted. Generate fresh FITS calibrations from the raw
inputs before reducing science with this version.
