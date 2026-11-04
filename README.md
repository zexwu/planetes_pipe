# PLANETES pipeline

Raw FITS → DARK/BAD → FLAT → WAVE → P2VM → per-frame instrumental observables.
Import as `planetes_pipe`; use the parent workspace's Python 3.13 + uv environment.

## Run

Each step is a module with a `run()` function: input files in, product files out,
returning `{product category: path}`. It is the same code the PyEsoRex recipe
`planetes_<step>` runs, and products carry the same DICD `PRO` provenance.

```python
from planetes_pipe.steps import dark, flat, wave, p2vm, science

cal = dark.run("dark.fits", output_dir="cal")                  # {"DARK": ..., "BAD": ...}
ff = flat.run(cal["DARK"], cal["BAD"], flat_raws, output_dir="cal")
wl = wave.run("lamp.fits", lamp_dark, ff["FLAT"], output_dir="cal")
pm = p2vm.run(cal["DARK"], ff["FLAT"], wl["WAVE"], flat_raws, scans, output_dir="cal")
science.run("science.fits", sci_dark, ff["FLAT"], wl["WAVE"], pm["P2VM"], output_dir="sci")

flat_product = flat.read_flat(ff["FLAT"])   # products in memory
```

[Two complete examples](examples/README.md):

```powershell
uv run --locked python planetes_pipe/examples/reduce_calibrations.py
uv run --locked python planetes_pipe/examples/reduce_science.py
```

FLAT and P2VM raw files may be given in any order: telescope and baseline identity
come from the recorded shutters. Raw files are read only; existing products are
never overwritten.
The optional PyEsoRex route is described in [docs/eso](docs/eso/README.md).

## What is where

```
planetes_pipe/
  steps/            the reduction
    dark.py flat.py wave.py p2vm.py science.py
                    one per product, top to bottom: data class, algorithm
                    (with its numba kernels), qc(), read_<x>/write_<x>, run()
    extraction.py   shared by WAVE, P2VM and science: spectral extraction,
                    resampling, group delay
  io/               the file format
    keywords.py     every FITS keyword, defined once with unit and comment (ESO dictionary style)
    raw.py          raw data: the Raw cube, reading raw files, DET/shutter headers
    products.py     product FITS: write/read, extensions, provenance header
  instrument.py     channel order, DetectorSetup, shutter identity
  plots.py          diagnostic PDF reports
  eso/recipes.py    PyEsoRex recipes; the only file that imports CPL
examples/           two complete reduction scripts
tests/              test_<module>.py per module; fixtures.py builds synthetic raw files;
                    test_headers.py checks product headers; test_recipes.py runs PyEsoRex
docs/               methods.md, products.md (FITS, QC, options), code_style.md,
                    changes.md, architecture.svg, eso/ (SOF usage)
```

Choosing which raw files and calibrations belong together is not part of the
pipeline: scripts name their inputs (as in `examples/`), and the bench script
selects them from a capture directory. The steps check that what they are given
is consistent (DIT, ROI, detector, shutters).

### Three levels per step

Each step can be used at the level that fits the task:

| Level | Input → output | DARK | FLAT | WAVE | P2VM | Science |
| --- | --- | --- | --- | --- | --- | --- |
| files | paths → product files | `dark.run` | `flat.run` | `wave.run` | `p2vm.run` | `science.run` |
| raw | `Raw` cubes + products → product | `make_dark_products` | `make_flat` | `make_wave` | `extract_p2vm_spectra` | `reduce_science` |
| arrays | arrays → product | `measure_dark`, `detect_bad_pixels` | `build_flat` | `fit_wavelength_solution` | `fit_p2vm` | `reduce_spectra` |

`run()` reads the files, calls the raw level, computes QC and writes the products;
its arguments are file paths (`dark_file`, `flat_files`, ...). The raw level checks
that calibrations match the raw detector setup; FLAT and P2VM take their raws keyed
by physical identity, as returned by `io.raw.read_raws`. The array level is pure
arithmetic, used by the live viewer and the tests.

```python
from planetes_pipe.io.raw import read_raw
from planetes_pipe.steps.dark import make_dark_products

master, bad = make_dark_products(read_raw("dark.fits", "DARK_RAW"))
```

Wavelength, OPD and GD use metres. Telescope order is 4,3,2,1; baselines are
43,42,41,32,31,21; output order is A,C,B,D. Products are instrumental P2VMRED,
not final calibrated OIFITS.

## Adding to the pipeline

- **A keyword**: define it once as a `Keyword` (ESO short name, comment, unit) and
  read or write it only through that object: `DIT.get(header)`, `DIT.set(header, 1.0)`.
  QC keywords sit beside the step's `qc()`, which returns `{Keyword: value}`;
  others go in `io/keywords.py`. `tests/test_headers.py` checks every product header.
- **A product or step**: a new module in `steps/` following `dark.py`, a `tests/test_<module>.py`,
  and a recipe class in `eso/recipes.py`.
- **A shared algorithm**: keep it in its step module until a second step needs it,
  then move it to `steps/extraction.py`. Numba kernels stay next to the code that calls them.

[Methods](docs/methods.md) · [Products and QC](docs/products.md) · [Code style](docs/code_style.md) ·
[Changes](docs/changes.md)

From the workspace root, check changes with:

```powershell
uv run --locked python -m unittest discover -s planetes_pipe/tests -v
uv run --locked python -m unittest discover -s bench/tests -v
```

Keep the sibling bench compatible when changing public APIs or product formats.
`build/` is disposable setuptools output, not source or a runtime dependency.
