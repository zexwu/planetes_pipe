# PLANETES pipeline

Raw FITS → DARK/BAD → FLAT → WAVE → P2VM → per-frame instrumental observables.
Import as `planetes_pipe`; use the parent workspace's Python 3.13 + uv environment.

## Run

Each step is a module with a `run()` function: input files in, product files out,
returning `{product category: path}`. It is the same code the PyEsoRex recipe
`planetes_<step>` runs, and products carry the same DICD `PRO` provenance.

```python
from planetes_pipe import dark, flat, wave, p2vm, reduce

cal = dark.run("dark.fits", output_dir="cal")                  # {"DARK": ..., "BAD": ...}
ff = flat.run(cal["DARK"], cal["BAD"], flat_raws, output_dir="cal")
wl = wave.run("lamp.fits", lamp_dark, ff["FLAT"], output_dir="cal")
pm = p2vm.run(cal["DARK"], ff["FLAT"], wl["WAVE"], flat_raws, scans, output_dir="cal")
reduce.run("science.fits", sci_dark, ff["FLAT"], wl["WAVE"], pm["P2VM"], output_dir="sci")

flat_product = flat.read_flat(ff["FLAT"])   # products in memory
```

[Two complete examples](examples/README.md):

```powershell
uv run --locked python planetes_pipe/examples/reduce_calibrations.py
uv run --locked python planetes_pipe/examples/reduce_science.py
```

Calibration inputs are selected by FITS type, shutter state and exposure, not
filename tags. Raw files are read only; existing products are never overwritten.
The optional PyEsoRex route is described in [docs/eso](docs/eso/README.md).

## What is where

```
planetes_pipe/            the package
  dark.py flat.py wave.py reduce.py
                          one per product, top to bottom: data class, algorithm,
                          qc(), read_<x>/write_<x>, run()
  p2vm/                   the same for P2VM; __init__.py is the step, fit.py the
                          matrix fit (response, inverse, ellipse phases)
  instrument.py           telescope/baseline/output order, DetectorSetup, Raw
  keywords.py             every FITS keyword, raw tag, QC keyword and recipe option
  files.py                raw reading (memory-mapped), product I/O, provenance header
  extraction.py           spectral extraction and group delay shared by steps
  kernels.py              all numba-compiled code
  association.py          pick calibration files from a directory by header
  plots/                  diagnostic PDF reports
  eso/recipes.py          PyEsoRex recipes; the only file that imports CPL
examples/                 two complete reduction scripts
tests/                    test_<module>.py per package module; fixtures.py builds
                          synthetic raw files; test_recipes.py runs PyEsoRex
docs/                     methods.md (algorithms), products.md (FITS, QC, options),
                          code_style.md, changes.md, architecture.svg, eso/ (SOF usage)
```

### Three levels per step

Each step can be used at the level that fits the task:

| Level | Input → output | DARK | FLAT | WAVE | P2VM | Science |
| --- | --- | --- | --- | --- | --- | --- |
| files | paths → product files | `dark.run` | `flat.run` | `wave.run` | `p2vm.run` | `reduce.run` |
| raw | `Raw` cubes + products → product | `make_dark_products` | `make_flat` | `make_wave` | `make_p2vm` | `reduce_science` |
| arrays | arrays → product | `measure_dark`, `detect_bad_pixels` | `build_flat` | `fit_wavelength_solution` | `p2vm.fit.fit_p2vm` | `reduce_spectra` |

`run()` reads the files, calls the raw level, computes QC and writes the products.
The raw level checks that calibrations match the raw detector setup. The array
level is pure arithmetic, used by the live viewer and the tests.

```python
from planetes_pipe.dark import make_dark_products
from planetes_pipe.files import read_raw

master, bad = make_dark_products(read_raw("dark.fits", "DARK_RAW"))
```

Wavelength, OPD and GD use metres. Telescope order is 4,3,2,1; baselines are
43,42,41,32,31,21; output order is A,C,B,D. Products are instrumental P2VMRED,
not final calibrated OIFITS.

## Adding to the pipeline

- **A keyword**: declare it in `keywords.py` first; `tests/test_keywords.py` rejects
  undeclared QC, DET and INS keywords and cards that do not fit 80 columns.
- **A product or step**: a new module following `dark.py`, a `tests/test_<module>.py`,
  and a recipe class in `eso/recipes.py`.
- **A shared algorithm**: keep it in its step module until a second step needs it,
  then move it to a shared module such as `extraction.py`. Numba code goes in
  `kernels.py`.

[Methods](docs/methods.md) · [Products and QC](docs/products.md) · [Code style](docs/code_style.md) ·
[Changes](docs/changes.md)

From the workspace root, check changes with:

```powershell
uv run --locked python -m unittest discover -s planetes_pipe/tests -v
uv run --locked python -m unittest discover -s bench/tests -v
```

Keep the sibling bench compatible when changing public APIs or product formats.
`build/` is disposable setuptools output, not source or a runtime dependency.
