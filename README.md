# PLANETES pipeline

Raw FITS → DARK/BAD → FLAT → WAVE → P2VM → per-frame instrumental observables.
Import as `planetes_pipe`; use the parent workspace's Python 3.13 + uv environment.

## Run

[Two Python examples](examples/README.md):

```powershell
uv run --locked python planetes_pipe/examples/reduce_calibrations.py
uv run --locked python planetes_pipe/examples/reduce_science.py
```

Edit the paths and scientific settings at the top first. Calibration inputs are
selected by FITS type, shutter state and exposure, not filename tags. Raw files
are read only; product writers refuse existing destinations.

## Code

- `model/`: product and QC dataclasses, detector setup and instrument conventions.
- `processing/`: scientific calculations and QC measurements. `common.py` contains
  shared frame averaging; extraction and group delay keep their named modules.
- `io/raw.py`: Raw metadata and one-pass frame chunks; `io/association.py`
  selects calibration files by header.
- `io/dark.py`, `flat.py`, `wave.py`, `p2vm.py`, `science.py`: each product's
  `read()`, `encode()` and `write()`. `io/common.py` holds shared FITS operations.
- `plots/`: diagnostic plots and PDF reports.
- `eso/`: optional CPL/PyEsoRex adapters; see [SOF usage](docs/eso/README.md).
Wavelength reference lines are defined directly in `processing/wave.py`.

Wavelength, OPD and GD use metres. Telescope order is 4,3,2,1; baselines are
43,42,41,32,31,21; output order is A,C,B,D. Products are instrumental P2VMRED,
not final calibrated OIFITS.

[Scientific methods](docs/SCIENTIFIC_METHODS.md) · [FITS contract](docs/PRODUCT_CONTRACT.md) ·
[QC definitions](docs/QC_PARAMETERS.md)

From the workspace root, check changes with:

```powershell
uv run --locked python -m unittest discover -s planetes_pipe/tests -v
uv run --locked python -m unittest discover -s bench/tests -v
```

Keep the sibling bench compatible when changing public APIs or product formats.
`build/` is disposable setuptools output, not source or a runtime dependency.

DARK calculation consumes raw chunks while the file is open:

```python
from planetes_pipe.io import raw, dark
from planetes_pipe.processing.dark import make_dark_products, measure_quality

with raw.read(input_path, category="DARK_RAW") as frames:
    result, bad = make_dark_products(frames)
dark.write(output_path, result, quality=measure_quality(result))
```
