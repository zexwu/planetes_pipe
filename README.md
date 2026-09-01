# PLANETES bench data reduction

PLANETES is a small scientific reduction pipeline for detector calibration,
wavelength calibration, P2VM construction, and visibility extraction. Raw FITS
files are read-only; all cuts, dimensions, and scientific parameters are recorded
in one YAML configuration.

## Install

```bash
pip install -e .
```

## Configuration

See [`conf.yaml`](./conf.yaml) for a complete working example. Its main structure is:

```yaml
schema_version: 1
data_dir: .
output_dir: ./reduced
chunk_frames: 16
automatic_plots: false

crop:
  x: [0, 2048]
  y: [0, 1024]

instrument:
  telescopes: ["4", "3", "2", "1"]
  baselines: ["43", "42", "41", "32", "31", "21"]
  abcd_order: ["A", "C", "B", "D"]

calibration:
  frames: [0, 256]
  dark: ./dark.fits
  flats:
    "4": ./tel4.fits
    "3": ./tel3.fits
    "2": ./tel2.fits
    "1": ./tel1.fits
  wavelength:
    data: ./wave.fits
    dark: ./wave_dark.fits
  p2vm:
    "43": ./bsl43.fits
    "42": ./bsl42.fits
    # one named file per configured baseline

targets:
  my_target:
    data: ./science.fits
    dark: ./science_dark.fits
    frames: [0, 512]
```

Telescope and baseline inputs are mappings rather than positional lists, so a
file cannot silently move to a different instrument channel. Relative raw-data
paths resolve against `data_dir`; `output_dir` resolves against the YAML file.

All scientific settings live under `parameters`. The CLI deliberately has no
threshold, polynomial-degree, wavelength-grid, phase-correction, or group-delay
overrides. Every run writes the fully resolved configuration to
`reduced/config_used.yaml`.

## Run

Run the full calibration:

```bash
planetes calib
```

Or run its explicit stages:

```bash
planetes flat
planetes wave
planetes preproc p2vm
planetes p2vm
```

Reduce or plot one configured target:

```bash
planetes reduce my_target
planetes plot my_target
```

Use `all` to process every configured target:

```bash
planetes reduce all
planetes plot all
```

Select another configuration with `planetes --conf path/to/config.yaml ...`.

## Code layout

- `config.py`: YAML to immutable `PipelineConfig` dataclasses.
- `cube.py`: chunked reading of raw detector FITS cubes.
- `numerics.py`: reusable array-to-array reductions.
- `product.py`: typed products and stateless serialization.
- `recipes/`: explicit scientific stages and plotting.
- `recipes/calibration.py`: the four-line linear calibration sequence.
- `main.py`: explicit argparse commands dispatched with `match`.

There is no pipeline context/service object, command registry, dynamic recipe
loading, DAG, or freshness manifest. A recipe receives `PipelineConfig`, reads
the products it needs, performs the scientific calculation, and writes one typed
product.

## Current products

| File | Content |
| --- | --- |
| `config_used.yaml` | resolved configuration and scientific parameters |
| `flat.npz` | dark/bad/flat maps and sparse traces |
| `wave.npz` | detector wavelength map |
| `wave_aberr.npz` | alternate aberration-model wavelength map |
| `p2vm_preproc.npz` | extracted calibration spectra and output mapping |
| `p2vm.npz` | V2PM, P2VM, OPD/GD diagnostics, and mapping |
| `<target>_preproc.npz` | resampled target spectra |
| `<target>_reduced.npz` | flux, coherent flux, visibility, flags, phase, and GD |
| `<target>.pdf` | optional diagnostic plots |

NPZ is currently the internal product carrier. Recipes only see the product
dataclasses, so replacing this serializer with schema-driven ESO-compatible FITS
does not require rewriting the numerical recipes.

## Tests

```bash
python -m unittest discover -s tests -v
```
