# PLANETES data reduction

PLANETES is a small scientific pipeline for detector calibration, wavelength
calibration, P2VM construction, and visibility extraction. Raw FITS files are
read-only.

## Install

```bash
pip install -e .
```

## Configuration

The YAML contains paths and the one retained calibration choice. Detector
layout, full-cube reading, wavelength grid, fitting degrees, and thresholds are
fixed by the current instrument implementation and live beside the relevant
recipe.

```yaml
schema_version: 1
data_dir: .
output_dir: ./reduced
phase_correction: true

calibration:
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
    "41": ./bsl41.fits
    "32": ./bsl32.fits
    "31": ./bsl31.fits
    "21": ./bsl21.fits
  phase_reference:
    data: ./phase_reference.fits
    dark: ./phase_reference_dark.fits

targets:
  my_target:
    data: ./science.fits
    dark: ./science_dark.fits
```

Relative input paths resolve against `data_dir`; `data_dir` and
`output_dir` resolve against the YAML file. Telescope and baseline files are
named mappings so that file order cannot silently change the instrument
channel. The implemented layout is explicitly PLANETES 4T/6B/24-output, with
3D detector cubes in FITS HDU 1. Every data and dark cube is read in full.

Each calibration or reduction run copies the source YAML to
`config_used.yaml`. Algorithm constants are provenance carried by the code
version, not user-facing controls.

## Run

```bash
planetes calib
planetes reduce my_target
planetes plot my_target
```

Use `all` for every configured target. Extra frame- and wavelength-sampled
plot pages are a runtime choice:

```bash
planetes reduce all
planetes plot all --full
```

Select another configuration with `planetes --conf path/to/config.yaml ...`.

The public commands are complete scientific operations. Internal
`flat → wave → extraction → p2vm` stages are ordinary Python calls, not
separate CLI workflows.

## Code layout

- `config.py`: small path dataclasses and YAML loading.
- `instrument.py`: fixed PLANETES detector and output ordering.
- `cube.py`: chunked, full-cube FITS reading.
- `numerics.py`: reusable array-to-array calculations.
- `product.py`: dataclasses for persisted products and current NPZ I/O.
- `data/wavelength_lines.ecsv`: versioned wavelength reference.
- `recipes/`: calibration, reduction, extraction, and plotting.
- `bench/`: experimental real-time code; not installed as pipeline code.
- `main.py`: the three-command CLI.

The calibration recipe passes intermediate arrays directly:

```text
flat -> wave -> CalibrationSpectra -> p2vm
```

Target extraction is likewise in memory during `reduce`. There is no pipeline
context, manifest, dynamic recipe registry, DAG, or persisted preprocessing
cache.

## Current products

| File | Content |
| --- | --- |
| `config_used.yaml` | exact source configuration used for the run |
| `flat.npz` | dark/bad/flat maps and sparse traces |
| `wave.npz` | detector wavelength map |
| `p2vm.npz` | V2PM, P2VM, OPD/GD diagnostics, and mappings |
| `<target>_reduced.npz` | flux, coherent flux, visibility, flags, phase, and GD |
| `<target>.pdf` | explicitly requested diagnostic plots |

NPZ remains a temporary internal product carrier. The numerical recipes consume
typed product objects, leaving a narrow boundary for the planned
schema-generated ESO-compatible FITS implementation.

## Tests

```bash
python -m unittest discover -s tests -v
```
