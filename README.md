# Bench Data Reduction Pipeline

## Setup

**Install dependencies:**
```bash
pip install -e .
```

## Configuration (`conf.yaml`)

Create a `conf.yaml` file defining the paths.
```yaml
frames: [0, 512]         # DIT range [start, end]
xrange: [0, 1024]        # Spectral crop [start, end]
yrange: [0, 1024]        # Spatial crop [start, end]
output_dir: ./reduced/   # Output path
log_level: DEBUG
io_chunk_frames: 16      # FITS frames processed per in-memory chunk
plot_to_pdf: false       # automatic calibration/reduction plots

parameters:
  flat:
    profile_width: 10
    bad_thresh: 5.0
  wave:
    sigma: 3
    deg: 3
    aber_deg: 3
    disp_deg: 3
    niter: 1000000
  preproc:
    nwave: 151
    min_wave: 1.15
    max_wave: 1.30
    object: p2vm
  p2vm:
    phase_correction: true
  plot:
    full: false

telescopes: ["4", "3", "2", "1"] # names of telescopes
baselines: ["43", "42", "41", "32", "31", "21"] # names of baselines
abcd_order: ["A", "C", "B", "D"] # labels in ascending detector-region order

calib:
  dark: /path/to/dark.fits
  wave: /path/to/wave.fits
  wave_dark: /path/to/wave_dark.fits
  flat: [/path/to/tel4.fits, /path/to/tel3.fits, ...]
  p2vm: [/path/to/bsl34.fits, /path/to/bsl24.fits, ...]

object:
  my_target:
    data: /path/to/science.fits
    dark: /path/to/science_dark.fits

  # A path alone remains valid when the calibration master dark applies.
  another_target: /path/to/another_science.fits

products:
  flat: "flat.npz"
  p2vm: "p2vm.npz"
  wave: "wave.npz"
  wave_aberr: "wave_aberr.npz"
  preproc: "{}_preproc.npz"
  reduced: "{}_reduced.npz"
```

## Usage

Run the steps sequentially using the CLI. Recipe parameters come from
`parameters` in the configuration; explicitly provided CLI options override them.

### 1. Run calibration sequence
- flat-fielding & spectra extraction
- wavelength calibration
- p2vm calibration

```bash
planetes calib # ~ 20 seconds

# or manually step by step
planetes flat
planetes wave
planetes preproc
planetes p2vm
```

### 2. Apply P2VM to science data to get visibilities

```bash
planetes reduce --object my_target # ~ 3 seconds
```

Generate a compact five-page summary explicitly with:

```bash
planetes plot --object my_target

# Include per-DIT, per-wavelength, and ABCD diagnostics.
planetes plot --object my_target --full
```
## Output Files


| File | Content |
| --- | --- |
| `flat.npz` | bad map, flat map, profile map, and trace data |
| `wave.npz` | wavelength map |
| `wave_aberr.npz` | aberration-corrected wavelength map |
| `p2vm.npz` | V2PM (phase, coherence & transmission) & P2VM matrix |
| `*_preproc.npz` | pre-processed data|
| `*_reduced.npz` | coherent flux, normalized `visamp`, `flag`, differential phase, and group delay |
| `*.manifest.json` | effective parameters, source-file state, and code fingerprint |
| `*.pdf` | Summary plots |
