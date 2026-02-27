# Bench Data Reduction Pipeline 

## Setup

**Install dependencies:**
```bash
pip install numpy numba matplotlib scipy astropy pyyaml colorlog
```

## Configuration (`conf.yaml`)

Create a `conf.yaml` file defining the paths.
```yaml
frames: [0, 512]         # DIT range [start, end]
xrange: [0, 1024]        # Spectral crop [start, end]
yrange: [0, 1024]        # Spatial crop [start, end]
output_dir: ./reduced/   # Output path
n_wave: 500              # number of wavelength channels
log_level: DEBUG

telescopes: ["4", "3", "2", "1"] # names of telescopes
baselines: ["43", "42", "41", "32", "31", "21"] # names of baselines

calib:
  dark: /path/to/dark.fits
  wave: /path/to/wave.fits
  wave_dark: /path/to/wave_dark.fits
  flat: [/path/to/tel4.fits, /path/to/tel3.fits, ...]
  p2vm: [/path/to/bsl34.fits, /path/to/bsl24.fits, ...]

object:
  my_target: /path/to/science.fits

products:
  flat: "flat.npz"
  p2vm: "p2vm.npz"
  wave: "wave.npz"
  wave_aberr: "wave_aberr.npz"
  preproc: "{}_preproc.npz"
  reduced: "{}_reduced.npz"
```

## Usage

Run the steps sequentially using the CLI.

### 1. Run calibration sequence 
- flat-fielding & spectra extraction
- wavelength calibration
- p2vm calibration

```bash
python main.py calib # ~ 20 seconds

# or manually step by step
python main.py flat
python main.py wave
python main.py preproc
python main.py p2vm
```

### 2. Apply P2VM to science data to get visibilities

```bash
python main.py reduce --object my_target # ~ 3 seconds
```
## Output Files


| File | Content |
| --- | --- |
| `flat.npz` | bad map, flat map, profile map, and trace data|
| `wave.npz` | wavelength map |
| `p2vm.npz` | V2PM (phase, coherence & transmission) & P2VM matrix |
| `*_preproc.npz` | pre-processed data|
| `*_reduced.npz` | `visdata` , `visphi` , `gdelay` . |
| `*.pdf` | Summary plots |
