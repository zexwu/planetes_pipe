# Calibration Procedures

This note summarizes the algorithms called by `planetes calib`:

- `flat`
- `wave`
- calibration-spectrum extraction
- `p2vm`

The description follows the implementation in the codebase and is meant to explain what is actually computed, in what order, and with what assumptions.

## 1. `flat`: detector calibration and spectral tracing

Implemented in [planetes/recipes/flat.py](/Users/zexwu/zexwu_lib/planetes_pipe/planetes/recipes/flat.py).

### Inputs

- dark cube: `calibration.dark`
- one named flat cube per telescope: `calibration.flats`

### Procedure

1. Compute the mean dark frame:
   - `dark_map = mean(dark_cube, axis=0)`
   - `dark_std = std(dark_cube, axis=0)`

2. Build a simple bad-pixel map from dark statistics:
   - hot pixels: unusually large mean signal
   - dead pixels: unusually low mean signal
   - noisy pixels: unusually large temporal RMS

3. Build the mean flat image:
   - for each telescope flat cube, compute `mean(cube) - dark_map`
   - average all telescope flat maps into one `flat_map`

4. Detect the approximate spectral output positions:
   - collapse `flat_map` along the dispersion direction
   - detect peaks in the spatial profile with `scipy.signal.find_peaks`
   - keep the strongest `n_reg = 4 * n_bsl` peaks

5. Trace each spectrum across the detector:
   - starting from the reference peak, scan in `x`
   - in each column, fit a Gaussian profile in `y`
   - propagate the current centroid from one column to the next
   - smooth the traced center with a median filter plus a cubic polynomial fit

6. Build a binary extraction mask for each output:
   - pixels within `profile_width / 2` of the traced center are accepted
   - bad pixels are removed from the mask

### Products

- `dark_map_adu`
- `bad_pixel_flags`
- `flat_map_adu`
- sparse extraction coordinates: `profile_x_pixels`, `profile_y_pixels` (`-1` marks excluded bad pixels)
- traced spatial centroids: `trace_y_pixels` (the corresponding `x` coordinate is the detector pixel index)

### Main assumption

Each interferometric output is well approximated by a narrow, smooth trace in the detector plane, and the trace can be followed column-by-column from the flat image.

## 2. `wave`: wavelength calibration

Implemented in [planetes/recipes/wavelength.py](/Users/zexwu/zexwu_lib/planetes_pipe/planetes/recipes/wavelength.py).

### Inputs

- `flat` product:
  - sparse extraction coordinates
  - traced output geometry
- wave lamp cube: `calibration.wavelength.data`
- wave dark cube: `calibration.wavelength.dark`

### Procedure

1. Build the mean wave-lamp image:
   - `wave_img = mean(wave_cube) - mean(wave_dark)`

2. Extract one 1D spectrum per detector output:
   - use the sparse extraction masks from the `flat` step

3. Detect emission lines in a reference spectrum:
   - take the mean of the last four extracted outputs as a reference spectrum
   - detect peaks above `median + sigma * MAD`

4. Match detected peaks to a laboratory line list:
   - known line wavelengths are loaded from `planetes/data/wavelength_lines.ecsv`
   - a quadratic wavelength-to-pixel mapping is estimated with a RANSAC-like search
   - this provides a reference mapping between known wavelengths and mean-spectrum peaks

5. For each detector output:
   - detect peaks in the extracted spectrum
   - refine peak positions to sub-pixel precision using a local logarithmic parabola
   - match the output peaks to the mean-spectrum peaks using a constrained quadratic grid search
   - combine that result with the global line identification to obtain `(x, y, wavelength)` samples

6. Fit a global 2D wavelength solution:
   - fit `wavelength = f(x, y)` with the recipe's cubic polynomial
   - reject outliers iteratively using residual clipping
   - evaluate the fitted polynomial on the traced output coordinates

### Products

- `wave.npz`: `wavelength_map_um`

### Main assumptions

- the detector-to-wavelength mapping is smooth and well represented by a low-order polynomial
- line ordering is preserved across outputs

## 3. Calibration-spectrum extraction and common-grid interpolation

Implemented in [planetes/recipes/preproc.py](/Users/zexwu/zexwu_lib/planetes_pipe/planetes/recipes/preproc.py).

### Inputs

- `flat` product:
  - `dark_map_adu`
  - sparse extraction coordinates
  - `flat_map_adu`
- `wave` product:
  - `wavelength_map_um`
- calibration cubes:
  - one P2VM cube per baseline
  - one photometric flat cube per telescope
  - optional `phase_reference` cube

### Procedure

1. Define a common wavelength grid:
   - `wl_grid = linspace(min_wave, max_wave, nwave)`

2. Extract the flat spectrum for every output:
   - use the sparse extraction masks on `flat_map`

3. Interpolate from pixel space to wavelength space:
   - interpolation is done in wavenumber, not wavelength
   - this uses `wavelength_map_um` and the target wavelength grid

4. For each input calibration cube:
   - subtract `dark_map`
   - extract all outputs with the sparse masks
   - divide by the flat spectrum in detector space
   - interpolate to the common wavelength grid
   - multiply by the interpolated flat response to keep the original flux scale

5. Identify which detector outputs belong to which telescope:
   - use the photometric telescope flats
   - for each telescope, compute the mean extracted flux per output
   - assign the brightest half of the outputs to that telescope

6. Identify which detector outputs belong to which baseline:
   - use the baseline calibration cubes
   - for each baseline, compute the temporal standard deviation per output
   - assign the four most modulated outputs to that baseline

7. Cross-check the mapping:
   - the outputs inferred from telescope intersections must match the outputs inferred from baseline modulation

### In-memory result

- `telescope_spectra`: extracted telescope calibration spectra
- `baseline_spectra`: extracted baseline calibration spectra
- `phase_reference`: extracted optional phase-reference spectra
- `baseline_regions`, `baseline_telescopes`
- `wavelength_um`

These arrays form `CalibrationSpectra` and are passed directly to `run_p2vm`;
there is no preprocessing product on disk.

### Main assumptions

- a telescope photometric flat illuminates all outputs containing that telescope
- a baseline calibration cube produces strong temporal modulation only in the four ABCD outputs of that baseline

## 4. `p2vm`: pixel-to-visibility matrix calibration

Implemented in [planetes/recipes/p2vm.py](/Users/zexwu/zexwu_lib/planetes_pipe/planetes/recipes/p2vm.py).

### Inputs

- in-memory `CalibrationSpectra`:
  - `telescope_spectra`
  - `baseline_spectra`
  - optional `phase_reference`
  - `baseline_regions`
  - `baseline_telescopes`
  - `wavelength_um`

### Procedure

1. For each baseline, isolate its four ABCD outputs.

2. Estimate fringe phase and amplitude by ellipse fitting:
   - form transformed coordinates from the four outputs:
     - `X = C - A`
     - `Y = D - B`
   - for each wavelength channel, fit a conic model that maps the cloud onto a unit circle
   - derive:
     - `phase(frame, wave)`
     - `visamp(frame, wave)`

3. Estimate OPD and group delay for that baseline:
   - use the middle 50% of the band
   - compute OPD from the phase slope
   - compute group delay from coherent-flux phasor summation
   - remove the OPD zero-point by fitting OPD against group delay

4. Fit the visibility-to-pixel coefficients:
   - for each wavelength, fit each output as
     - `c + a cos(phi) + b sin(phi)`
   - store:
     - DC term
     - coherence amplitude `sqrt(a^2 + b^2)`
     - coherence phase `atan2(b, a)`

5. Fit the photometric transmission terms:
   - for each telescope, take the median photometric response across frames

6. Normalize the model:
   - phase normalization: set the A output as the phase reference within each baseline
   - transmission normalization: normalize telescope transmission terms
   - coherence normalization: scale fringe terms by the geometric mean of the contributing photometric fluxes

7. Convert the polar fringe representation into Cartesian form:
   - real term: `coh * cos(phase)`
   - imaginary term: `coh * sin(phase)`

8. Invert the per-wavelength V2PM:
   - compute the pseudoinverse for every wavelength channel
   - this yields the final P2VM

9. Optional phase correction:
   - if `phase_reference` is available, apply an additional phase-reference correction and rebuild the P2VM

The V2PM is persisted through its physical calibration terms: photometric
transmission, coherence amplitude, and coherence phase. The recipe expands
these into a Cartesian matrix only while computing the per-wavelength P2VM.

### Products

- `p2vm`
- `wavelength_um`
- `baseline_regions`
- `baseline_telescopes`
- `transmission`
- `coherence_amplitude`
- `coherence_phase_rad`
- `opd_um`
- `group_delay_um`
- `ellipse_phase_rad`
- `ellipse_amplitude`

### Main assumptions

- each baseline is encoded as four ABCD outputs
- the coherent signal at fixed wavelength traces an ellipse in the transformed ABCD plane
- the instrument response is separable into photometric terms plus sinusoidal fringe terms

## Summary of calibration flow

The calibration steps are chained as follows:

1. `flat`
   - detector background, bad pixels, and output traces
2. `wave`
   - detector-coordinate to wavelength mapping
3. calibration-spectrum extraction
   - extract and interpolate all calibration cubes onto a common spectral grid
4. `p2vm`
   - derive the visibility-to-pixel model and its inverse

After these steps, science reduction can consume:

- sparse extraction geometry from `flat`
- wavelength calibration from `wave`
- inversion matrices from `p2vm`

Only `flat.npz`, `wave.npz`, and `p2vm.npz` are persisted by calibration;
extracted calibration spectra remain in memory.

## Persisted array contract

Axis names below describe the current scientific meaning and will become the
input to the FITS product schema.

| Product | Field | Axes | Unit / meaning |
| --- | --- | --- | --- |
| Flat | `profile_x_pixels` | output, sample | detector pixel; `-1` is invalid |
| Flat | `profile_y_pixels` | output, sample | detector pixel; `-1` is invalid |
| Flat | `flat_map_adu` | detector-y, detector-x | ADU |
| Flat | `dark_map_adu` | detector-y, detector-x | ADU |
| Flat | `bad_pixel_flags` | detector-y, detector-x | bit flags: hot=1, dead=2, noisy=4 |
| Flat | `trace_y_pixels` | output, detector-x | detector pixel |
| Wave | `wavelength_map_um` | output, detector-x | micrometre |
| P2VM | `transmission` | output, telescope, wavelength | normalized photometric V2PM term |
| P2VM | `p2vm` | data-channel, output, wavelength | inverse of the expanded Cartesian V2PM |
| P2VM | `wavelength_um` | wavelength | micrometre |
| P2VM | `coherence_amplitude` | output, baseline, wavelength | normalized amplitude, including cross-talk terms |
| P2VM | `coherence_phase_rad` | output, baseline, wavelength | radian; includes optional phase-reference correction |
| P2VM | `opd_um` | baseline, frame | micrometre |
| P2VM | `group_delay_um` | baseline, frame | micrometre |
| P2VM | `ellipse_phase_rad` | baseline, frame, wavelength | radian |
| P2VM | `ellipse_amplitude` | baseline, frame, wavelength | dimensionless unit-circle radius |
| Reduced | `photometric_flux_adu` | frame, telescope, wavelength | ADU |
| Reduced | `coherent_flux_adu` | frame, baseline, wavelength | complex ADU |
| Reduced | `visibility_amplitude` | frame, baseline, wavelength | dimensionless |
| Reduced | `flags` | frame, baseline, wavelength | boolean; true means invalid |
| Reduced | `wavelength_um` | wavelength | micrometre |
| Reduced | `group_delay_um` | frame, baseline | micrometre |
| Reduced | `differential_phase_deg` | frame, baseline, wavelength | degree |

`baseline_regions` records the four detector outputs in A, C, B, D order;
`baseline_telescopes` records the two telescope indices for each baseline.
