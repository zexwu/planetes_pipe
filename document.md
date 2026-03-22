# Calibration Procedures

This note summarizes the algorithms currently used by the calibration stages of the pipeline:

- `flat`
- `wave`
- `preproc --object p2vm`
- `p2vm`

The description follows the implementation in the codebase and is meant to explain what is actually computed, in what order, and with what assumptions.

## 1. `flat`: detector calibration and spectral tracing

Implemented in [recipes/flat.py](/Users/zexwu/zexwu_lib/planetes_pipe/recipes/flat.py).

### Inputs

- dark cube: `conf["calib"]["dark"]`
- one flat cube per telescope: `conf["calib"]["flat"]`

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

- `dark_map`
- `bad_map`
- `flat_map`
- `profile_map`
- sparse extraction coordinates: `profile_xs`, `profile_ys`
- traced centroids: `xs`, `ys`

### Main assumption

Each interferometric output is well approximated by a narrow, smooth trace in the detector plane, and the trace can be followed column-by-column from the flat image.

## 2. `wave`: wavelength calibration

Implemented in [recipes/wave.py](/Users/zexwu/zexwu_lib/planetes_pipe/recipes/wave.py).

### Inputs

- `flat` product:
  - sparse extraction coordinates
  - traced output geometry
- wave lamp cube: `conf["calib"]["wave"]`
- wave dark cube: `conf["calib"]["wave_dark"]`

### Procedure

1. Build the mean wave-lamp image:
   - `wave_img = mean(wave_cube) - mean(wave_dark)`

2. Extract one 1D spectrum per detector output:
   - use the sparse extraction masks from the `flat` step

3. Detect emission lines in a reference spectrum:
   - take the mean of the last four extracted outputs as a reference spectrum
   - detect peaks above `median + sigma * MAD`

4. Match detected peaks to a laboratory line list:
   - a list of known line wavelengths is hardcoded in the recipe
   - a quadratic wavelength-to-pixel mapping is estimated with a RANSAC-like search
   - this provides a reference mapping between known wavelengths and mean-spectrum peaks

5. For each detector output:
   - detect peaks in the extracted spectrum
   - refine peak positions to sub-pixel precision using a local logarithmic parabola
   - match the output peaks to the mean-spectrum peaks using a constrained quadratic grid search
   - combine that result with the global line identification to obtain `(x, y, wavelength)` samples

6. Fit a global 2D wavelength solution:
   - fit `wavelength = f(x, y)` with a polynomial of configurable degree
   - reject outliers iteratively using residual clipping
   - evaluate the fitted polynomial on the traced output coordinates

7. Fit an alternate aberration-based model:
   - keep only lines matched in all outputs
   - fit a 1D dispersion relation along the mean trace
   - fit a 2D polynomial correction in detector coordinates for the line shifts
   - combine the two to build an aberration-corrected wavelength map

### Products

- `wave.npz`
  - `wave_map`
  - QC summary and region match counts
- `wave_aberr.npz`
  - alternate `wave_map`
  - QC summary and region match counts

### Main assumptions

- the detector-to-wavelength mapping is smooth and well represented by a low-order polynomial
- line ordering is preserved across outputs
- the aberration term is a smooth low-order correction to a dominant 1D dispersion law

## 3. `preproc --object p2vm`: extraction and common-grid interpolation

Implemented in [recipes/preproc.py](/Users/zexwu/zexwu_lib/planetes_pipe/recipes/preproc.py).

### Inputs

- `flat` product:
  - `dark_map`
  - sparse extraction coordinates
  - `flat_map`
- `wave` product:
  - `wave_map`
- calibration cubes:
  - one P2VM cube per baseline
  - one photometric flat cube per telescope
  - optional `wavesc` cube

### Procedure

1. Define a common wavelength grid:
   - `wl_grid = linspace(min_wave, max_wave, nwave)`

2. Extract the flat spectrum for every output:
   - use the sparse extraction masks on `flat_map`

3. Interpolate from pixel space to wavelength space:
   - interpolation is done in wavenumber, not wavelength
   - this uses the detector `wave_map` and the target `wl_grid`

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

### Products

- `spec_tel`: extracted telescope calibration spectra
- `spec_bsl`: extracted baseline calibration spectra
- `spec_wavesc`: extracted optional phase-reference spectra
- `spec_flat`: flat response on the common grid
- `tel_regs`, `bsl_regs`
- `bsl_to_reg`, `bsl_to_tel`
- `wl_grid`

### Main assumptions

- a telescope photometric flat illuminates all outputs containing that telescope
- a baseline calibration cube produces strong temporal modulation only in the four ABCD outputs of that baseline

## 4. `p2vm`: pixel-to-visibility matrix calibration

Implemented in [recipes/p2vm.py](/Users/zexwu/zexwu_lib/planetes_pipe/recipes/p2vm.py).

### Inputs

- `preproc("p2vm")` product:
  - `spec_tel`
  - `spec_bsl`
  - optional `spec_wavesc`
  - `spec_flat`
  - `bsl_to_reg`
  - `bsl_to_tel`
  - `wl_grid`

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
   - if `spec_wavesc` is available, apply an additional phase-reference correction and rebuild the P2VM

### Products

- `v2pm`
- `p2vm`
- `opd_per_baseline`
- `gd_per_baseline`
- `wl_grid`
- `bsl_to_reg`
- `bsl_to_tel`
- `ellipse_results`

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
3. `preproc --object p2vm`
   - extract and interpolate all calibration cubes onto a common spectral grid
4. `p2vm`
   - derive the visibility-to-pixel model and its inverse

After these steps, science reduction can consume:

- sparse extraction geometry from `flat`
- wavelength calibration from `wave`
- common-grid flat response and baseline/output mapping from `preproc`
- inversion matrices from `p2vm`
