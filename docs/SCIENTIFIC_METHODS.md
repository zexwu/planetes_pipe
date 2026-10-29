# Scientific methods in PLANETES 0.14.0

This note describes the retained numerical operations. Public arrays, units,
FITS encoding and parameters are defined in [the product contract](PRODUCT_CONTRACT.md).
Each step file contains its file entry point, parameters and array calculations.
Python and PyEsoRex call the same functions; CPL imports stay in recipes/.

## Dark statistics (`dark.py`)

Read the complete raw stack in 16-frame float64 chunks. For each chunk compute
its mean and centered sum of squares M2; merge chunks using the difference
between their means (Chan's parallel variance formula). Return the mean and
`sqrt(M2 / n_frames)`, the population RMS. This avoids subtracting large second
moments. Complete calibration requires at least two frames. The algorithm does
not select or convert dtypes; the I/O boundary supplies calculation arrays.

The independent BAD product uses the spatial median of the mean image and of
the temporal RMS. Mean values above/below median mean ± threshold × median RMS
set hot/dead bits 1/2; RMS above threshold × median RMS sets noisy bit 4.
The default threshold remains 5. This RMS includes whatever temporal variations
are present in the stack; it is not independently calibrated readout noise.
A single frame returns Dark with n_frames=1 and temporal_rms_adu=None; a
loaded 2-D image has unknown frame count. Unknown RMS is never replaced by zero.
Subtraction only consumes mean_adu; BAD detection requires measured temporal RMS.

## Flat and traces (`flat.py`)

Subtract the explicitly selected dark mean from each telescope's mean flat and average
those four maps. Collapse along dispersion, select the strongest 24 spatial
peaks, trace a Gaussian centroid column by column, then smooth with the retained
median filter and cubic fit. Gaussian iteration counts and arithmetic are unchanged.

Accept pixels under the original strict inequality
`abs(y - trace_y) < aperture_width_pixels / 2`, excluding the explicitly selected BAD map. Sparse X/Y arrays
retain the original accumulation order. Paired (-1,-1) entries represent invalid
pixels and padding when different outputs have different sample counts. The
flat product contains response, geometry and retained Gaussian fit sigmas;
DARK statistics and BAD flags are separate products. Fit sigmas remain available for inspection; the extraction aperture remains aperture_width_pixels.

## Wavelength solution (`wavelength.py`)

Subtract the selected lamp dark from the complete mean lamp image and extract
using flat's geometry. Use the mean of the last four outputs as the reference
spectrum. Detect lines above the retained median/MAD threshold and match to the
30 laboratory lines with the original quadratic RANSAC search. Refine individual
output line positions using a logarithmic parabola and match them to the reference
with the constrained quadratic grid search.

Fit the global cubic `wavelength = f(detector_x, trace_y)` with the retained
iterative residual clipping. Evaluate at each traced output. The result is a
24 × detector-x wavelength map in metres. Smooth mapping and preserved line order
are assumptions; coverage alone is not a wavelength-accuracy measurement.

## Extraction and interpolation (`extraction.py`)

Prepare sparse geometry and flat response once. Subtract the explicitly selected
dark once, accumulate selected pixels, divide the extracted spectrum by the
flat spectrum in detector coordinates, interpolate in wavenumber, and multiply
by the interpolated flat response to preserve the original flux scale.

P2VM calibration defines its explicit target grid (default 1.15e-6–1.30e-6 m, 151
channels). Science and live processing consume this saved grid. Every output
must cover it fully; detector dimensions, sparse geometry and wavelength ordering
are checked before extraction. Calibration files are selected explicitly by the caller. A coverage edge tolerates
only the float32 storage rounding bound (`2**-24` relative); larger gaps fail.
Within that encoding allowance, interpolation retains its endpoint behavior.

File workflows receive explicit telescope/baseline path mappings. SOF shutter
identification is performed by the recipe before calling these workflows. Each
file loop reads its metadata, checks the selected calibrations and extracts its
spectra; it does not convert paths into nested path/metadata tuples.

Photometric telescope flats identify illuminated outputs; modulation in baseline
cubes identifies four outputs per baseline. Their intersections must agree.
The resulting P2VMSpectra stays in memory, without a preprocessing file.
The main dark applies to telescope and baseline cubes. A phase reference has
its independently selected dark, or uses the main dark when none is supplied.

## P2VM (`p2vm.py`)

[P2VM flow diagram](p2vm_flow.png) shows the calculation and file boundaries.
`fit_p2vm` reads as five stages: scan response fitting, photometric transmission,
response normalization, V2PM inversion, and optional instrumental phase reference.
Array shape/identity checks sit at the array entry boundary. File reading and
output identification remain in `make_p2vm` / `extract_p2vm_spectra`, below the
scientific functions. All use the fixed pairs in `instrument.BASELINE_PAIRS`.

The scan routine explicitly follows ellipse -> OPD/GD -> response fitting for each
baseline. Its optional diagnostic dictionary supplies the six existing product
fields; no new product class or validation framework is involved.
`keep_diagnostics=True` retains OPD/GD, ellipse phase/radius, DC and residual RMS.
It changes only retention, not fitting. The persisted radius field keeps its
existing name `ellipse_amplitude`.

For each physical baseline, the four outputs are A,C,B,D. At each wavelength fit
`X=C-A`, `Y=D-B` with the same pseudoinverse and a unit right-hand side. The old
`envs` argument was always one and was not residual weighting; it is removed.
Unwrap `atan2(Y_fit,X_fit)` along frames. For ideal intensities
`I_q=c+A*cos(phi+[0,pi,pi/2,3*pi/2]_q)`, this ellipse phase decreases with phi.
Use the central half-band to form the mean `phase*lambda/(2*pi)` scan OPD and
coherent-sum GD. Remove the intercept of the linear OPD-versus-GD fit. Both use m.

Fit all 24 outputs, including cross-talk, as `c+a*cos(phi)+b*sin(phi)` with
`phi=2*pi*OPD/lambda`. The design columns remain `(1,sin(phi),cos(phi))`.
The amplitude is `sqrt(a*a+b*b)` and the response phase is `atan2(b,a)`, giving
`c+amplitude*cos(phi-phase)`. Amplitude, DC and residual RMS are ADU at this stage;
returned response arrays have axes `(output, baseline, wavelength)`.

Normalization retains its existing order and arithmetic:

1. Within each physical output quartet, subtract its A output phase from **all
   six** baseline response components, including cross-talk.
2. The frame-median telescope spectra first have their global telescope totals
   removed, then the transmission mean across outputs is set to one at each
   telescope/wavelength. Both divisions remain to preserve floating arithmetic.
3. Solve transmission against each baseline's DC terms with `pinv`; divide its
   amplitude by `sqrt(P_i*P_j)`. Nonfinite/nonpositive products fail locally.

V2PM has axes `(24 outputs, 16 data channels, wavelength)`: columns T4,T3,T2,T1,
then six real and six imaginary coherent components in 43,42,41,32,31,21 order.
The coherent columns are amplitude*cos(phase) and amplitude*sin(phase); the whole
matrix is divided by 24. Require numerical rank 16, then compute the original
per-wavelength `pinv`. No new condition-number rejection is added.

The optional phase reference is applied in four explicit stages:

1. Reconstruct reference coherent flux with the initial P2VM. Use baselines
   43,42,41 to obtain central-channel phases relative to T4 (fixed phase zero),
   using GD to retain the existing phase branch. Subtract each telescope's
   temporal mean. These are phases in radians, not path lengths.
2. Scale telescope phase differences by `lambda_center/lambda`, rotate each
   frame, then **sum** coherent flux over frames.
3. Measure GD of that integrated coherence, construct the T4-anchored telescope
   phasors, and apply the existing residual chromatic-slope correction.
4. Subtract the integrated reference phase only from each detector output's
   **own physical baseline** response. Rebuild and invert V2PM.

The fitting function receives reference spectra explicitly, or None to skip this
stage. The script or recipe selects the reference file and its DARK and performs
extraction first. The recipe retains its phase-correction switch and explicitly
selects the main DARK when no PHASE_DARK_CALIB is supplied. Disabled or absent
references are not read and their optional DARK is excluded from DFS used inputs.

The original ellipse thresholds (`1e-9` for scaling; nonpositive quadratic terms
replaced by `1e-6`) remain unchanged. On the regression bench sequence, none of
906 fits (six baselines x 151 channels) used these branches. This does not define
a failure policy for other data: an all-zero scan still returns zero phase/radius
through the old fallbacks. Deciding which channels to reject is a separate
scientific change requiring an explicit criterion.

## Instrumental science and live frames (`reduction.py`)

Apply P2VM to extracted spectra to obtain four photometric channels and six
complex coherent channels. Use the retained visibility normalization and
invalid-data rule: nonpositive/nonfinite photometric products or nonfinite
coherent flux yield NaN and flags. Full reduction also computes coherent-sum
GD with Newton refinement and the original differential phase.

Default coarse GD ranges are ±30e-6 m for calibration and ±100e-6 m for science,
with the retained physical 1e-8 m step. Newton refinement is unconstrained by these
initialization bounds. Real-time reduction can skip GD/phase diagnostics while
sharing extraction, normalization and flags. It always receives raw unbinned
frames. Display binning belongs to the consuming application.

Flux is instrumental ADU. No new uncertainty model, transfer-function correction,
closure-phase definition or OIFITS/Phase 3 product is introduced here.

## File representations and QC

Floating input is adapted to float64/complex128 at the I/O or live-frame boundary;
scientific arrays stay at that precision. New floating files use float32/complex64.
Integers/flags retain their types. Precision is defined in `io/hdu.py`, not inside
the numerical steps. This changes rounding compared with 0.9.0; the report
separates calculation changes from calibration-file and final-output rounding.

Wavelength, OPD and GD use metres throughout calculations and files. There is no
micrometre/metre conversion or retained-representation cache in the numerical path.
Profile centers and valid sparse pixel indices are one-based on disk, with paired
(-1,-1) invalid sentinels retained. QC in [QC_PARAMETERS](QC_PARAMETERS.md) is
descriptive and changes no weights, masks or rejection. GRAVITY's existing UM QC
cards are converted only when writing the header.

The 0.5.0 migration to metre arithmetic introduced floating-point rounding
differences documented in the historical report linked from
[CHANGE_REPORT](CHANGE_REPORT.md). Version 0.11.0 retains the numerical definitions
and per-wavelength pseudoinverse. Physically invalid mappings and rank-deficient
V2PM matrices fail locally; rank uses NumPy's default numerical criterion.


## Subsequent scientific stages

The implemented chain stops at per-frame instrumental P2VMRED. Averaging needs
explicit estimators based on coherent flux, photometry, selected samples and noise;
it cannot be defined by simply taking the mean of the current VISAMP arrays.
Transfer-function calibration then needs astronomical calibrator observations and
a source model. The optional instrumental phase reference in P2VM has a separate
role. These stages are documented here without introducing unimplemented modules.
