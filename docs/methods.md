# Scientific methods

This note describes the retained numerical operations. Public arrays, units,
FITS encoding, QC and parameters are defined in [products.md](products.md).
Each step file contains its file entry point, parameters and array calculations.
Python and PyEsoRex call the same functions; CPL imports stay in eso/.

## Dark statistics (`steps/dark.py`)

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

## Flat and traces (`steps/flat.py`)

Subtract the explicitly selected dark mean from each telescope's mean flat and average
those four maps. Collapse along dispersion and select the most prominent 24 (or 25,
with the metrology trace) spatial peaks, whatever their height: telescope fluxes can
differ by 19:1. Average the map over 21 columns, so faint parts can be followed, and
trace a Gaussian centre column by column from the brightest column, starting from a
recentred centroid. Then smooth with the retained median filter and cubic fit.

The raw mean image and selected DARK must have exactly the same detector shape,
including when DARK setup metadata is unknown; NumPy broadcasting is not accepted.
The cubic fit uses columns with a center error < 0.5 pixel and an accepted width,
after a three-column median filter, whose signal exceeds five background MAD sigmas in
a window following the trace (traces are tilted, and leave any fixed strip). A
fraction of each trace's peak is not used: the range would depend on its brightest
feature. The fit and the saved x range cover the longest run of such columns, across
gaps of up to 20 columns. These are fixed choices, not derived per exposure.

Accept pixels under the original strict inequality
`abs(y - trace_y) < aperture_width_pixels / 2`, excluding the explicitly selected BAD map. Sparse X/Y arrays
retain the original accumulation order. Paired (-1,-1) entries represent invalid
pixels and padding when different outputs have different sample counts. The
flat product contains response, geometry and retained Gaussian fit sigmas;
DARK statistics and BAD flags are separate products. Fit sigmas remain available for inspection; the extraction aperture remains aperture_width_pixels.

## Wavelength solution (`steps/wave.py`)

Subtract the selected lamp dark from the complete mean lamp image and extract
using flat's geometry. Use the mean of the last four outputs as the reference
spectrum. Detect lines above the retained median/MAD threshold and match to the
30 laboratory lines with the original quadratic RANSAC search. Refine individual
output line positions using a logarithmic parabola and match them to the reference
with the constrained quadratic grid search.

The reference quartet is a fixed bench choice, not selected by measured S/N.
The three-sample logarithmic parabola estimates the center of a locally Gaussian
line. Its retained ADU and log-curvature cuts prevent applying that refinement
to weak or flat samples; their numerical values are bench criteria. Laboratory
lines must be increasing because matching indices refer to sorted arrays.

Fit the global cubic `wavelength = f(detector_x, trace_y)` with the retained
iterative residual clipping. Evaluate at each traced output. The result is a
24 × detector-x wavelength map in metres. Smooth mapping and preserved line order
are assumptions; coverage alone is not a wavelength-accuracy measurement.

## Extraction and interpolation (`steps/extraction.py`)

Prepare sparse geometry and flat response once. Subtract the explicitly selected
dark once, accumulate selected pixels, divide the extracted spectrum by the
flat spectrum in detector coordinates, interpolate in wavenumber, and multiply
by the target-grid flat response to preserve the original flux scale. That
response is `1 / interpolate(1 / detector_flat)`, not direct interpolation of the
flat. The interpolation is pointwise in wavenumber, not an integral over bins.

P2VM calibration defines its explicit target grid: the wavelength range covered by
the fitted x range of every output, at the median native pixel wavelengths, limited
further by `wave_min_m` and `wave_max_m` (the bench uses 1100-1270 nm, which leaves out
the SuperK pump near 1064 nm and the faint band edges). Science and live processing
consume this saved grid. Every output must cover it fully; detector dimensions, sparse
geometry and wavelength ordering are checked before extraction. Calibration files are
selected explicitly by the caller.

File workflows receive explicit telescope/baseline path mappings. SOF shutter
identification is performed by the recipe before calling these workflows. Each
file loop reads its metadata, checks the selected calibrations and extracts its
spectra; it does not convert paths into nested path/metadata tuples.

Photometric telescope flats identify illuminated outputs; modulation in baseline
cubes identifies four outputs per baseline. Their intersections must agree.
The resulting P2VMSpectra stays in memory, without a preprocessing file.
The main dark applies to telescope and baseline cubes. A phase reference has
its independently selected dark, or uses the main dark when none is supplied.

## P2VM (`steps/p2vm.py`)


`fit_p2vm` reads as five stages: scan response fitting, photometric transmission,
response normalization, V2PM inversion, and optional instrumental phase reference.
Array shape/identity checks sit at the array entry boundary. File reading and
output identification are in `p2vm.run` / `extract_p2vm_spectra`, before the fit. All use the fixed pairs in `instrument.BASELINE_PAIRS`.

The scan routine explicitly follows ellipse -> OPD/GD -> response fitting for each
baseline. Its optional diagnostic dictionary supplies the six existing product
fields; no new product class or validation framework is involved.
`keep_diagnostics=True` retains OPD/GD, ellipse phase/radius, DC and residual RMS.
It changes only retention, not fitting. The persisted radius field keeps its
existing name `ellipse_amplitude`.

For each physical baseline, the four outputs are A,C,B,D. At each wavelength fit
`X=C-A`, `Y=D-B` with the same pseudoinverse and a unit right-hand side. The old
`envs` argument was always one and was not residual weighting; it is removed.
The ellipse phase is `atan2(Y_fit,X_fit)`. For ideal intensities
`I_q=c+A*cos(phi+[0,pi,pi/2,3*pi/2]_q)`, it decreases with phi. The fringes alone
give each frame's OPD; the delay line's recorded position is not used. On the central
half-band, the coherent-sum GD of `radius*exp(1j*phase)` (`scan_group_delay`: the
scan's centre from all frames over +-1000 um, then each frame within 150 um of it)
does not depend on whole fringes, and fixes them (`unwrap_scan_phase`, relative to
the middle frame). The OPD is then the mean `phase*lambda/(2*pi)`, precise to nm where
the GD is precise to 0.04-0.12 um. Remove the intercept of the linear OPD-versus-GD
fit. Both use m. A scan step spans several fringes, so unwrapping along frames fails,
and the recorded position is wrong by up to 9 fringes (backlash, 2026-11-06). With
`keep_diagnostics`, the P2VM keeps that recorded OPD (`delay_line_opd`) and its report
plots it against the fringe OPD, so the delay line's gain and backlash can be judged.

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
4. Give all baselines the orientation `C_ij = E_i E_j*`, for which the OPD of baseline
   ij falls as the delay line of its second telescope moves out (OPD gain -1). A
   baseline whose scan records a positive gain has its response phases negated, so
   its reduced VISDATA, GD and closure phases follow the same convention. On
   2026-11-05 this is S3-S2 alone, whose B-A then reads 270 deg instead of 90. Without
   it, closure phases containing that baseline do not close.

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
extraction first, with the same extractor as the scans (`p2vm.run`). A supplied PHASE_REFERENCE is
always applied; the main DARK is used when no PHASE_DARK_CALIB is supplied. An
absent reference is not read and is not recorded as an input.

The original ellipse thresholds (`1e-9` for scaling; nonpositive quadratic terms
replaced by `1e-6`) remain unchanged. On the regression bench sequence, none of
906 fits (six baselines x 151 channels) used these branches. This does not define
a failure policy for other data: an all-zero scan still returns zero phase/radius
through the old fallbacks. Deciding which channels to reject is a separate
scientific change requiring an explicit criterion.

## Instrumental science and live frames (`steps/science.py`)

Apply P2VM to extracted spectra to obtain four photometric channels and six
complex coherent channels. Use the retained visibility normalization and
invalid-data rule: nonpositive/nonfinite photometric products or nonfinite
coherent flux yield NaN and flags. Full reduction also computes coherent-sum
GD with Newton refinement and the original differential phase.

The science GD is tracked (`extraction.tracked_group_delay`): each block of 64 frames
finds its fringe over ±1000e-6 m by the summed coherent power of its frames
(`stacked_group_delay`), and each frame is then searched within ±50e-6 m of it with a
1e-8 m step and Newton refinement. A block whose peak is below 20 times the median
power has no fringe in the range; the reduction warns. With at least 64 frames showing
fringes, the dispersion of the data is then removed and every frame searched again, twice
(`dispersion_phase`: the frames, shifted by their GD and mean phase, are summed, and the
phase of the sum is fitted with a 4th-order polynomial in wavenumber whose constant and
linear terms are dropped). Dispersion flattens the top of the delay peak, and the GD of a
noisy frame slides along it: on the bench, baselines with a 5-11 rad bow scattered by
0.5-1.1 um per frame, and by 0.01-0.05 um once it was removed. Without a linear term the
GD keeps its meaning (the group delay at the mean wavenumber). The phase reference of the
P2VM keeps a single ±30e-6 m search.
Real-time reduction shares extraction, normalization and flags, with a 0.5e-6 m step
for speed. It always receives raw unbinned frames. Display binning belongs to the
consuming application.

Errors follow GRAVITY's `gravi_p2vm_reduce`. The variance of an extracted spectrum is
its signal over the gain plus the read noise, the DARK's temporal variance summed over
the aperture; both are scaled and resampled as the signal, neglecting resampling
correlations. It goes through the squared P2VM inverse, neglecting correlations between
outputs, to give FLUXERR and VISERR. CHI2 compares the outputs with the V2PM model of
the reduced channels, with 8 degrees of freedom. SNR is
`|sum VISDATA| / sqrt(sum |VISERR|^2)` over the unflagged wavelengths of a frame. The
gain, 36 e-/ADU, is a photon-transfer fit to the frame differences of the 2026-11-05
flats (34-40 across the four flats); it is a constant, not measured per exposure.

Closure quantities use the bispectrum `C_ij C_jk conj(C_ik)` of telescopes i,j,k, as
GRAVITY, in the descending telescope order 432, 431, 421, 321. T3AMP is the product of
the three VISAMP. T3PHIERR adds the relative errors of the three coherent fluxes in
quadrature, neglecting flux noise in the amplitudes' normalization.

Flux is instrumental ADU. No transfer-function correction or OIFITS/Phase 3 product is
introduced here. TIME assumes one frame per DIT from MJD-OBS, as the camera records no
per-frame timestamps.

## File representations and QC

Calculations use float64/complex128 and products are stored exactly as computed
(since 0.17.0; earlier products were float32). Integers and flags keep their types.
Raw cubes are read as recorded and converted to float64 frame by frame.

Wavelength, OPD and GD use metres throughout calculations and files. There is no
micrometre/metre conversion or retained-representation cache in the numerical path.
Profile centers and valid sparse pixel indices are one-based on disk, with paired
(-1,-1) invalid sentinels retained. QC in [products.md](products.md#qc) is
descriptive and changes no weights, masks or rejection.

The 0.5.0 migration to metre arithmetic introduced floating-point rounding
differences documented in the historical report linked from
[changes.md](changes.md). Version 0.11.0 retains the numerical definitions
and per-wavelength pseudoinverse. Physically invalid mappings and rank-deficient
V2PM matrices fail locally; rank uses NumPy's default numerical criterion.


## Subsequent scientific stages

The implemented chain stops at per-frame instrumental P2VMRED. Averaging needs
explicit estimators based on coherent flux, photometry, selected samples and noise;
it cannot be defined by simply taking the mean of the current VISAMP arrays.
Transfer-function calibration then needs astronomical calibrator observations and
a source model. The optional instrumental phase reference in P2VM has a separate
role. These stages are documented here without introducing unimplemented modules.
