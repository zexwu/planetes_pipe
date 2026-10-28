# Scientific methods in PLANETES 0.10.0

This note describes the retained numerical operations. Public arrays, units,
FITS encoding and parameters are defined in [the product contract](PRODUCT_CONTRACT.md).
Each step file contains its file entry point, parameters and array calculations.
Python, PyEsoRex and the GUI call the same functions; CPL imports stay in recipes/.

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

Photometric telescope flats identify illuminated outputs; modulation in baseline
cubes identifies four outputs per baseline. Their intersections must agree.
The resulting P2VMSpectra stays in memory, without a preprocessing file.
The main dark applies to telescope and baseline cubes. A phase reference has
its independently selected dark, or uses the main dark when none is supplied.

## P2VM (`p2vm.py`)

`fit_p2vm` follows baseline ellipse/OPD/response fitting, telescope transmission,
response normalization, V2PM inversion, and optional phase-reference correction.
The calculation blocks are named helpers in the same file. The normalization and
per-wavelength inversion order are retained. Fixed telescope pairs are defined
once in `instrument.BASELINE_PAIRS`, while detector output mapping is inferred
from the calibration data. `keep_diagnostics=True` retains scan OPD/GD, ellipse
phase/radius, fitted DC and residual RMS; the default discards these diagnostics
after computing the same response and inverse.

`fit_abcd_ellipse` returns normalized ellipse radius. Its persisted diagnostic
field remains `ellipse_amplitude` for compatibility. In the phase-reference helper,
`telescope_phase_rad` expresses the original reference-channel phase in radians;
it is not a separately measured path length.

For each baseline, use its A,C,B,D detector outputs. Ellipse fitting operates on
`X=C-A`, `Y=D-B` at each wavelength to estimate phase and radius. The middle
half of the band supplies OPD phase slope and coherent-sum group delay. Fit
OPD against GD to remove the OPD zero point.

At each wavelength, fit each of all 24 detector outputs as
`c + a*cos(phi) + b*sin(phi)`. Compute the existing fitted DC coefficient and residual ADU RMS without refitting;
retain them only when diagnostics are requested.
Persist coherence amplitude `sqrt(a²+b²)` and
phase `atan2(b,a)`, including cross-talk outputs. Fit photometric transmission
from telescope flats, apply the original transmission/coherence normalizations,
and set the A output's phase reference within each baseline.

Expand these physical terms transiently into the Cartesian V2PM and compute
its per-wavelength pseudoinverse. Persist the inverse and physical terms;
there is no duplicate Cartesian matrix product. Optional phase-reference
correction rebuilds the P2VM and records whether it was actually applied.

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
frames. GUI binning affects only display.

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
[CHANGE_REPORT](CHANGE_REPORT.md). Version 0.8.0 retains the numerical definitions
and per-wavelength pseudoinverse. Physically invalid mappings and rank-deficient
V2PM matrices fail locally; rank uses NumPy's default numerical criterion.


## Subsequent scientific stages

The implemented chain stops at per-frame instrumental P2VMRED. Averaging needs
explicit estimators based on coherent flux, photometry, selected samples and noise;
it cannot be defined by simply taking the mean of the current VISAMP arrays.
Transfer-function calibration then needs astronomical calibrator observations and
a source model. The optional instrumental phase reference in P2VM has a separate
role. These stages are documented here without introducing unimplemented modules.
