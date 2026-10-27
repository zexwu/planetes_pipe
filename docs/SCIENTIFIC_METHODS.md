# Scientific methods in PLANETES 0.4.1

This note describes the retained numerical operations. Public arrays, units,
FITS encoding and parameters are defined in [the product contract](PRODUCT_CONTRACT.md).
File tasks live in tasks.py. Numerical modules accept arrays and scientific objects,
without FITS or CPL dependencies. Python recipes and the GUI share those calculations.

## Dark statistics (`calibration/dark.py`)

Read the complete raw stack, accumulate sum and squared sum in float64, and
return float32 temporal mean and population RMS. The variance is
`max(E[x²] - E[x]², 0)`. Complete calibration requires at least two frames.

The independent BAD product uses the original bad-pixel rule uses the spatial median of the mean image and of
the temporal RMS. Mean values above/below median mean ± threshold × median RMS
set hot/dead bits 1/2; RMS above threshold × median RMS sets noisy bit 4.
The default threshold remains 5. This RMS includes whatever temporal variations
are present in the stack; it is not independently calibrated readout noise.
A subtraction-only DarkImage carries neither an RMS nor inferred BPM.

## Flat and traces (`calibration/flat.py`)

Subtract the explicit complete dark from each telescope's mean flat and average
those four maps. Collapse along dispersion, select the strongest 24 spatial
peaks, trace a Gaussian centroid column by column, then smooth with the retained
median filter and cubic fit. Gaussian iteration counts and arithmetic are unchanged.

Accept pixels under the original strict inequality
`abs(y - trace_y) < profile_width / 2`, excluding the explicitly selected BAD map. Sparse X/Y arrays
retain the original accumulation order. Paired (-1,-1) entries represent invalid
pixels and padding when different outputs have different sample counts. The
flat product contains response, geometry and retained Gaussian fit sigmas;
DARK statistics and BAD flags are separate products. Fit sigmas are descriptive
QC; the extraction aperture remains profile_width.

## Wavelength solution (`calibration/wavelength.py`)

Subtract the selected lamp dark from the complete mean lamp image and extract
using flat's geometry. Use the mean of the last four outputs as the reference
spectrum. Detect lines above the retained median/MAD threshold and match to the
30 laboratory lines with the original quadratic RANSAC search. Refine individual
output line positions using a logarithmic parabola and match them to the reference
with the constrained quadratic grid search.

Fit the global cubic `wavelength = f(detector_x, trace_y)` with the retained
iterative residual clipping. Evaluate at each traced output. The result is a
24 × detector-x wavelength map in um. Smooth mapping and preserved line order
are assumptions; coverage alone is not a wavelength-accuracy measurement.

## Extraction and interpolation (`extraction.py`)

Prepare sparse geometry and flat response once. Subtract the explicitly selected
dark once, accumulate selected pixels, divide the extracted spectrum by the
flat spectrum in detector coordinates, interpolate in wavenumber, and multiply
by the interpolated flat response to preserve the original flux scale.

P2VM calibration defines its explicit target grid (default 1.15–1.30 um, 151
channels). Science and live processing consume this saved grid. Every output
must cover it fully; parent flat/wave relations and wavelength ordering are
checked before extraction. No extension of boundary values is accepted.

Photometric telescope flats identify illuminated outputs; modulation in baseline
cubes identifies four outputs per baseline. Their intersections must agree.
The resulting CalibrationSpectra stays in memory, without a preprocessing file.
The main dark applies to telescope and baseline cubes. A phase reference has
its independently selected dark, or records reuse of the main dark.

## P2VM (`calibration/p2vm.py`)

For each baseline, use its A,C,B,D detector outputs. Ellipse fitting operates on
`X=C-A`, `Y=D-B` at each wavelength to estimate phase and radius. The middle
half of the band supplies OPD phase slope and coherent-sum group delay. Fit
OPD against GD to remove the OPD zero point.

At each wavelength, fit each of all 24 detector outputs as
`c + a*cos(phi) + b*sin(phi)`. Retain the existing fitted DC coefficient and residual ADU RMS without refitting.
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

Default coarse GD ranges are ±30 um for calibration and ±100 um for science,
with the retained 0.01 um step. Newton refinement is unconstrained by these
initialization bounds. Real-time reduction can skip GD/phase diagnostics while
sharing extraction, normalization and flags. It always receives raw unbinned
frames. GUI binning affects only display.

Flux is instrumental ADU. No new uncertainty model, transfer-function correction,
closure-phase definition or OIFITS/Phase 3 product is introduced here.

## File representations and QC

Internal wavelength/OPD/GD units remain um. Corresponding FITS quantities use m;
profile centers and valid sparse pixel indices are one-based on disk, with paired
(-1,-1) invalid sentinels retained. The codec remembers original storage quantities
only while decoded scientific state is unchanged, preventing repeated conversion
drift. These floating-point boundary conversions are evaluated separately from
same-array core equivalence. QC in [QC_PARAMETERS](QC_PARAMETERS.md) is descriptive:
it changes no fitting weights, masks, rejection or scientific defaults.
