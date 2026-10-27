# PLANETES 0.5.0 product and raw contract (products 4 / raw 3)

Science functions return named Python objects. `products.py` defines their units,
axes and parent relationships; `io/fits.py` maps them to files. `recipes/` binds SOF and
parameters to those ordinary functions and asks CPL DFS to sign/register results.
Numerical code imports neither PyCPL nor Qt. No YAML scheduler is added.

## Reference and adaptations

The fixed reference is GRAVITY **1.9.6**, available locally at
`/Users/zexwu/software/gravi_pipeline/gravity-1.9.6/`; paths below are relative to
that source. These are instrument pipeline conventions, not an assertion that
all GRAVITY keywords belong to an ESO-wide dictionary.

| Reference | Meaning | PLANETES representation and difference |
| --- | --- | --- |
| `gravi/gravi_dfs.h:43–99` | raw/product categories | DARK_RAW, FLAT_RAW, P2VM_RAW, WAVELAMP_RAW, SINGLE_SCI_RAW; DARK, BAD, FLAT, WAVE, P2VM, SINGLE_SCI_P2VMRED |
| `gravi/gravi_dfs.c` parameter constructors | instrument.stage.hyphenated-parameter | explicit `planetes.calib.*` / `planetes.reduce.*` registry; original scientific defaults |
| `gravi/gravi_utils.h:131–140`, `gravi_utils.c:gravi_get_shutter` | physical shutters, GRAVITY ESO INS SHUT11..14/IPAG variants | private PLANETES INS SHUT1..4 ST, physical channels 1..4; no invented GRAVITY hardware |
| `gravi/gravi_pfits.h:41–95` | named SC detector/profile/wave/P2VM extensions | same names where quantities match; one PLANETES camera, no imaginary FT |
| `gravi/gravi_calib.c:1293–1407` | PROFILE_PARAMS CENTERY and Gaussian WIDTH | one-based smoothed trace and retained accepted Gaussian sigma; flat response remains private, not a weighted GRAVITY PROFILE_DATA |
| `gravi/gravi_p2vm.c:125–171,637–639` | output rows with physical response vectors and C_MATRIX | full 24-output TRANSMISSION, COHERENCE, PHASE and fitted DC C_MATRIX; inverse has its own private extension |
| `gravi/gravi_wave.c:gravi_wave_fit_2d` | fitted wavelength surface | lamp-line surface retained; the raw input is WAVELAMP_RAW, not GRAVITY's internal interferometric WAVE_RAW |
| supplied ESO-044156 v8, p24, p47–48 | exposure times, UTC start, detector window | real per-integration readback DIT1 or requested UIT1, actual NDIT, ROI and physical binning |

Physical telescope order is **4,3,2,1**, baselines **43,42,41,32,31,21** and
outputs within each baseline **A,C,B,D**. These PLANETES conventions are retained,
including baseline signs. SC means the sole spectroscopic camera here.

## Raw acquisition contract

An empty primary carries INSTRUME=PLANETES, DPR classification and
`HIERARCH PLANETES FORMAT=3`. A named IMAGING_DATA_SC image contains the original
(frame, detector_y, detector_x) cube in ADU. Dtype and pixel values are preserved;
there is no unconditional int16 cast or lossy floating compression. Files are
published without overwrite. The complete cube is read in 16-frame float32
chunks for the original core arithmetic.

| SOF/raw category | DPR CATG / TYPE / TECH | Open physical shutters |
| --- | --- | --- |
| DARK_RAW | CALIB / DARK / IMAGE | 0 |
| FLAT_RAW | CALIB / FLAT / SPECTRUM | 1 |
| P2VM_RAW | CALIB / P2VM / INTERFEROMETRY | 2 |
| WAVELAMP_RAW | CALIB / WAVE,LAMP / SPECTRUM | 4 |
| SINGLE_SCI_RAW | SCIENCE / OBJECT,SINGLE / INTERFEROMETRY | 4 |

Four explicit boolean `PLANETES INS SHUTn ST` cards refer to **physical** n=1..4.
Configuration sources are readback, command, operator or an explicit legacy-role-map.
`PLANETES_META` stores named RawMetadata, detector setup and original/source records.
It never infers identity from filenames, caller numbering or SOF order.

New captures require exposure seconds and source, detector identity, orientation,
ROI origin and dimensions, physical binning, category and shutters. ROI origins
are zero-based internally and one-based in ESO DET WIN1 STRX/STRY; NX/NY and
BINX/BINY describe sensor geometry, independently of GUI display binning.

Exposure readback uses ESO DET WIN1 DIT1 [s]; command/operator requests use UIT1.
NDIT is the actual cube frame count. Only actual/readback DIT creates EXPTIME as
the sum of unaveraged integration durations; host elapsed time is not EXPTIME.
The established camera exposure_time property is microseconds. Host start/end UTC
are separately sourced in metadata. DATE-OBS/MJD-OBS are emitted only for an actual
camera exposure-start UTC; neither saving time nor GUI refresh rate supplies it.

Legacy data must be standardized into separate files using an explicit adopted
role/shutter map. Width/height come from the actual image, known header settings
are retained with source, and unrecoverable settings stay None. Legacy is explicit;
missing association fields are returned as `provenance.unverified`. Original
EXP_TIME, DATE/TIME and other available header values remain in original metadata;
historical local saving times do not become observation timestamps.

## Products and layouts

All products have empty primary, format 4, ESO PRO CATG/TYPE/SCIENCE and private
`PLANETES CONTENT SHA256`. Every HDU has CHECKSUM/DATASUM. PLANETES_META holds
producer, software version, actual parameters, all source roles, parent scientific
fingerprints, unverified associations and unavailable QC reasons. Known DARK/FLAT
ROI/exposure cards are also exposed in the primary and checked against metadata.
CPL products additionally carry its DFS recipe, parameters, all used frames and
signature. Standalone Python uses the same payload, units, QC and validation.

| Python class / category | Extensions and quantities |
| --- | --- |
| DarkCalibration / DARK | IMAGING_DATA_SC mean ADU; IMAGING_ERR_SC temporal **population** RMS ADU; actual NFRAMES >=2; DetectorSetup |
| BadPixelMap / BAD | IMAGING_DATA_SC integer bits hot=1, dead=2, noisy=4, explicitly declared; mandatory DARK parent |
| FlatCalibration / FLAT | PLANETES_FLAT_IMAGE ADU; PROFILE_PARAMS(REGNAME,CENTERY,optional WIDTH); PLANETES_PROFILE(PIXEL_X,PIXEL_Y); optional PLANETES_FLAT_FLUX ADU |
| WavelengthCalibration / WAVE | WAVE_DATA_SC, one row, DATA1..DATA24 metre vectors; PLANETES_WAVE_FIT(RESIDUAL [m], X/Y [zero-based pixel]) for accepted lines |
| P2VMCalibration / P2VM | P2VM_SC, 24 REGNAME rows; TRANSMISSION (4,nwave), COHERENCE/PHASE/C_MATRIX (6,nwave); PHASE rad and DC C_MATRIX ADU |
| ReducedObservables / SINGLE_SCI_P2VMRED | PLANETES_FLUX: FRAME, physical STA_INDEX, FLUXDATA [ADU]; PLANETES_VIS: FRAME, station pair, VISDATA [complex ADU], VISAMP, FLAG, GDELAY [m], DIFFPHASE [deg] |
| WavelengthReference / LINE_LIST | LINE_LIST(WAVELENGTH [m], ORIGIN), 30 laboratory wavelengths in metres with original ECSV source labels; ECSV SHA in provenance |

FLAT/WAVE contain IMAGING_DETECTOR_SC with REG01..REG24 labels until the output
mapping is measured. P2VM/science use actual baseline-phase labels such as 43-A;
P2VM_SC rows and the detector table must agree. Science rows are frame-major:
four telescope rows or six baseline rows per frame. FRAME and physical STA_INDEX
are positive one-based identifiers, never Python offsets. REGNAME reconstructs
the measured ABCD output mapping independently of dictionary insertion order.

Vector columns preserve float32/float64 or complex64/complex128 precision. FITS
TDIM reverses the per-row Python shape: e.g. TRANSMISSION has (nwave,4), while
COHERENCE/PHASE/C_MATRIX have (nwave,6). Boolean FLAG uses FITS L (true=invalid).
Complex values, NaNs and signed zero are retained. Flat integer coordinates and
trace centers are one-based on disk; paired (-1,-1) sparse sentinels stay -1.

Necessary PLANETES extensions retain scientific quantities with no matching
reference field:

| Extension | Python shape | File unit / reason |
| --- | --- | --- |
| PLANETES_PROFILE | output,sample for each coordinate | pixel; sparse aperture indices, not weighted GRAVITY PROFILE_DATA |
| PLANETES_FLAT_IMAGE / FLAT_FLUX | detector_y,detector_x / telescope | ADU; response and mean-frame total flux |
| PLANETES_WAVELENGTH | wavelength | m; target coordinate without invented effective bandwidth |
| PLANETES_P2VM_INVERSE | 16,24,wavelength | dimensionless; inverse is **not** GRAVITY C_MATRIX |
| PLANETES_OPD / GD | baseline,calibration_frame | m; retained ellipse/scan diagnostics |
| PLANETES_ELLIPSE_PHASE / AMPLITUDE | baseline,frame,wavelength | rad / dimensionless |
| PLANETES_P2VM_FIT_RMS | output,baseline,wavelength | ADU; unweighted fit RMS is not chi-square |
| PLANETES_FLUX / VIS | frame-major table rows | per-frame instrumental observables, without fictitious OIFITS time, errors, bandwidth or target/station tables |

DarkImage, CalibrationSpectra and live VisibilityObservables remain memory-only.
A two-dimensional or single-frame explicitly selected GUI dark is subtraction-only;
all file entry points require a full DARK. Earlier product formats are rejected;
regenerate them from standardized raw. This format does not implement final OIFITS.

## Relationships, hashing and publication

BAD names its DARK; FLAT names DARK and BAD; WAVE names FLAT; P2VM names FLAT
and WAVE. Same-shaped wrong parents fail. At each raw subtraction, known DARK
exposure and detector configuration must match. At extraction, FLAT geometry must
match; science/lamp may have different DIT from flats when using their own matching
DARK. No dark scaling is added. Unknown matching fields only pass for explicitly
legacy data/calibrations and remain recorded as unverified. Exposure matching
permits only four ULP of encoding roundoff, no arbitrary physical tolerance.

Scientific fingerprints identify arrays, units, axes, fixed mappings, parameters,
detector setup and parent fingerprints. Paths, producer and derived QC status do
not change them. `products.fingerprint()` computes this directly; there is no
retained representation or decoded-state cache. `read_product()` verifies the
recorded hash once after decoding, in addition to FITS checksums and unit checks.
Lengths are already metres, so reading and saving never rescale them.

DARK and BAD are fully encoded, signed and read back before either final name is
published. Existing targets are refused. An ordinary second publication failure
removes only links still owned by this operation; concurrent replacements survive.
The two names do not form a crash-atomic filesystem transaction.

## Parameters

Each step declares canonical names and aliases in its own `PARAMETERS` dictionary.
Python function signatures supply defaults. `recipes/parameters.py` only translates
these ordinary definitions into CPL parameters; steps import no CPL code.

| Python keyword | Canonical CPL name | Default |
| --- | --- | --- |
| bad_pixel_threshold | planetes.calib.bad-dark-threshold | 5.0 |
| profile_width | planetes.calib.profile-width | 10.0 pixel full aperture |
| phase_correction | planetes.calib.phase-correction | true |
| wavelength_min_m / max_m / channels | planetes.calib.wavelength-min-m / wavelength-max-m / wavelength-channels | 1.15e-6 / 1.30e-6 m / 151 |
| gd_search_min_m / max_m (P2VM) | planetes.calib.gd-search-min-m / gd-search-max-m | -30e-6 / +30e-6 m |
| gd_search_min_m / max_m (science) | planetes.reduce.gd-search-min-m / gd-search-max-m | -100e-6 / +100e-6 m |

CLI aliases are the final hyphenated component. The coarse GD step remains
1e-8 m; Newton refinement can leave initialization bounds. No clipping rule,
mask, line rejection, weighting or scientific default is changed.

SOF quantities and allowed classes are stated in README and implemented in each
recipe. All entries have RAW/CALIB group and image/table type matching their role;
outputs are final PRODUCT frames. LINE_LIST and PHASE_REFERENCE/PHASE_DARK_CALIB
are documented project roles where no strictly equivalent GRAVITY input exists.
A phase DARK without reference fails; a disabled reference never enters the used
input provenance. Repeated FLAT_RAW/P2VM_RAW tags are required; duplicate physical
shutter combinations or incomplete sets fail.

## Python organization in 0.5.0

DARK, FLAT, WAVE, P2VM and REDUCE are peer step modules at package root. Each owns
its input handling, parameters, scientific functions and returned products. Shared
extraction stays in extraction.py; the P2VM input sequence and output mapping stay
in p2vm.py. Saving uses io/fits.py explicitly. RawMetadata belongs to io/raw.py;
DetectorSetup belongs to instrument.py; products and their fingerprints belong to
products.py. Every CPL import and DFS operation stays inside recipes/.

Checks cover physical associations, array/index safety and external file contracts.
They are not a separate validation workflow. QC reports measurements and missing
reasons without changing scientific cuts. Product format 4 records the new metre
field/parameter names and fingerprints; standardized raw remain format 3.
