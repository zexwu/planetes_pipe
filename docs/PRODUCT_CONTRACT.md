# Scientific objects, FITS schema 2 and parameters

## Boundaries

The ordinary file tasks read complete HDU-1 raw cubes in 16-frame float32 chunks
and return in-memory dataclasses. Array-level functions compute the same science
without file paths. Python callers explicitly save. ESO adapters bind SOF roles
and parameters, call those file tasks and register signed products with CPL DFS.
The GUI sends raw, unbinned frames to `FrameReducer` and renders the returned flags.

## Primary header and common representation

Every persisted product has an empty primary array, `PLTSCHEM=2`, `PLTVERS`, a
scientific SHA256 `PLTHASH`, `ESO PRO CATG`, `ESO PRO TYPE`, `ESO PRO SCIENCE`,
`PLTTELS=4,3,2,1`, `PLTBSLS=43,42,41,32,31,21` and `PLTABCD=A,C,B,D`.
Reduced products additionally set `PLTLEVEL=INSTRUMENTAL`.
Every HDU requires valid CHECKSUM/DATASUM. Each image extension has BUNIT and
PLTAXES; PLTAXES describes **Python array order**, the reverse of FITS NAXIS order.
Empty BUNIT denotes dimensionless values or flags.

Image extension names are the uppercase dataclass field names. Complex
COHERENT_FLUX_ADU is split into `_REAL` and `_IMAG`; reading assigns components
separately to preserve signed zero and NaN bits. FLAGS uses uint8 0/1 on disk,
bool in memory, where true means invalid. A complete dark records integer
NFRAMES >= 2 in the primary header. A DarkImage has no NFRAMES, RMS or BPM.

P2VM/reduced include BASELINE_MAP, with six rows and columns BASELINE_INDEX (J),
TELESCOPE_INDICES (2J), OUTPUT_INDICES (4J). INDEX0=T explicitly declares zero-based
indices. Telescope indices address 4,3,2,1; output indices within each row are
ordered A,C,B,D, also recorded in PLTABCD. Mapping content is checked against the
fixed instrument conventions.

The static reference uses a LINE_LIST binary table: WAVELENGTH_UM (D, unit um)
and ORIGIN (ASCII), with PLTAXES=wavelength. All 30 wavelengths and origin labels
come from the retained ECSV; its filename and SHA256 are recorded as provenance.

PROVENANCE is a UTF-8 JSON byte image containing producer, parameters, inputs
(role to source path) and dependencies (role to scientific SHA256). It holds no
scientific arrays or baseline mapping. Python and ESO encode the same payload;
ESO additionally records DFS recipe, input and parameter headers and signatures.
Missing observation metadata are not inferred.

## Persisted classes

| Class | ESO category | Content |
| --- | --- | --- |
| DarkImage | PLANETES_DARK_IMAGE | Subtraction-only mean ADU image |
| DarkCalibration | PLANETES_DARK | Mean, population temporal RMS, bad-pixel bits and actual frame count |
| FlatCalibration | PLANETES_FLAT | Flat response, traces and sparse extraction geometry |
| WavelengthCalibration | PLANETES_WAVE | Detector sample wavelength map |
| P2VMCalibration | PLANETES_P2VM | Inverse matrix, full physical response, grid, mappings and fit diagnostics |
| ReducedObservables | PLANETES_REDUCED | Instrumental observables, GD and differential phase |
| WavelengthReference | PLANETES_LINE_LIST | Static laboratory wavelengths and origin labels |

VisibilityObservables (live without diagnostics) and CalibrationSpectra are
memory-only types. A partial live result cannot be written as a complete reduced
FITS with invented GD or phase fields. All observables preserve the frame axis;
a single frame has frame length one.

## Array fields

| Class | Field | Python axes | Unit |
| --- | --- | --- | --- |
| DarkImage | `mean_adu` | detector_y,detector_x | adu |
| DarkCalibration | `mean_adu` | detector_y,detector_x | adu |
| DarkCalibration | `temporal_rms_adu` | detector_y,detector_x | adu |
| DarkCalibration | `bad_pixel_flags` | detector_y,detector_x | dimensionless / flags |
| FlatCalibration | `profile_x_pixels` | output,sample | pixel |
| FlatCalibration | `profile_y_pixels` | output,sample | pixel |
| FlatCalibration | `flat_map_adu` | detector_y,detector_x | adu |
| FlatCalibration | `trace_y_pixels` | output,detector_x | pixel |
| WavelengthCalibration | `wavelength_map_um` | output,detector_x | um |
| P2VMCalibration | `p2vm` | data_channel,output,wavelength | dimensionless / flags |
| P2VMCalibration | `wavelength_um` | wavelength | um |
| P2VMCalibration | `transmission` | output,telescope,wavelength | dimensionless / flags |
| P2VMCalibration | `coherence_amplitude` | output,baseline,wavelength | dimensionless / flags |
| P2VMCalibration | `coherence_phase_rad` | output,baseline,wavelength | rad |
| P2VMCalibration | `opd_um` | baseline,frame | um |
| P2VMCalibration | `group_delay_um` | baseline,frame | um |
| P2VMCalibration | `ellipse_phase_rad` | baseline,frame,wavelength | rad |
| P2VMCalibration | `ellipse_amplitude` | baseline,frame,wavelength | dimensionless / flags |
| ReducedObservables | `photometric_flux_adu` | frame,telescope,wavelength | adu |
| ReducedObservables | `coherent_flux_adu` | frame,baseline,wavelength | adu |
| ReducedObservables | `visibility_amplitude` | frame,baseline,wavelength | dimensionless / flags |
| ReducedObservables | `flags` | frame,baseline,wavelength | dimensionless / flags |
| ReducedObservables | `wavelength_um` | wavelength | um |
| ReducedObservables | `group_delay_um` | frame,baseline | um |
| ReducedObservables | `differential_phase_deg` | frame,baseline,wavelength | deg |
| WavelengthReference | `wavelength_um` | wavelength | um |

## Parameters

Python names below are the actual keyword names. ESO canonical names prepend
`planetes.<stage>.`; CLI aliases replace underscores with hyphens.

| Stage | Python name | Default | Meaning |
| --- | --- | --- | --- |
| dark | bad_threshold | 5.0 | Multiplier for retained dark mean/RMS bad-pixel tests |
| flat | profile_width | 10 | Full spatial aperture width in detector pixels |
| p2vm | wavelength_min_um | 1.15 | Target grid lower wavelength |
| p2vm | wavelength_max_um | 1.30 | Target grid upper wavelength |
| p2vm | wavelength_channels | 151 | Number of linearly spaced wavelength samples |
| p2vm | phase_correction | true | Apply supplied phase reference |
| p2vm | gd_search_min_um / gd_search_max_um | -30 / 30 | Coarse calibration GD bounds |
| reduce / live | gd_search_min_um / gd_search_max_um | -100 / 100 | Coarse science GD bounds |

Threshold/width must be finite and positive. Wavelength bounds must be finite,
positive and ordered; channels must be an integer (not bool) and the middle-half
slice must contain at least two samples. GD bounds must be finite and ordered.
The retained coarse step is 0.01 um; Newton refinement can leave the coarse
interval. Bounds are not final clipping thresholds. Internal iteration counts,
line rejection, masks, normalization and cache/fastmath remain unchanged.

## Relationships and validity

Flat records the complete dark used to define its mask. Wave records its flat;
P2VM records its flat and wave. Task preparation recomputes scientific hashes
and rejects inconsistent parent combinations, even for identically shaped arrays.
Science can use a different dark without changing flat's established mask.
A scientific hash includes type, units, axes, fixed instrument order, all scientific
arrays/scalars, parameters and parent dependencies, using canonical little-endian
bytes. It excludes producer name and source paths. FITS checksums cover storage
integrity; PLTHASH covers the scientific content and dependency contract.

Sparse geometry uses paired (-1,-1) padding; these entries never contribute to
extraction. Different widths retain all selected pixels and their original order.
Each output must cover the entire P2VM target grid. Coverage failures identify
the output and missing band and are not silently extrapolated.

Full dark requires >=2 raw frames. A 2-D or one-frame input becomes DarkImage
through the explicit `load_dark` API and is usable only for subtraction.
Temporal RMS is not a measured pure readout-noise model. BPM bits are hot=1,
dead=2, noisy=4. Flat rejects subtraction-only darks.

Nonpositive/nonfinite photometric products or nonfinite coherent flux yield
NaN visibility and invalid flags. GUI means and curves honor these flags.
Precomputed extraction and P2VM arrays are read-only snapshots prepared when
calibrations change, rather than rehashed on every frame.

## Publication and migration

Both writers validate science, use a sibling temporary directory with the final
basename, complete encoding/signing/checksums, and read back before atomic
non-overwriting publication. Existing files and symlinks fail. A failed write
leaves no final-name partial product. Unsupported hard-link publication fails
explicitly rather than falling back to an overwriting rename.

Schema-1 FITS/NPZ and old Python APIs are not accepted or automatically converted.
Keep old products and regenerate new calibrations from the raw inputs. The original
raw files and adopted outputs are not modified by this migration.
