# PLANETES science products and FITS interface (v0.14.0)

## Design

The scientific functions return **plain dataclasses** from `planetes.products`.
They store arrays and optional detector context; file provenance and association
are handled at the I/O/recipe boundary. Units and array axes are visible in each field's name/comment.
`planetes.io.fits` alone defines serialization; `recipes/` binds SOF roles,
recipe parameters, CPL DFS and file registration. These layers are independent.

ESO GRAVITY's **DARK, BAD, FLAT, WAVE, P2VM** categories and the distinct
**P2VMRED** intermediate product are the naming reference. The FITS layout is
a PLANETES adaptation, **not** a byte-for-byte implementation of GRAVITY's
internal products. GRAVITY `VIS` is a further (averaged, uncalibrated) OIFITS
product; it is not implemented here.

Official references:

- [ESO GRAVITY product names](https://www.eso.org/observing/dfo/quality/GRAVITY/pipeline/product_codes.html)
- [ESO GRAVITY Pipeline User Manual, issue 1.10.2 (12 June 2026)](https://ftp.eso.org/pub/dfs/pipelines/gravity/gravity-pipeline-manual-1.10.2.pdf), §2.4–2.5

## Precision boundary

`io/hdu.py` defines storage as float32/complex64 and computation as
float64/complex128. Product FITS reads promote floating payloads once; raw reads
promote each 16-frame block. Writers narrow new floating payloads once. Integer
raw pixels, indices, flags and labels retain their types. Existing files stay
unchanged; old double-precision payloads remain readable without narrowing on read.
FITS header scalars retain their normal textual representation.

The array-level API expects calculation-precision arrays. It does not cast inside
scientific algorithms. Python input adapters can use `as_compute_array()` once;
file entry points and product readers already do this. No dtype setting is stored
in scientific dataclasses or passed through algorithm parameters.

## Python objects

| Dataclass | Important contents | Usage |
| --- | --- | --- |
| `Dark` | required `mean_adu` `(y,x)`; optional `temporal_rms_adu`, `n_frames`, `setup` | Mean-only or measured dark |
| `Bad` | `bad_pixel_flags` `(y,x)` | Bit flags: hot 1, dead 2, noisy 4 |
| `Flat` | sparse `profile_x_pixels`/`profile_y_pixels`, `flat_map_adu`, `trace_y_pixels`; optional sigma/flux/setup | FLAT and extraction geometry |
| `Wave` | `wavelength_map_m` `(output,x)`; optional fit residuals | WAVE solution |
| `P2VM` | `matrix` `(16,24,wavelength)` plus wavelength, inferred output mapping and physical response; optional fit diagnostics | P2VM response/inverse |
| `P2VMRed` | per-frame `photometric_flux_adu`, complex `coherent_flux_adu`, `visibility_amplitude`, `flags`; optional GD and differential phase | Science intermediate, **not VIS** |
| `WavelengthReference` | laboratory `wavelength_m`, origins | Static line list |
| `P2VMSpectra` | telescope/scan spectra, wavelength and explicit output identity | Internal P2VM fit input, not a file product |

`P2VMRed.group_delay_m` and `.differential_phase_deg` are `None` in the
fast live preview (`include_phase=False`). Formal file reductions compute them; only complete
`P2VMRed` objects can be saved to FITS.

`DarkImage` and the `DarkInput` union are removed. Subtraction accepts one `Dark`
type. Temporal RMS is None until measured; frame count is None when unknown and
1 for a single measured frame. BAD detection requires RMS. QC always measures
the available mean and omits unknown RMS. The existing
complete DARK file category still requires RMS and a frame count of at least two.
`DetectorSetup` is defined in `instrument.py`; unknown
header fields are `None`. Static telescope order is 4,3,2,1; baseline order
43,42,41,32,31,21; each baseline's outputs are A,C,B,D.
`instrument.BASELINE_PAIRS` defines telescope-axis indices once. Products no longer
carry `baseline_telescopes`; their data-derived `baseline_outputs` is retained.
`keep_diagnostics=True` retains P2VM scan OPD/GD, ellipse phase/radius, fitted DC
and residual RMS. The default omits these arrays from memory products and FITS,
while all calculations used to derive the response and inverse stay unchanged.

`make_wave` requires a `WavelengthReference`; loading a default is the caller's
choice. `fit_p2vm(spectra, phase_reference=reference)` applies instrumental phase
correction when reference spectra are provided, and omits it for `None`. The
reference has axes `(output, frame, wavelength)` on the exact scan grid and has
already had its selected DARK subtracted. It is not a `P2VMSpectra` field.
`make_p2vm` accepts the same extracted reference array. File selection, extraction
and choice of reference DARK belong to the script or recipe. The P2VM recipe
explicitly reuses its main DARK when no PHASE_DARK_CALIB is selected.

## Raw bench input

The reader selects IMAGING_DATA_SC when present, otherwise the unique nonempty
3-D primary/image extension. Ambiguous cubes require explicit low-level `hdu`
selection. Known non-ADU units are rejected; missing BUNIT follows the bench ADU
convention. Missing physical fields remain None. The reader returns only category,
shutter states and detector setup from FITS headers and image dimensions.
There is no JSON reader, fallback or header-versus-JSON reconciliation. The writer
in `io/acquisition.py` accepts the same `RawMetadata` and writes physical FITS
cards, without acquisition-source or capture-history records.

Python telescope/baseline dictionaries supply identities when shutter readback
is incomplete. All known shutter states must agree with that assignment.
SOF keeps the existing repeated FLAT_RAW/P2VM_RAW categories and uses recorded
shutters; no additional temporary SOF tags are introduced.

## FITS products

Python field renames (`matrix`, `baseline_outputs`) do not change FITS extension
or column names. `FrameReducer` contains calculation state only and is constructed
from explicitly loaded FLAT, WAVE and P2VM products.

Reduced FITS files use `HIERARCH PLANETES FORMAT = 7`. P2VM scan diagnostics are
optional. Older generated calibration products must be regenerated; the reader
has no JSON compatibility path. Raw reading uses the physical header and cube,
without a private raw-format version or metadata extension.

| Category | FITS payload |
| --- | --- |
| DARK | `IMAGING_DATA_SC` mean [ADU], `IMAGING_ERR_SC` temporal RMS [ADU], `NFRAMES` |
| BAD | `IMAGING_DATA_SC` bit flags |
| FLAT | `IMAGING_DETECTOR_SC` labels, `PLANETES_FLAT_IMAGE`, `PROFILE_PARAMS`, `PLANETES_PROFILE`, optional `PLANETES_FLAT_FLUX` |
| WAVE | `WAVE_DATA_SC` metre-valued detector wavelength columns; optional `PLANETES_WAVE_FIT` |
| P2VM | `IMAGING_DETECTOR_SC`, `PLANETES_WAVELENGTH`, `P2VM_SC` (transmission, coherence, phase), `PLANETES_P2VM_INVERSE` and optional fitting diagnostics |
| SINGLE_SCI_P2VMRED | `PLANETES_FLUX` and `PLANETES_VIS`: per-frame station indices, instrumental flux/coherence, visibility amplitude, flags, GD [m] and differential phase [deg] |
| LINE_LIST | laboratory wavelength [m] and origin labels |

All products carry `ESO PRO CATG/TYPE/SCIENCE`, named extensions, physical
units and per-HDU FITS checksum/DATASUM. DARK/FLAT detector settings are stored
only in the primary header: `ESO DET ID`, window/ROI/binning cards and DIT.
There is no `PLANETES_META` extension. FITS checksums detect transmission errors. New standalone
Python files do not encode input paths or processing parameters. PyCPL/ESO DFS
files additionally record recipe parameters and the frames actually used.

FITS station indices are physical telescope numbers. Sparse profile indices and
trace centers are one-based on disk and zero-based in Python; accepted WAVE_FIT
X/Y fit samples retain their existing zero-based coordinates. The sparse pixel
`-1` remains a sentinel. Wavelength and group delay use **metres**, fringe phase
**radians**, differential phase **degrees**, instrumental counts **ADU**.
Boolean `FLAG=True` marks an invalid visibility sample. Floating binary-table
columns use E/C and floating image payloads use 32-bit real components. Reads
return native float64/complex128 arrays, so FITS round trips preserve the stored
precision rather than the original unrounded calculation arrays.

Writers generate per-HDU checksums, stage outputs and refuse overwrites. Recipes
also apply CPL DFS signatures. They do not automatically reopen and decode the
completed files; round-trip checks are exercised by the development tests. Product
reads still verify checksums, units, category and physical row identities.

## What is checked and where

- At the calculation entry points, fixed telescope/baseline identities and
  detector/wavelength dimensions are checked before fitting or matrix application.
  This prevents NumPy broadcasting physically incompatible input spectra.
- Sparse extraction indices and wavelength coverage/order are checked once before
  Numba gathering and interpolation. Coverage endpoints allow only the relative
  float32 round-to-nearest error bound (`2**-24`), defined at the I/O boundary;
  larger gaps are rejected. V2PM inversion requires full numerical rank
  using NumPy matrix_rank; the pseudoinverse algorithm itself is unchanged.
- At the acquisition/calibration boundary, **known** ROI/DIT/detector
  contradictions fail; missing provisional header fields are *unknown*, not
  fabricated or silently interpreted as matching.
- At the FITS boundary, unit codes, checksums, product categories, physical
  telescope/baseline identity and frame order are checked. Output files are
  published without overwriting existing paths.
- General purpose `validate_wavelength()`, `validate_gd_range()`, `positive()`,
  `fingerprint()`, `require_parent()` and object-level `validate()` are gone.
- QC remains optional through `include_qc`. Existing Python and recipe publication
  defaults retain QC (`True`); use `False` to skip it. The FITS encoder only
  serializes supplied measurements. QC never changes scientific masks or weights.

The raw FITS keyword dictionary and calibrated VIS/VIS2/T3 OIFITS remain
subsequent work. Python-only calls keep their input selection and parameters
in the calling script; they do not build a product provenance graph.

## Parameters

Each recipe declares canonical names and aliases locally. Scientific parameters
are listed in its `PARAMETERS` dictionary; the P2VM file-selection switch
`phase-correction` is declared directly on that recipe.
Python step functions own scientific arguments, units and defaults.
`recipes/parameters.py` translates the recipe declarations and reads defaults from
the function signature; science modules contain no CPL parameter registry.
Each recipe passes its required tag/count pairs to `read_sof`. The result contains
one path list per tag and classified DFS input frames. Product loading is explicit
in the recipe; the SOF helper neither reads products nor changes value types.

| Python keyword | Canonical CPL name | Default |
| --- | --- | --- |
| bad_pixel_threshold | planetes.calib.bad-dark-threshold | 5.0 |
| aperture_width_pixels | planetes.calib.profile-width | 10.0 pixel full aperture |
| phase-correction (recipe only) | planetes.calib.phase-correction | true |
| keep_diagnostics | planetes.calib.keep-diagnostics | false |
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
With a disabled or absent reference, neither optional reference input is read or
registered among DFS used inputs. An unused PHASE_DARK_CALIB alone is ignored. Repeated FLAT_RAW/P2VM_RAW tags remain unchanged; duplicate physical
shutter combinations or incomplete sets fail.

## Step APIs

FLAT/P2VM file arguments accept only explicit `Mapping[str, Path]`: telescope
keys `4,3,2,1` and baseline keys `43,42,41,32,31,21`. Known shutter contradictions,
missing identities and repeated file assignments are rejected. Files with unknown
shutters remain usable with those explicit Python mappings. Repeated SOF raw files
are identified by `read_shutter_id` at the recipe boundary; unknown shutters need
an explicit Python mapping. No path/metadata tuples flow into scientific steps.

Raw metadata contains only category, four shutter states and physical detector
settings. `DetectorSetup` has exposure, ROI, binning and detector fields; no
exposure-source, legacy or orientation state is carried. Unknown physical header
fields stay unknown; dimensions can be read directly from the detector cube.

Raw-file workflows are `make_dark_products`, `make_flat`, `make_wave`,
`make_p2vm` and `reduce_science`. They return products and leave saving explicit.
Array-level functions are `measure_dark`, `detect_bad_pixels`,
`build_flat(flat_images, bad)`, `fit_wavelength_solution`, `fit_p2vm` and
`reduce_spectra`. `flat_images` are already dark-subtracted; `build_flat` does
not receive or subtract a DARK. `make_flat` carries the selected raw detector setup.

`aperture_width_pixels` is the full extraction aperture (default 10 pixels),
separate from Gaussian sigma. Its existing CPL name/alias remains
`planetes.calib.profile-width` / `profile-width`. `trace_spectrum` always returns
`(centers, position_errors, gaussian_sigma)`. `include_phase` controls GD and
differential phase together: True for complete file reduction, False by default
for FrameReducer preview. The deleted wavelength-grid helper is replaced by
`np.linspace` at the call site; the fit retains its central-band sampling check.
