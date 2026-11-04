# Products, FITS layout and QC

What each step writes, how it is named, and what the headers contain.
Algorithms are described in [methods.md](methods.md).

The categories follow ESO GRAVITY (DARK, BAD, FLAT, WAVE, P2VM, and the per-frame
P2VMRED intermediate). The FITS layout is a PLANETES adaptation, not a byte copy
of GRAVITY. Calibrated, averaged OIFITS (GRAVITY `VIS`) is not implemented yet.
References: [GRAVITY product codes](https://www.eso.org/observing/dfo/quality/GRAVITY/pipeline/product_codes.html),
[GRAVITY Pipeline User Manual 1.10.2](https://ftp.eso.org/pub/dfs/pipelines/gravity/gravity-pipeline-manual-1.10.2.pdf) §2.4–2.5.

## Products

| Module | Python class | PRO CATG | FITS extensions |
| --- | --- | --- | --- |
| `dark.py` | `Dark` | DARK | `IMAGING_DATA_SC` mean [ADU], `IMAGING_ERR_SC` temporal RMS [ADU]; `NFRAMES` |
| `dark.py` | `Bad` | BAD | `IMAGING_DATA_SC` bit flags: hot 1, dead 2, noisy 4 |
| `flat.py` | `Flat` | FLAT | `IMAGING_DETECTOR_SC`, `PLANETES_FLAT_IMAGE`, `PROFILE_PARAMS`, `PLANETES_PROFILE`, optional `PLANETES_FLAT_FLUX` |
| `wave.py` | `Wave` | WAVE | `WAVE_DATA_SC` wavelength per output and x [m]; optional `PLANETES_WAVE_FIT` |
| `p2vm.py` | `P2VM` | P2VM | `IMAGING_DETECTOR_SC`, `PLANETES_WAVELENGTH`, `P2VM_SC`, `PLANETES_P2VM_INVERSE`; optional scan diagnostics |
| `science.py` | `P2VMRed` | SINGLE_SCI_P2VMRED | `PLANETES_FLUX`, `PLANETES_VIS`: per-frame flux, coherent flux, VISAMP, flags, GD [m], differential phase [deg] |

`P2VMSpectra` (P2VM fit input) and `FrameReducer` (live reduction state) are
in-memory only. A `P2VMRed` from the live preview has no GD/phase and cannot be saved.

## Conventions

- Telescopes 4,3,2,1; baselines 43,42,41,32,31,21; outputs per baseline A,C,B,D.
  FITS station indices are physical telescope numbers.
- Wavelength, OPD and GD in metres; fringe phase in radians; differential phase
  in degrees; counts in ADU. `FLAG=True` marks an invalid visibility sample.
- Pixel indices and trace centres are one-based on disk and zero-based in Python;
  `-1` is the invalid sparse-pixel sentinel. WAVE fit X/Y stay zero-based.
- Calculations use float64/complex128 and products store them as computed;
  integers and flags keep their types. Products written before 0.17.0 are float32.

## Raw input

`raw.read_raw(path, category, hdu="IMAGING_DATA_SC")` returns `Raw`: the
memory-mapped cube as stored, the four shutter states and the `DetectorSetup`
(DIT, ROI, binning, detector id). Missing header fields stay `None`. Raw tags
map to `ESO DPR TYPE` in `raw.RAW_TYPES`. Shutters are `ESO INS SHUTn ST`;
files written before 2026-11 use `PLANETES INS SHUTn ST`, which is still read.
FLAT and P2VM raw lists may be in any order: physical identity comes from shutters.

## File names and headers

A product is named `<reference raw stem>_<category without underscores>.fits`,
e.g. `PLANETES.2026-02-05T210609_singlescip2vmred.fits`. The reference is DARK_RAW,
WAVELAMP_RAW or SINGLE_SCI_RAW, the T4 flat for FLAT, and the 43 scan for P2VM.
Existing files are never overwritten (`OSError`).

Every keyword is a `Keyword` in ESO's short form (`DET.WIN1.DIT1` is the card
`HIERARCH ESO DET WIN1 DIT1`) with its comment and unit, as in an ESO dictionary
(DICD ESO-044156 §7.3). Cards are written as `value / [unit] comment`, shortened to
one 80-column card. The primary header (`products.provenance`, `save_product`) holds:

- the reference raw's cards, without structural, checksum, file-identity, DPR and
  non-ESO hierarchical cards (as CPL DFS);
- the DICD §4.7.1 keywords: `PIPEFILE`, `PRO.CATG/TYPE/TECH/SCIENCE/DID`,
  `PRO.REC1.ID/DRS.ID/PIPE.ID`, `PRO.REC1.RAWi.NAME/CATG`, `CALi.NAME/CATG`,
  `PARAMi.NAME/VALUE`, `PRO.DATANCOM`; `DATE` as `YYYY-MM-DDThh:mm:ss.sss` (§4.5);
- the detector setup (`DET...`) for DARK and FLAT;
- QC (below).

Cards are ordered primary first, then by category DPR, OBS, TPL, GEN, TEL, ADA,
INS, DET, then PRO and QC (§3.1.1.3). Extensions carry `INHERIT = T`. PyEsoRex
additionally signs each product (`DATAMD5`). Every HDU has CHECKSUM/DATASUM.

## QC

QC keywords are defined beside each module's `qc(product)`, which returns
`{Keyword: value}`; `write_<product>()` writes them.
Values that could not be measured are omitted. QC is descriptive: it never
changes masks, weights or fits. Definitions follow GRAVITY 1.9.6
(`gravi_calib.c`, `gravi_wave.c`, `gravi_p2vm.c`); they are not yet agreed with ESO.

| Product | Keyword (after `ESO QC`) | Definition |
| --- | --- | --- |
| DARK | MEDIANDARK SC / DARKRMS SC | spatial median of the temporal mean / population RMS [ADU] |
| BAD | BADPIX SC / BADPIX_DARK SC / BADPIX_RMS SC | flagged pixels: any, hot or dead, noisy |
| FLAT | FLATFLUX SC1..4 | total flux of each dark-subtracted mean frame, T4..T1 [ADU/DIT] |
| WAVE | MINWAVE SC / MAXWAVE SC | wavelength range covered by every output [m] |
| WAVE | RMSWAVE SC / NWAVE SC | sample stdev of accepted lamp residuals [m] / their number |
| P2VM | P2VM_COHERENCE_SC43 … SC21 | coherence / (2 √T1 √T2), mean over outputs and wavelengths |

P2VMRED has no QC yet.

## Recipe options

The CPL parameter is the Python keyword with dashes, under `planetes.calib` or
`planetes.reduce`. Help texts are in `eso/recipes.py`.

| Python keyword | CPL parameter | Default |
| --- | --- | --- |
| `bad_pixel_threshold` | planetes.calib.bad-pixel-threshold | 5.0 |
| `aperture_width_pixels` | planetes.calib.aperture-width-pixels | 10.0 (full aperture, pixels) |
| `metrology_trace` | planetes.calib.metrology-trace | false |
| `keep_diagnostics` | planetes.calib.keep-diagnostics | false |
| `gd_search_min_m` / `max_m` (P2VM) | planetes.calib.gd-search-min-m / -max-m | −30e-6 / 30e-6 m |
| `dl_gain` (P2VM) | planetes.calib.dl-gain | none (e.g. `43:-1,42:-1,41:-1,32:1,31:-1,21:-1`) |
| `gd_search_min_m` / `max_m` (science) | planetes.reduce.gd-search-min-m / -max-m | −100e-6 / 100e-6 m |

A PHASE_REFERENCE in the P2VM SOF is always applied, with PHASE_DARK_CALIB if given
and the main DARK otherwise.

## What is checked

Only mistakes a user or the data can make are checked:

- a calibration whose known DIT, ROI, binning or detector differs from the raw;
- missing or duplicate shutter configurations in FLAT/P2VM inputs;
- detector-shape mismatches between raw, DARK, BAD, FLAT and WAVE;
- wavelength coverage and sparse pixel bounds before the numba kernels;
- rank-deficient P2VM matrices and failed wavelength fits;
- product category when a product is read (checksums are written, not re-verified).
