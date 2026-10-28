# Qt6 viewer and automated acquisition

The local files are updated at `/Users/zexwu/zexwu_lib/devel/`:
`camview_p2vm.py`, its parent `camview.py`, `scripts/take_p2vm.py` and
`scripts/utils.py`. Their exact patches and before/after SHA256 are in
`integration/qt6/manifest.json`. The patches start at the approved 0.3 files
(frozen with pipeline commit 61f519f), not the earlier legacy GUI. Qt5 historical
entries are untouched. The parent has optional writer/check hooks without a
mandatory pipeline import; the current viewer supplies the new hooks.

## Viewer

Use the existing Python 3.12 GUI environment; numerical and Qt dependencies have
not been upgraded. PLANETES is already editable there. Put format-4, format-5 or format-6 FLAT, WAVE
and P2VM products together with their ordinary `planetes_*.fits` filenames:

```bash
/Users/zexwu/zexwu_lib/devel/.venv/bin/python /Users/zexwu/zexwu_lib/devel/camview_p2vm.py \
  --source fake --reduced-dir /absolute/calibrations \
  --fake-cube /absolute/science.fits --fake-dark /absolute/planetes_dark.fits
```

Capture types now include OBJECT, DARK, WAVE (lamp), FLAT and P2VM. Before capture,
confirm physical shutter 1..4 states, detector identity, orientation and **camera**
binning. These describe the actual cube; the separate display binning selector
is not physical binning. Missing configuration or a type/shutter conflict prevents
capture. Settings are snapshotted at start; capture controls are locked, known
camera readbacks are checked during capture, and changes prevent publication.
New floating raw arrays are stored as float32; integer pixels retain their type.
Filenames never overwrite old raw.

The driver currently exposes exposure_time [us], offset_x/y and width/height.
Those readbacks populate exposure/ROI. Detector/orientation/physical binning are
explicit operator configuration because no verified driver interface for them is
available here. Shutters are operator-confirmed, not asserted to be hardware
feedback. Host start/end UTC are recorded separately; no fictitious DATE-OBS is
created. External unobservable shutter changes cannot be detected by this interface.

Unbinned raw frames feed the common FrameReducer. Dark off, a wrong-category dark
or incompatible calibration pauses updates and clears the current spectrum.
Taking the current frame creates `Dark(mean_adu, n_frames=1)` with RMS=None.
Loaded 2-D images return the same Dark type with unknown RMS/frame count.
Subtraction only needs the mean; BAD detection needs measured RMS. Complete
DARK FITS publication retains its measured-statistics requirements. Invalid channels remain flagged/gaps in curves.

## Automated take_p2vm

The existing acquisition configuration still describes device commands and output
locations; it is not a pipeline recipe scheduler. Add explicitly confirmed camera
settings to that existing file, for example (replace values with the actual setup):

```yaml
camera:
  detector: actual-camera-identifier
  orientation: actual-sensor-orientation
  binning_x: 1
  binning_y: 1
```

The script checks these before connecting to hardware. Existing DIT/scan arguments,
scan-arm selection, OPD grid, motion order, settling, shutter closure and DL restore
remain. Physical telescope 1/2/3/4 corresponds to actuator 0/1/2/3. Each dark, flat,
baseline and lamp capture calls the same PLANETES raw writer as the viewer.
Shutter states are the issued commands; camera settings prefer actual readback,
with explicitly recorded command exposure allowed when readback is unavailable.
Detector/orientation/binning from the configuration are labeled operator-supplied.
Scan positions are separately recorded command metadata, not treated as forbidden
fixed camera-configuration changes. A no-DL camera failure also closes shutters.

## Validation, transfer and recovery

The earlier Qt6 integration passed offscreen window checks with recorded frames: capture,
float pixel preservation, metadata, changing camera settings, display-binning
science equivalence, dark-off and single-frame dark. A fake full automated run
wrote 13 classified cubes and verified shutter/DL recovery and sources. No test
connected to or operated hardware. The local GUI environment has no catkit2;
actual bench deployment remains a separate hardware acceptance step.

For 0.8.0, the actual GUI Python 3.12 environment reads format-4 calibration files
and reduces a recorded real frame to preview P2VMRed. The updated camera metadata
helper and raw publication passed a fake-camera check. All four manifest hashes
match the local consumers, and the acquisition patch reproduces the new helper
when applied to its original source.

On another machine, verify the original hashes, back up all four files and run
`git apply --check` for each patch from the devel directory before applying.
Do not apply again to already updated files. Install the matching PLANETES 0.10.0 version
without replacing existing camera/Qt dependencies. To restore, preserve later local
edits and recover the four original files from:

`/Users/zexwu/zexwu_lib/planetes_pipe_backups/20261027_223235_gravity_conventions/devel/`

Restore the matching 0.3 pipeline/calibrations too if reverting to format 2.

The current viewer imports `load_reducer` from planetes.reduction and `load_dark` from planetes.dark and passes
raw arrays to FrameReducer. Acquisition imports CaptureMetadata/build_raw_header/write_raw from
planetes.io.acquisition and DetectorSetup from planetes.instrument. These local consumers
and their portable patches are updated together; no compatibility shim is required.

Version 0.5 uses `FrameReducer.wavelength_m` and `extractor.wavelength_map_m`.
The viewer consumes those metre arrays directly and labels its axes accordingly.
Format-4/5 calibrations remain readable; newly saved products use format 6.

The 0.8.0 product simplification preserves the viewer entry points. Acquisition
uses `build_raw_header(metadata, shape)` before capture and `write_raw` for publication;
the removed standalone raw validator is no longer imported. The portable patch
and manifest match the updated local acquisition helper. Science products use
`Dark` and `P2VMRed`; the GUI needs no CPL parameter names or SOF declarations.
The viewer owns `reducer_dir` for its selector/status text; `FrameReducer` no longer
stores a directory. The viewer reads `baseline_outputs`, and preview reduction
uses the default `include_phase=False`.

Version 0.8.1 checks the presence of temporal RMS for the status label, rather than
dispatching on separate Dark/DarkImage classes. The current-frame button stores
known count 1 and never fabricates a zero noise image.

Version 0.9.0 separates the raw reader from acquisition writing. The viewer's
capture helper uses `CaptureMetadata` from `io.acquisition`; pipeline reading
returns only physical raw fields. Capture timestamps and configuration sources
remain recorded in raw format 3. The viewer reads P2VM products without scan
diagnostics. Fixed telescope pairs come from `instrument.BASELINE_PAIRS`.

Version 0.10.0 promotes raw frames at the FrameReducer input boundary. The
current-frame dark button also adapts its copied image once, so the scientific
core receives float64 arrays. Calibrations loaded from FITS are promoted at read;
floating raw/products are stored as float32 and complex signals as complex64.
No camera or existing environment dependency was changed.
