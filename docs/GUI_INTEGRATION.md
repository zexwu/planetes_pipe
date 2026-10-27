# Qt6 viewer integration

The migrated entry point is `/Users/zexwu/zexwu_lib/devel/camview_p2vm.py`.
It imports the installed `planetes` package and calls FrameReducer. The local
file is already updated. The parent camera module, acquisition/capture code,
Qt5 historical viewer and historical independent reducer are retained.
The precise portable change is `integration/qt6/camview_p2vm.patch`, with before
and after SHA256 in the adjacent manifest.

## Environment and launch

The existing GUI environment is `/Users/zexwu/zexwu_lib/devel/.venv` (Python 3.12).
PLANETES was installed editable with `--no-deps`; its NumPy/SciPy/Numba/Astropy/Qt
packages were not replaced. Qt and catkit2 are not dependencies of the science
package. This environment supports fake frames; catkit2 is not installed there.

Put the three schema-2 calibrations together as `planetes_flat.fits`,
`planetes_wave.fits` and `planetes_p2vm.fits`, then use:

```bash
/Users/zexwu/zexwu_lib/devel/.venv/bin/python /Users/zexwu/zexwu_lib/devel/camview_p2vm.py \
  --source fake --reduced-dir /absolute/calibrations \
  --fake-cube /absolute/science.fits --fake-dark /absolute/planetes_dark.fits
```

Select the actual dark and calibration directory in the viewer. Raw frames go
to the science core; binning only changes the display. Dark off, a wrong-category
dark or incompatible calibration pauses updates and clears the current spectrum.
Taking the current frame as dark creates subtraction-only DarkImage and displays
that status. Complete raw dark stacks use the common dark calibration function.
The 24 labels use the shared A,C,B,D order; invalid channels are excluded from
means and left as gaps in curves.

The headless check is `validation.gui_smoke.validate_gui`, run with the GUI
interpreter and `QT_QPA_PLATFORM=offscreen`. It exercises the real window using
recorded frames and saves screenshots, check results and warmed single-frame
timings. It does not contact the bench. Tests validate the new core integration;
live camera timing and bench deployment still require the actual bench environment.

## Transfer and recovery

Before applying the patch on another machine, verify its original file hash,
back up that file and run `git apply --check /absolute/camview_p2vm.patch` from
the directory containing it. Then apply with `git apply`. Install PLANETES in
the chosen GUI environment without replacing its acquisition dependencies.
Do not apply the patch twice to the already updated local file.

The exact local original is backed up at:

`/Users/zexwu/zexwu_lib/planetes_pipe_backups/20261027_132125_restructure/gui/camview_p2vm.py`

To recover, preserve any subsequent local edits, restore that saved viewer and
use its matching old calibrations/reducer. The original parent `camview.py`
and historical `planetes_pipe/rtp2vm.py` are also in the backup and unchanged
at their original locations. Backups were rechecked against their original hashes.
