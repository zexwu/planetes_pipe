# Single-frame benchmark

`rtreduce.py` calls the installed `FrameReducer`; it contains no separate science
implementation. Put the schema-2 `planetes_flat.fits`, `planetes_wave.fits` and
`planetes_p2vm.fits` together in the configured calibration directory, and set
the explicit raw science and dark paths in the script.

Run `python bench/rtreduce.py` with the scientific environment. It warms JIT and
times 128 calls on a complete raw frame, including dark subtraction and flags.
GD/phase diagnostics are disabled, matching the normal live display. The full
file task remains `planetes.reduction.reduce_file`.
