# Optional PyEsoRex route

SOF means **Set Of Frames**: a text file with one input per line:

```text
/absolute/path/to/raw.fits DARK_RAW
```

The first column is a file path; the second is the recipe's input role. Shutter
identity still comes from the FITS header. Python examples do not need SOFs.

These `.sof.in` templates describe the legacy `test_data` dataset. Copy a template
to `.sof`, replace `/absolute/PLANETES_new` with your workspace path, and update
product paths if you generated a different reduced directory.

For example, with CPL/PyCPL/PyEsoRex installed, run in a fresh output directory:

```bash
pyesorex --recipe-dir=/absolute/PLANETES_new/planetes_pipe/planetes_pipe/eso \
  planetes_dark /absolute/path/to/dark.sof
```

Run order: `planetes_dark` (each DIT) → `planetes_flat` → `planetes_wave` →
`planetes_p2vm` → `planetes_reduce`. Each recipe's SOF lists its raw files and
previously generated products. The standard Windows environment does not include PyEsoRex.
