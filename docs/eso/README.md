# Optional PyEsoRex route

On the Apple Silicon Mac, install into the shared workspace environment:

```bash
cd /Users/zexwu/PLANETES_new
brew install cpl
PATH="$PWD/.venv/bin:$PATH" CPLDIR=/opt/homebrew uv sync --locked
uv run --locked pyesorex --recipe-dir=planetes_pipe/planetes_pipe/eso --recipes
```

The workspace enables the `eso` extra on macOS only. Python packages stay in
the uv `.venv`; Homebrew supplies the native CPL library.

SOF means **Set Of Frames**: a text file with one input per line:

```text
/absolute/path/to/raw.fits DARK_RAW
```

The first column is a file path; the second is the recipe's input role. Shutter
identity still comes from the FITS header. Python examples do not need SOFs.

Each recipe's inputs (tag and count) are listed in `planetes_pipe/eso/recipes.py`
and [products.md](../products.md). Ready-to-run SOFs for the 2026-02-05 bench data
are in the workspace at `test_data/sof/`.

For example, with CPL/PyCPL/PyEsoRex installed, run in a fresh output directory:

```bash
pyesorex --recipe-dir=/absolute/PLANETES_new/planetes_pipe/planetes_pipe/eso \
  planetes_dark /absolute/path/to/dark.sof
```

Run order: `planetes_dark` (each DIT) → `planetes_flat` → `planetes_wave` →
`planetes_p2vm` → `planetes_reduce`. Each recipe's SOF lists its raw files and
previously generated products. The standard Windows environment does not include PyEsoRex.
