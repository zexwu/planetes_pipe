from typing import Any

import numpy as np
from numba import njit, prange
from numpy.typing import NDArray

from . import PipelineContext, arg, command, log
from .products import PREPROC_CALIB_PRODUCT, PREPROC_OBJECT_PRODUCT, FLAT_PRODUCT, WAVE_PRODUCT


@command("preproc", "Preprocessing data cubes.",
         requires=["flat", "wave"], produces=["preproc"])
@arg("--object", type=str, default="p2vm", help="target to preprocess (p2vm or object name)")
@arg("--nwave", type=int, default=151, help="number of wavelength points for interpolation")
@arg("--min-wave", type=float, default=1.15, help="minimum wavelength")
@arg("--max-wave", type=float, default=1.3, help="maximum wavelength")
def run_preproc(ctx: PipelineContext, **kwargs: Any) -> None:
    """
    Extracts flux for P2VM and Flat data, Re-interpolate onto a common wavelength grid.
    """
    log.info("--- Step: PREPROC ---")
    obj = kwargs.get("object", "p2vm")
    if not obj: return

    # Load flat & profile maps
    with ctx.load_product("flat", schema=FLAT_PRODUCT) as d:
        dark_map = d["dark_map"]
        # profile_map = d["profile_map"]
        profile_ys = d["profile_ys"]
        profile_xs = d["profile_xs"]
        flat_map = d["flat_map"]

    # Load wavelength map
    with ctx.load_product("wave", schema=WAVE_PRODUCT) as d:
        wave_map = d["wave_map"]

    # wavelength grid for interpolation
    wl_grid = np.linspace(kwargs["min_wave"],
                          kwargs["max_wave"],
                          kwargs["nwave"])

    _spec_flat  = extract_spec_sparse(flat_map[None, :, :], profile_ys, profile_xs)
    _spec_flat = np.clip(_spec_flat, a_min=1e-8, a_max=None)
    spec_flat = 1 / twopx_interp(1 / _spec_flat, wave_map, wl_grid)

    def _extract(fname: str, dark=dark_map) -> NDArray:
        """Helper to process extraction."""
        log.info(f"    Extracting spectra from {fname}...")
        data = ctx.load_fits(fname) - dark
        spec = extract_spec_sparse(data, profile_ys, profile_xs)

        # NOTE:
        # divide by flat in pixel space before interpolation
        # Then multiply by the flat again to preserve the
        # original flux scale on the new wavelength grid.
        spec_aligned = twopx_interp(spec / _spec_flat, wave_map, wl_grid)
        spec_aligned *= spec_flat

        return spec_aligned

    if obj == "p2vm":
        log.info("Preprocessing P2VM and FLAT files...")
        calib_files = ctx.conf["calib"]
        spec_bsl = [_extract(_) for _ in calib_files["p2vm"]]
        spec_tel = [_extract(_) for _ in calib_files["flat"]]
        spec_wavesc = None
        if "wavesc" in calib_files:
            dark_wavesc = ctx.load_fits(calib_files["wavesc_dark"])
            dark_wavesc = np.median(dark_wavesc, axis=0)  # collapse frames
            spec_wavesc = _extract(calib_files["wavesc"], dark=dark_wavesc)

        # ---------------------------------------------------------
        # Match Baselines to Detector Regions
        # ---------------------------------------------------------

        # Identify which regions belong to which Telescope (based on intensity)
        tel_regs = {}
        for tel_idx, flux in enumerate(spec_tel):
            # Mean flux over frame and wavelength
            mean_flux = flux.mean(axis=(1, 2))
            # Top regions correspond to this telescope
            active_regs = np.argsort(mean_flux)[-ctx.n_reg // 2 :]
            tel_regs[tel_idx] = np.sort(active_regs).tolist()

        # Identify which regions belong to which Baseline (based on modulation/std dev)
        bsl_regs = {}
        for bsl_idx, flux in enumerate(spec_bsl):
            # High std deviation implies interference fringe presence
            std_flux = flux.std(axis=1).mean(axis=-1)
            active_regs = np.argsort(std_flux)[-4:]  # 4 outputs (ABCD) per baseline
            bsl_regs[bsl_idx] = np.sort(active_regs).tolist()

        # Map Baseline ID -> Region Indices and Telescope Pairs
        bsl_to_reg = {}
        bsl_to_tel = {}

        for t1 in range(ctx.n_tel):
            for t2 in range(ctx.n_tel):
                if t1 >= t2: continue  # Unique pairs only

                bsl_name = ctx.telescopes[t1] + ctx.telescopes[t2]
                bsl_idx = ctx.baselines.index(bsl_name)

                # intersection of the two telescopes -> regions for baseline
                # should match regions_from_data for consistency
                regions_from_tels = set(tel_regs[t1]) & set(tel_regs[t2])
                regions_from_data = set(bsl_regs[bsl_idx])
                if str(regions_from_tels) != str(regions_from_data):
                    log.error(f"    Mismatch in region mapping for {bsl_name}: "\
                              f"{regions_from_tels} vs {regions_from_data}")
                    raise ValueError("Baseline mapping consistency check failed.")

                bsl_to_reg[bsl_idx] = list(regions_from_tels)
                bsl_to_tel[bsl_idx] = (t1, t2)

        to_save = ["spec_tel", "spec_bsl", "spec_wavesc", "spec_flat",
                   "tel_regs", "bsl_regs", "bsl_to_reg", "bsl_to_tel",
                   "wl_grid"]
        ctx.save_product(("preproc", obj), schema=PREPROC_CALIB_PRODUCT, **{k: locals()[k] for k in to_save})
        log.info("Preprocessed P2VM and FLAT data saved.")
    else:
        obj_list = ctx.conf["object"] if obj == "all" else [obj]
        to_save = ["spec", "spec_flat", "wl_grid"]
        for obj in obj_list:
            spec = _extract(ctx.conf["object"][obj])
            ctx.save_product(("preproc", obj), schema=PREPROC_OBJECT_PRODUCT, **{k: locals()[k] for k in to_save})
        log.info(f"Preprocessed objects: {', '.join(obj_list)}")
    log.info("--- Step: PREPROC [DONE] ---")
    log.info("")
    return


@njit(parallel=True, fastmath=True, cache=True)
def extract_spec_cpu(cube: NDArray, profile_map: NDArray) -> NDArray:
    F, Y, X = cube.shape
    O = profile_map.shape[0]

    out = np.zeros((O, F, X), dtype=np.float32)

    # Reordered loops: O -> F -> Y -> X
    for o in prange(O):
        for f in range(F):
            for y in range(Y):
                for x in range(X):
                    pixel_val = cube[f, y, x]
                    mask_val = profile_map[o, y, x]
                    out[o, f, x] += pixel_val * mask_val
    return out


@njit(parallel=True, fastmath=True, cache=True)
def extract_spec_sparse(cube: NDArray, ys: NDArray, xs: NDArray) -> NDArray:
    """
    Extract spectra from a cube using sparse pixel indices per output channel.

    Args:
        cube (NDArray): Input cube with shape `(n_frame, ny, nx)`.
        ys (NDArray): Per-output Y indices.
        xs (NDArray): Per-output X indices.

    Returns:
        NDArray: Extracted spectra with shape `(n_reg, n_frame, nx)`.
    """
    n_frame, _, nx = cube.shape
    n_reg = len(ys)
    output = np.zeros((n_reg, n_frame, nx), dtype=np.float32)

    for o in prange(n_reg):
        y_idx = ys[o]
        x_idx = xs[o]
        for f in range(n_frame):
            for i in range(len(y_idx)):
                x = x_idx[i]
                output[o, f, x] += cube[f, y_idx[i], x]
    return output


@njit(parallel=True, cache=True)
def twopx_interp(
    spec: NDArray, wave_map: NDArray, wl_grid: NDArray
) -> NDArray:
    """
    Resample spectra from detector pixels onto a uniform wavelength grid.

    Args:
        spec (NDArray): Input spectra with shape `(n_reg, n_frame, n_pixels)`.
        wave_map (NDArray): Wavelength at each detector pixel with shape `(n_reg, n_pixels)`.
        wl_grid (NDArray): Target wavelength grid with shape `(n_wave,)`.

    Returns:
        NDArray: Resampled spectra with shape `(n_reg, n_frame, n_wave)`.
    """
    n_reg, n_frame, _ = spec.shape
    n_wave = len(wl_grid)

    wn_map = 1.0 / wave_map
    wn_grid = 1.0 / wl_grid

    out = np.zeros((n_reg, n_frame, n_wave))

    for r in prange(n_reg):
        # We reverse (::-1) because np.interp expects increasing X values
        # and wavenumber usually decreases as pixel index increases.
        x_in = wn_map[r, ::-1]

        for f in range(n_frame):
            y_in = spec[r, f, ::-1]
            out[r, f, :] = np.interp(wn_grid, x_in, y_in)

    return out
