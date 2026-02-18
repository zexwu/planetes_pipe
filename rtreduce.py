import numpy as np
from numba import njit, prange
import recipes

@njit(parallel=True, fastmath=True, cache=True, boundscheck=False)
def extract_spec_sparse_single(
    img: np.ndarray, ys: np.ndarray, xs: np.ndarray
) -> np.ndarray:
    """
    Sparse spectral extraction from a 2D detector image.

    Parameters
    ----------
    img : (Y, X) float32
        Calibrated detector image.
    ys, xs : sequence of 1D int arrays
        For each output spectral channel o:
        ys[o], xs[o] define sparse pixel coordinates belonging
        to that channel's extraction mask.

    Returns
    -------
    output : (O, X) float32
        Extracted spectra per channel, accumulated along Y.
    """
    _, X = img.shape
    O = len(ys)
    output = np.zeros((O, X), dtype=np.float32)

    for o in prange(O):
        y_idx = ys[o]
        x_idx = xs[o]
        for i in range(len(y_idx)):
            x = x_idx[i]
            output[o, x] += img[y_idx[i], x]
    return output


@njit(parallel=True, cache=True, boundscheck=False)
def twopx_interp_single(
    spec: np.ndarray, wn_map: np.ndarray, wn_grid: np.ndarray
) -> np.ndarray:
    """
    Resamples spectra from pixel space to a uniform wavenumber grid.

    Args:
        spec (np.ndarray): Input spectra (n_reg, n_pixels).
        wn_map (np.ndarray): wavenumber at each pixel (n_reg, n_pixels).
        wn_grid (np.ndarray): Target wavenumber grid.

    Returns:
        np.ndarray: Resampled spectra (n_reg, n_wave_out).
    """
    n_reg, _ = spec.shape
    n_wave = len(wl_grid)

    out = np.zeros((n_reg, n_wave))

    for r in prange(n_reg):
        x_in = wn_map[r, ::-1]
        y_in = spec[r, ::-1]
        out[r, :] = np.interp(wn_grid, x_in, y_in)

    return out


# Pre-load all calibration data
# -----------------------------------------------------------
with np.load("./reduced/flat.npz") as d:
    profile_ys = np.array(d["profile_ys"])
    profile_xs = np.array(d["profile_xs"])
    flat_map = d["flat_map"]

# Pixel-to-visibility matrix
with np.load("./reduced/p2vm.npz", allow_pickle=True) as d:
    p2vm = d["p2vm"]
    wl_grid = d["wl_grid"]
    bsl_to_tel = d["bsl_to_tel"][()]
bsl_to_tel = np.array([list(bsl_to_tel[i]) for i in range(6)])

# Wavelength calibration: map pixel coordinates to wavenumber (1/micron)
with np.load("./reduced/wave.npz") as d:
    wave_map = d["wave_map"]
wn_map = 1 / wave_map # units of 1/micron
wn_grid = 1 / wl_grid # units of 1/micron


spec_flat = extract_spec_sparse_single(flat_map, profile_ys, profile_xs)
spec_flat = np.clip(spec_flat, a_min=1e-8, a_max=None)
spec_flat_aligned = twopx_interp_single(1 / spec_flat, wn_map, wn_grid)
# -----------------------------------------------------------

def reduce_single(img):
    # -------------------------------------------------------
    # 1. Sparse extraction from detector frame
    # -------------------------------------------------------
    spec = extract_spec_sparse_single(img, profile_ys, profile_xs)

    # -------------------------------------------------------
    # 2. Flat-field correction in detector space
    # -------------------------------------------------------
    spec /= spec_flat

    # -------------------------------------------------------
    # 3. Resample spectra onto uniform wavelength grid
    # -------------------------------------------------------
    spec_aligned = twopx_interp_single(spec, wn_map, wn_grid)
    spec_aligned /= spec_flat_aligned

    # -------------------------------------------------------
    # 4. P2VM Converts photometric channels to:
    #   - Fluxes
    #   - Real & Imaginary coherent flux
    # -------------------------------------------------------
    p2vmred = np.einsum("dow,ow->dw", p2vm, spec_aligned)

    # Photometric fluxes (first 4 rows)
    flux = p2vmred[0:4]

    # Complex coherent flux components
    real = p2vmred[4:10]
    imag = p2vmred[10:16]
    visamp = np.sqrt(real**2 + imag**2)

    # -------------------------------------------------------
    # 5. Normalize by photometric flux per baseline
    # -------------------------------------------------------
    f1f2 = (flux[bsl_to_tel[:, 0]] * flux[bsl_to_tel[:, 1]]) ** 0.5
    visamp /= f1f2

    return visamp

# -----------------------------------------------------------

ctx = recipes.PipelineContext("./conf.yaml")
cube = ctx.load_fits("./2026_02_05/sci2/sci.fits")
dark = ctx.load_fits("./2026_02_05/sci2/sci_dark.fits")
cube -= dark.mean(axis=0)[None, ...]

import timeit
t = timeit.timeit(lambda: reduce_single(cube[0]), number=10000)
print(f"Time per DIT: {t/10000 * 1e3:.3f} ms")

import matplotlib.pyplot as plt
plt.rcParams["figure.dpi"] = 100
fig, axs = plt.subplots(2, 3, figsize=(8, 6))
axs = axs.flatten()
n_frame = len(cube)
visamps = np.zeros((n_frame, 6, len(wl_grid)))
for i in range(n_frame):
    visamps[i] = reduce_single(cube[i])

colors = plt.cm.viridis(np.linspace(0, 1, n_frame))
for bsl in range(6):
    axs[bsl].set_prop_cycle(color=colors)
    axs[bsl].plot(visamps[:, bsl, :].T, alpha=0.05, lw=0.3)
plt.show()
