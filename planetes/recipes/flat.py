from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from numba import njit
from numpy.typing import NDArray
from scipy.ndimage import median_filter
from scipy.signal import find_peaks, medfilt

from .. import log
from ..config import FlatParameters, PipelineConfig
from ..fits import CubeSelection, iter_cube
from ..numerics import frame_mean, frame_mean_std
from ..product_io import write_product
from ..products import FlatProduct
from .plotting import genfig, plt


@dataclass(slots=True)
class _FlatDiagnostics:
    flat_maps: list[NDArray]
    ys_raw: NDArray
    ys_error: NDArray


def run_flat(config: PipelineConfig) -> FlatProduct:
    """Build detector maps and sparse spectral traces."""
    log.info("--- Step: FLAT ---")
    log.info("Building DARK, BAD and FLAT maps...")
    selection = CubeSelection(
        frames=config.calibration.frames.slice,
        y=config.crop.y.slice,
        x=config.crop.x.slice,
    )
    dark_map, dark_std = frame_mean_std(
        iter_cube(
            config.calibration.dark,
            selection,
            chunk_frames=config.chunk_frames,
        )
    )
    flat_maps = [
        frame_mean(
            iter_cube(
                config.calibration.flats[name],
                selection,
                chunk_frames=config.chunk_frames,
            )
        )
        - dark_map
        for name in config.instrument.telescopes
    ]
    product, diagnostics = build_flat_product(
        dark_map,
        dark_std,
        flat_maps,
        n_outputs=config.instrument.n_outputs,
        parameters=config.parameters.flat,
    )
    write_product(config.output_dir / "flat.npz", product)
    log.info("Saved profile, flat, dark, and trace data.")

    if config.automatic_plots:
        with PdfPages(config.output_dir / "flat.pdf") as pdf:
            _plot_flat(
                pdf,
                product,
                diagnostics,
                config.instrument.telescopes,
            )

    log.info("--- Step: FLAT [DONE] ---")
    log.info("")
    return product


def build_flat_product(
    dark_map: NDArray,
    dark_std: NDArray,
    flat_maps: list[NDArray],
    *,
    n_outputs: int,
    parameters: FlatParameters,
) -> tuple[FlatProduct, _FlatDiagnostics]:
    """Compute flat maps and traces from pre-reduced detector images."""
    bad_map = compute_bad_map_from_stats(
        dark_map, dark_std, threshold=parameters.bad_threshold
    )

    flat_map = np.mean(flat_maps, axis=0)

    h, w = flat_map.shape
    h_half = h // n_outputs // 2

    spatial_profile = flat_map.sum(axis=1)
    thresh = np.percentile(spatial_profile, 75)
    ref_ys, props = find_peaks(spatial_profile, distance=h_half, height=thresh)
    log.debug(f"Found {len(ref_ys)} Peaks with THRESH {thresh:.0f}...")

    top_indices = np.argsort(props["peak_heights"])[-n_outputs:]
    ref_ys = np.sort(ref_ys[top_indices])

    profile_map = np.zeros((n_outputs, h, w), dtype=np.uint8)
    xs_list, ys_list, ys_raw_list, ys_err_list = [], [], [], []
    profile_xs = []
    profile_ys = []

    log.info("Building PROFILE map...")
    for i, ref_y in enumerate(ref_ys):
        log.debug(f"    Tracing spectrum {i+1}/{n_outputs} at Y={ref_y}...")
        xs = np.arange(w)
        ys, ys_err = trace_spectrum(
            flat_map, bad_map,
            ref_x=w // 2, ref_y=ref_y, range_y=h_half
        )

        ys_med = medfilt(ys, kernel_size=3)
        good = (ys_err < 0.5) & (np.arange(w) < w - 50) & (np.arange(w) > 50)
        poly_coeffs = np.polyfit(xs[good], ys_med[good], deg=3)
        ys_smoothed = np.polyval(poly_coeffs, xs)
        Y, X = np.indices((h, w))

        spatial_mask = np.abs(Y - ys_smoothed[X]) < parameters.profile_width / 2
        profile_map[i, spatial_mask & (bad_map == 0)] = 1

        # Keep a rectangular sparse array for NPZ/Numba; extractors skip -1 sentinels.
        sparse_x = X[spatial_mask].copy()
        sparse_y = Y[spatial_mask].copy()
        invalid = bad_map[sparse_y, sparse_x] > 0
        sparse_x[invalid] = -1
        sparse_y[invalid] = -1
        profile_xs.append(sparse_x)
        profile_ys.append(sparse_y)

        xs_list.append(xs)
        ys_list.append(ys_smoothed)
        ys_raw_list.append(ys)
        ys_err_list.append(ys_err)

    product = FlatProduct(
        profile_map=profile_map,
        profile_xs=np.array(profile_xs),
        profile_ys=np.array(profile_ys),
        flat_map=flat_map,
        dark_map=dark_map,
        bad_map=bad_map,
        xs=np.array(xs_list),
        ys=np.array(ys_list),
    )
    diagnostics = _FlatDiagnostics(
        flat_maps=flat_maps,
        ys_raw=np.array(ys_raw_list),
        ys_error=np.array(ys_err_list),
    )
    return product, diagnostics


def _plot_flat(
    pdf: PdfPages,
    product: FlatProduct,
    diagnostics: _FlatDiagnostics,
    telescopes: tuple[str, ...],
) -> None:
    fig, axs = plt.subplots(1, 3, sharey=True)
    titles = ["DARK", "FLAT", "BAD"]
    images = [product.dark_map, product.flat_map, product.bad_map]
    for ax, image, title in zip(axs, images, titles, strict=True):
        vmin, vmax = np.percentile(image, [0, 99])
        if title == "BAD":
            vmin, vmax = 0, 1
        ax.imshow(image, vmin=vmin, vmax=vmax)
        ax.set_title(title)
    pdf.savefig(fig)
    plt.close(fig)

    fig, axs = plt.subplots(1, len(telescopes))
    vmin, vmax = np.percentile(product.flat_map, [1, 99])
    for i, image in enumerate(diagnostics.flat_maps):
        axs[i].imshow(image, vmin=vmin, vmax=vmax)
        axs[i].set_title(f"Tel {telescopes[i]}")
        axs[i].axis("off")
        for reg in range(len(product.profile_map)):
            axs[i].plot(product.xs[reg], product.ys[reg], c="blue", lw=0.3)
            good = diagnostics.ys_error[reg] < 5
            axs[i].plot(
                product.xs[reg][good],
                diagnostics.ys_raw[reg][good],
                c="red",
                lw=0.3,
                ls="--",
                alpha=0.5,
            )
    pdf.savefig(fig)
    plt.close(fig)

    fig, axs = genfig(4, len(product.profile_map) // 4, xlabel="X (px)", ylabel="Y (px)")
    for index, profile in enumerate(product.profile_map):
        axs[index].axis("off")
        axs[index].imshow(profile)
        axs[index].text(
            0.5,
            0.9,
            f"OUTPUT {index}",
            c="white",
            transform=axs[index].transAxes,
            ha="center",
            fontsize=8,
        )
        axs[index].set_xticks([])
        axs[index].set_yticks([])
    plt.subplots_adjust(wspace=0, hspace=0)
    pdf.savefig(fig)
    plt.close(fig)


def compute_bad_map_gravi(dark_map: NDArray,
                          dark_std: NDArray,
                          flats: Optional[List[NDArray]] = None,
                          bad_dark_factor: float = 30) -> NDArray:
    BADPIX_DARK = 1
    BADPIX_RMS = 2
    BADPIX_FLAT = 4
    ny, nx = dark_map.shape

    # --- High-frequency dark ---
    darkhf = median_filter(dark_map, size=(9, 9), mode="reflect") - dark_map

    # --- DARK thresholds ---
    dark_rms = np.median(dark_std)
    dark_mean = darkhf.mean()

    dark_max = dark_mean + 2 * bad_dark_factor * dark_rms
    dark_min = dark_mean - 2 * bad_dark_factor * dark_rms

    std_rms = dark_std.std()
    std_max = bad_dark_factor * std_rms  # + dark_rms
    std_min = 0.05 * std_rms

    # --- DARK + RMS masks ---
    bad_dark = (darkhf > dark_max) | (darkhf < dark_min)
    bad_rms = (dark_std > std_max) | (dark_std < std_min)

    bad_map = bad_dark.astype(int) * BADPIX_DARK +\
              bad_rms.astype(int) * BADPIX_RMS

    # --- FLAT processing ---
    count_flat = 0
    if flats:
        flat = np.zeros_like(dark_map, dtype=float)
        for cube in flats:
            _mean = np.mean(cube, axis=0)
            _std = np.std(cube, axis=0)
            _mask = np.abs(cube - _mean) < (5 * _std)
            img = np.sum(cube * _mask, axis=0) / np.clip(np.sum(_mask, axis=0), 1, None)
            img -= dark_map
            img -= np.quantile(img, 0.25)
            flat += img

        kx = 31
        flatmed = median_filter(flat, size=(1, kx), mode="reflect")

        flatmask = flatmed > 300

        xmask = np.zeros(nx, bool)
        xmask[5 : nx - 5] = True

        bad_flat = (flatmask) & (flat < 0.5 * flatmed) & xmask[None, :]

        bad_map += bad_flat.astype(np.int32) * BADPIX_FLAT
        count_flat = bad_flat.sum()

    # qc = dict(
    #     QC_BADPIX_SC=int(np.count_nonzero(bad_map)),
    #     QC_BADPIX_DARK_SC=int(bad_dark.sum()),
    #     QC_BADPIX_RMS_SC=int(bad_rms.sum()),
    #     QC_BADPIX_FLAT_SC=int(count_flat),
    #     FRACTION_BADPIX_SC=100 * np.count_nonzero(bad_map) / (nx * ny),
    # )

    return bad_map


@njit(fastmath=True, cache=True)
def gaussian(x, a, x0, sigma, c):
    return a * np.exp(-0.5 * ((x - x0) / sigma) ** 2) + c


@njit(fastmath=True, cache=True)
def gaussian_jacobian(x, a, x0, sigma):
    """
    Returns partial derivatives wrt (a, x0, sigma, c)
    """
    dx = x - x0
    inv_s2 = 1.0 / (sigma * sigma)
    e = np.exp(-0.5 * dx * dx * inv_s2)

    da = e
    dx0 = a * e * dx * inv_s2
    ds = a * e * dx * dx / sigma**3
    dc = np.ones_like(x)

    return da, dx0, ds, dc


@njit(fastmath=True, cache=True)
def fit_gaussian(x, y, p0, max_iter=6):
    """
    Fast nonlinear Gaussian fit using Gauss–Newton.
    """

    a, x0, sigma, c = p0
    for _ in range(max_iter):

        model = gaussian(x, a, x0, sigma, c)
        r = y - model

        da, dx0, ds, dc = gaussian_jacobian(x, a, x0, sigma)

        # Normal equations J^T J dp = J^T r
        JtJ = np.zeros((4, 4))
        Jtr = np.zeros(4)

        for i in range(x.size):
            Ji = np.array([da[i], dx0[i], ds[i], dc[i]])
            JtJ += np.outer(Ji, Ji)
            Jtr += Ji * r[i]

        # Damping for stability
        for i in range(4):
            JtJ[i, i] += 1e-3

        det = np.linalg.det(JtJ)
        if det == 0.0 or not np.isfinite(det):
            return p0, 1e8
        dp = np.linalg.solve(JtJ, Jtr)

        a += dp[0]
        x0 += dp[1]
        sigma += dp[2]
        c += dp[3]

        # Hard constraints
        if sigma < 0.3:
            sigma = 0.3
        # ---- Error estimate ----
        dof = max(1, x.size - 4)
        resid_var = np.sum(r * r) / dof
        cov = np.linalg.inv(JtJ) * resid_var
        x0_err = np.sqrt(cov[1, 1])

    return np.array([a, x0, sigma, c]), x0_err


def trace_spectrum(image, bad_mask, ref_x, ref_y, range_y):

    ny, nx = image.shape
    y_centers = np.full(nx, np.nan)
    y_errs = np.full(nx, np.nan)

    y_full = np.arange(ny)

    def _scan(start_x, end_x, step, p0):

        popt = p0.copy()
        y_curr = popt[1]

        for x in range(start_x, end_x, step):

            y0 = int(max(0, y_curr - range_y // 2))
            y1 = int(min(ny, y0 + range_y))

            valid = bad_mask[y0:y1, x] == 0
            if valid.sum() < 5:
                y_centers[x] = y_curr
                y_errs[x] = popt[2]
                continue

            y = y_full[y0:y1][valid].astype(np.float64)
            z = image[y0:y1, x][valid].astype(np.float64)

            if not (y[0] <= popt[1] <= y[-1]):
                popt[1] = y_curr = y.mean()

            new_popt, y_err = fit_gaussian(y, z, popt)

            if abs(new_popt[1] - y_curr) < 2.0 and y_err < 0.5:
                popt = new_popt
                y_curr = popt[1]

            y_centers[x] = y_curr
            y_errs[x] = y_err

        return popt

    col = image[:, ref_x]
    p0 = np.array(
        [col.max() - np.median(col), ref_y, 3, np.median(col)],
        dtype=np.float64,
    )

    _scan(ref_x, nx, 1, p0)
    _scan(ref_x - 1, -1, -1, p0)

    return y_centers, y_errs


def compute_bad_map(dark_cube: NDArray, threshold: float = 5.0) -> NDArray:
    """
    Build a simple bad-pixel map from dark-frame statistics.

    Args:
        dark_cube (NDArray): Dark cube with shape `(n_frame, ny, nx)`.
        threshold (float): Sigma threshold used for hot, dead, and noisy pixels.

    Returns:
        NDArray: Integer bad-pixel map with flags:
            1: Hot pixel (Mean > threshold)
            2: Dead pixel (Mean < threshold)
            4: Noisy pixel (StdDev > threshold)
    """
    mean_img = dark_cube.mean(axis=0)
    std_img = dark_cube.std(axis=0)
    return compute_bad_map_from_stats(mean_img, std_img, threshold=threshold)


def compute_bad_map_from_stats(
    mean_img: NDArray, std_img: NDArray, threshold: float = 5.0
) -> NDArray:
    """Build a bad-pixel map from precomputed temporal mean and RMS images."""
    BAD_HOT, BAD_DEAD, BAD_RON = 1, 2, 4

    med_mean = np.median(mean_img)
    med_rms = np.median(std_img)

    range_max = med_mean + threshold * med_rms
    range_min = med_mean - threshold * med_rms
    std_max = med_rms * threshold

    bpm = np.zeros(mean_img.shape, dtype=np.int16)

    bpm[mean_img > range_max] |= BAD_HOT
    bpm[mean_img < range_min] |= BAD_DEAD
    bpm[std_img > std_max] |= BAD_RON

    return bpm
