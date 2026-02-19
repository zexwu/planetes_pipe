import numpy as np

from typing import Any, List
from numba import njit
from scipy.signal import find_peaks, medfilt

from . import PipelineContext, arg, command, log
from .visualize import genfig, plt


@command("flat", "Build flat, dark, and profile images.", requires=[], produces=["flat"])
@arg("--profile_width", type=int, default=10, help="Width of the spectral profile in pixels.")
@arg("--bad_thres", type=float, default=5.0, help="Sigma threshold for bad pixel detection.")
def run_flat(ctx: PipelineContext, **kwargs: Any) -> None:
    """
    Creates Dark map, Bad Pixel map, Flat map, and traces spectral profiles.
    """
    log.info("--- Step: FLAT ---")
    log.info("Building DARK, BAD and FLAT maps...")
    # 1. Load and Compute Maps
    dark_cube = ctx.load_fits(ctx.conf["calib"]["dark"])
    dark_map = dark_cube.mean(axis=0)
    dark_std = dark_cube.std(axis=0)

    flat_files = ctx.conf["calib"]["flat"]
    flat_cubes = [ctx.load_fits(f) for f in flat_files]

    # bad_map = compute_bad_map_gravi(dark_map, dark_std, flat_cubes, 
    #                                 bad_dark_factor=20)
    bad_map = compute_bad_map(dark_cube, threshold=kwargs["bad_thres"])

    flat_maps = [img.mean(axis=0) - dark_map for img in flat_cubes]
    flat_map = np.mean(flat_maps, axis=0)

    # 2. Trace Spectra
    h, w = flat_map.shape
    h_half = h // ctx.n_reg // 2

    # Find peaks in the collapsed spatial direction
    spatial_profile = flat_map.mean(axis=1)
    thresh = np.percentile(spatial_profile, 75)
    ref_ys, props = find_peaks(spatial_profile, distance=h_half, height=thresh)
    log.debug(f"Found {len(ref_ys)} Peaks with THRESH {thresh:.0f}...")

    # Select the strongest peaks corresponding to the n_reg regions
    top_indices = np.argsort(props["peak_heights"])[-ctx.n_reg :]
    ref_ys = np.sort(ref_ys[top_indices])

    profile_map = np.empty((ctx.n_reg, h, w))
    xs_list, ys_list, ys_raw_list, ys_err_list = [], [], [], []
    profile_xs = []
    profile_ys = []

    log.info("Building PROFILE map...")
    for i, ref_y in enumerate(ref_ys):
        # Trace spectrum curvature
        log.debug(f"    Tracing spectrum {i+1}/{ctx.n_reg} at Y={ref_y}...")
        xs = np.arange(w)
        ys, ys_err = trace_spectrum(
            flat_map, bad_map, 
            ref_x=w // 2, ref_y=ref_y, range_y=h_half
        )

        # Smooth the trace
        ys_med = medfilt(ys, kernel_size=3)
        good = (ys_err < 0.5) & (np.arange(w) < w - 50) & (np.arange(w) > 50)
        poly_coeffs = np.polyfit(xs[good], ys_med[good], deg=3)
        ys_smoothed = np.polyval(poly_coeffs, xs)
        Y, X = np.indices((h, w))

        spatial_mask = np.abs(Y - ys_smoothed[X]) < kwargs["profile_width"] / 2
        profile_map[i, spatial_mask] = 1
        profile_map[i, bad_map > 0] = 0

        profile_xs.append(X[spatial_mask])
        profile_ys.append(Y[spatial_mask])

        xs_list.append(xs)
        ys_list.append(ys_smoothed)
        ys_raw_list.append(ys)
        ys_err_list.append(ys_err)

    xs_list = np.array(xs_list)
    ys_list = np.array(ys_list)
    ys_raw_list = np.array(ys_raw_list)
    ys_err_list = np.array(ys_err_list)

    # 3. Save Results
    ctx.save_product(
        "flat",
        # profile_map=profile_map,
        profile_xs=profile_xs,
        profile_ys=profile_ys,
        flat_map=flat_map,
        dark_map=dark_map,
        xs=xs_list,
        ys=ys_list,
    )
    log.info("Saved profile, flat, dark, and trace data.")

    # 4. Plotting
    if ctx.conf.get("plot_to_pdf", False):
        pdf = ctx.plot_ctx("flat.pdf")
        # Plot 1: Maps and Traces
        fig, axs = plt.subplots(1, 3, sharey=True)
        titles = ["DARK", "FLAT", "BAD"]
        for ax, img, title in zip(axs, [dark_map, flat_map, bad_map], titles):
            vmin, vmax = np.percentile(img, [0, 99])
            if title == "BAD": vmin, vmax = 0, 1
            ax.imshow(img, vmin=vmin, vmax=vmax)
            ax.set_title(title)
        pdf.savefig(fig); plt.close(fig) # fmt: skip

        # Plot 2: Per-Telescope Flats
        fig, axs = plt.subplots(1, 4)
        vmin, vmax = np.percentile(flat_map, [1, 99])
        for i, dat in enumerate(flat_maps):
            axs[i].imshow(dat, vmin=vmin, vmax=vmax)
            axs[i].set_title(f"Tel {ctx.telescopes[i]}")
            axs[i].axis("off")
            for reg in range(ctx.n_reg):
                axs[i].plot(
                    xs_list[reg], ys_list[reg],
                    c="blue", lw=0.3, ls="-", alpha=1
                )
                axs[i].plot(
                    xs_list[reg][ys_err_list[reg] < 5],
                    ys_raw_list[reg][ys_err_list[reg] < 5],
                    c="red", lw=0.3, ls="--", alpha=0.5,
                )
        pdf.savefig(fig); plt.close(fig) # fmt: skip

        fig, axs = genfig(4, ctx.n_reg // 4, xlabel="X (px)", ylabel="Y (px)")
        for k in range(ctx.n_reg):
            axs[k].axis("off")
            axs[k].imshow(profile_map[k, :, :])
            axs[k].text(
                0.5, 0.9, f"OUTPUT {k}",
                c="white", transform=axs[k].transAxes, 
                ha="center", fontsize=8,
            )
            axs[k].set_xticks([])
            axs[k].set_yticks([])
        plt.subplots_adjust(wspace=0, hspace=0)
        pdf.savefig(fig); plt.close(fig) # fmt: skip
        pdf.close()

    log.info("--- Step: FLAT [DONE] ---")
    log.info("")
    return


def compute_bad_map_gravi(dark_map: np.ndarray, 
                          dark_std: np.ndarray, 
                          flats: Optional[List[np.ndarray]] = None, 
                          bad_dark_factor: float = 30) -> np.ndarray:
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

    qc = dict(
        QC_BADPIX_SC=int(np.count_nonzero(bad_map)),
        QC_BADPIX_DARK_SC=int(bad_dark.sum()),
        QC_BADPIX_RMS_SC=int(bad_rms.sum()),
        QC_BADPIX_FLAT_SC=int(count_flat),
        FRACTION_BADPIX_SC=100 * np.count_nonzero(bad_map) / (nx * ny),
    )

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


def compute_bad_map(dark_cube: np.ndarray, threshold: float = 5.0) -> np.ndarray:
    """
    Creates a bad pixel map from a dark cube using statistical outliers.

    Args:
        dark_cube (np.ndarray): Cube of dark frames (frames, y, x).
        bad_threshold (float): Sigma threshold for detecting bad pixels.

    Returns:
        np.ndarray: 2D integer map with flags:
            1: Hot pixel (Mean > threshold)
            2: Dead pixel (Mean < threshold)
            4: Noisy pixel (StdDev > threshold)
    """
    BAD_HOT, BAD_DEAD, BAD_RON = 1, 2, 4

    mean_img = dark_cube.mean(axis=0)
    std_img = dark_cube.std(axis=0)

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
