from typing import Any, List, Optional, Tuple

import numpy as np
from numba import njit
from scipy.signal import find_peaks

from . import PipelineContext, arg, command, log
from .visualize import genfig, plt
from .preproc import extract_spec_sparse


@command("wave", "Wavelength calibration step",
         requires=["flat"],
         produces=["wave"])
@arg("--sigma", type=int, default=5, help="Line detection threshold")
@arg("--deg", type=int, default=3, help="Degree of polynomial for wavemap fit")
@arg("--aber-deg", type=int, default=3, help="Degree of polynomial for aberration fit")
@arg("--disp-deg", type=int, default=3, help="Degree of polynomial for pixel-to-wavelength fit")
@arg("--niter", type=int, default=int(1e6), help="Number of iterations for RANSAC line matching")
def run_wave(ctx: PipelineContext, **kwargs: Any) -> None:

    log.info("--- Step: WAVE ---")

    # Load flat Data
    with ctx.load_product("flat") as d:
        profile_xs = d["profile_xs"]
        profile_ys = d["profile_ys"]
        flat_map = d["flat_map"]
        ny, nx = flat_map.shape
        x_trace, y_trace = d["xs"], d["ys"]

    wave_dark = ctx.load_fits(ctx.conf["calib"]["wave_dark"])
    wave_cube = ctx.load_fits(ctx.conf["calib"]["wave"])
    wave_img  = wave_cube.mean(axis=0) - wave_dark.mean(axis=0)
    wave_spec = extract_spec_sparse(wave_img[None, :, :],
                                    profile_ys, profile_xs)[:, 0, :]


    # --- 2. Identify and match lines in the mean spectrum ---
    # Known laboratory emission lines and their relative intensities

    # fmt: off
    line_height=np.array([10, 20, 6, 12, 42, 8,
                          10, 33, 8, 12, 16, 29,
                          25, 21, 58, 29, 83, 8,
                          25, 8, 100, 7, 7, 10])
    line_pos = np.array([ 965.78, 1047.01, 1148.81, 1211.23,
                         1243.93, 1270.23, 1280.27, 1295.67,
                         1300.83, 1321.40, 1322.81, 1323.09,
                         1327.26, 1331.32, 1336.71, 1340.65,
                         1350.42, 1359.93, 1362.27, 1367.86,
                         1371.86, 1382.57, 1390.75, 1409.37])

    # NOTE: the following lines are added based on the NIST database
    add = [ 978.450, 1067.357, 1166.871, 1213.974,
           1234.339, 1240.283, 1245.612, 1248.766]
    # fmt: on

    # NOTE: the lines at 1322.81 and 1323.09 are removed
    line_height = line_height[~np.isin(line_pos, [1322.81, 1323.09])]
    line_pos = line_pos[~np.isin(line_pos, [1322.81, 1323.09])]

    line_pos = np.concatenate([line_pos, add]) / 1e3  # nm -> um
    line_pos.sort()

    # lines in 2.0-2.4 um range; for GRAVITY data
    # line_height = np.array([60, 160, 90, 1800, 50, 30, 30, 1500, 130, 510, 240, 169, 64, 73]) # fmt: skip
    # line_pos = np.array([1.982291, 1.997118, 2.032256, 2.062186, 2.065277, 2.073922, 2.081672, 2.099184, 2.133871, 2.154009, 2.208321, 2.313952, 2.385154, 2.397306]) # fmt: skip

    matched_ind_all = []
    x_centroids = []  # To store x-centroids for all lines
    y_centroids = []  # To store y-centroids for all lines
    y_heights = []  # To store line heights for weighting
    wl_truth = []  # To store wavelengths for all matched lines
    trans_all = []  # To store transformations for all regions
    spec_mean = wave_spec[-4:].mean(axis=0)

    # line detection threshold: median + 5*MAD
    def thresh(_spec: np.ndarray) -> np.floating[Any]:
        med = np.median(_spec)
        mad = np.median(abs(_spec - med))
        return med + kwargs["sigma"] * mad

    peak_pos_mean, _ = find_peaks(spec_mean, height=thresh(spec_mean), distance=5)
    matched_ind_mean, pixel_to_wl_mean =\
    match_lines(line_pos, peak_pos_mean, tol=3, n_iter=kwargs["n-iter"])
    log.info("Matched mean spectrum peaks to known lines: "
             f"{len(peak_pos_mean)} peaks -> {len(matched_ind_mean)} matches.")

    if ctx.conf.get("plot_to_pdf", False):
        pdf = ctx.plot_ctx("wave.pdf")
        fig, ax = plt.subplots()
        ax.imshow(wave_img, vmax=np.percentile(wave_img, 99), vmin=0)
        ax.set_xlabel("X [px]"); ax.set_ylabel("Y [px]")
        pdf.savefig(fig); plt.close(fig)

    for reg in range(ctx.n_reg):
        spec = wave_spec[reg]

        # Find peaks in the spectrum of individual region;
        peak_pos, _ = find_peaks(spec, height=thresh(spec), distance=5)

        # Add sub-pixel refinement for the detected peaks;
        # assume the log(line) is approximated by a parabola near the peak
        pn1, pct, pp1 = spec[peak_pos - 1], spec[peak_pos], spec[peak_pos + 1]
        _pn1, _pct, _pp1 = np.log(abs(pn1)+1e-8), \
                           np.log(abs(pct)+1e-8), \
                           np.log(abs(pp1)+1e-8)

        good = (pn1 > 1) & (pp1 > 1) & (pct > 1) &\
               (abs(2 * _pct - _pp1 - _pn1) > 1)

        p_shift = np.zeros_like(peak_pos, dtype=float)
        p_shift[good] = 0.5 * (_pp1 - _pn1)[good] / (2 * _pct - _pp1 - _pn1)[good]
        peak_pos = 1.0 * peak_pos + p_shift

        peak_height = pct
        peak_height[good] = (_pct * np.exp(0.25 * (_pn1 - _pp1) * p_shift))[good]

        # match the observed peaks to the mean spectrum peaks
        # to filter out spurious detections
        matched_ind, trans = match_lines_grid(
            peak_pos_mean, peak_pos, tol=1,
            offset_limit=15, curve_limit=1e-3, slope_bounds=(0.99, 1.01),
            steps_per_unit=4,
        )

        both = list(set(matched_ind[:, 0]) & set(matched_ind_mean[:, 1]))
        log.info(
                f"OUTPUT {reg:02d}: {len(peak_pos)} peaks ∩ {len(line_pos)} "
                f"known lines -> {len(both)} matched."
        )


        # idx_wave: indices of the matched lines in the known line list
        # idx_peak: indices of the matched peaks in the observed spectrum
        idx_wave = np.where(np.isin(matched_ind_mean[:, 1], both))[0]
        idx_peak = np.where(np.isin(matched_ind[:, 0], both))[0]
        idx_wave = matched_ind_mean[idx_wave, 0]
        idx_peak = matched_ind[idx_peak, 1]

        matched_ind_all.append(np.c_[idx_wave, idx_peak])

        _x = peak_pos[idx_peak]
        x_centroids.append(_x)
        y_centroids.append(np.interp(_x, x_trace[reg], y_trace[reg]))
        y_heights.append(peak_height[idx_peak])
        wl_truth.append(line_pos[idx_wave])
        trans_all.append(trans)

    # Flatten arrays for fitting: we have a set of (x, y, wl) points.
    x_flat = np.concatenate(x_centroids)
    y_flat = np.concatenate(y_centroids)
    idx2reg = np.hstack([[i] * len(c) for i, c in enumerate(x_centroids)])
    reg_offsets = np.cumsum([0] + [len(c) for c in x_centroids])
    wl_flat = np.concatenate(wl_truth)
    ht_flat = np.concatenate(y_heights)

    # helper function to build the design matrix for 2D polynomial fitting
    def dmat_arr(x, y, max_deg: int = 3) -> np.ndarray:
        terms = [x ** (d - i) * y**i for d in range(max_deg + 1) for i in range(d + 1)]
        return np.array(terms)

    def dmat(x, y, max_deg: int = 3) -> np.ndarray:
        terms = [x ** (d - i) * y**i for d in range(max_deg + 1) for i in range(d + 1)]
        return np.column_stack(terms)

    # Fit the wl = f(x, y), using a 3rd order polynomial
    deg = kwargs["deg"]
    log.info("Fitted 2D wavelength solution.")
    design_matrix = dmat(x_flat, y_flat, max_deg=deg)
    terms = [f"x{d-i}y{i}" for d in range(deg+1) for i in range(d + 1)]
    log.debug("Kernel used for wavemap fit: ")
    log.debug(", ".join(terms))
    wavemap_coeffs, _, _, _ = np.linalg.lstsq(design_matrix, wl_flat, rcond=None)
    residuals = wl_flat - design_matrix @ wavemap_coeffs
    std = np.std(residuals)
    std0 = std

    log.info(f"Initial std dev of residuals: {std * 1e3:.2f} nm")
    for _ in range(3):
        log.info(f"Performing outlier rejection [{_ + 1}/3]")
        reject = abs(residuals) > 3 * std
        wavemap_coeffs = np.linalg.lstsq(design_matrix[~reject], wl_flat[~reject], rcond=None)[0]
        residuals = wl_flat - design_matrix @ wavemap_coeffs
        std = np.std(residuals[~reject])
        log.info(
            f"    Std dev of residuals: {std * 1e3:.2f} nm "
            f"after rejecting {reject.sum()} outliers"
        )

    # outliers at each region
    outliers = [reject & (idx2reg == reg) for reg in range(ctx.n_reg)]
    outliers = [np.where(outliers[reg])[0] - reg_offsets[reg]
                for reg in range(ctx.n_reg)]

    # --- 5. Build and save the full wavemap ---
    det_x_coords = np.repeat(np.arange(nx)[None, :], ctx.n_reg, axis=0)
    det_y_coords = y_trace
    full_design_matrix = dmat_arr(det_x_coords, det_y_coords)
    wavemap = np.einsum("i,iyx->yx", wavemap_coeffs, full_design_matrix)
    ctx.save_product("wave", wave_map=wavemap)
    log.info("Saved full wavemap")

    if ctx.conf.get("plot_to_pdf", False):
        log.info("Generating diagnostic plots...")
        fig, ax = plt.subplots()
        norm = plt.Normalize(vmin=wl_flat.min(), vmax=wl_flat.max())
        ax.scatter(x_flat, y_flat, c=wl_flat, cmap="viridis", s=15, norm=norm)
        cbar = plt.colorbar(ax.collections[0], ax=ax)
        cbar.set_label("Wavelength (um)")
        ax.set_xlabel("X [px]")
        ax.set_ylabel("Y [px]")
        ax.set_title("True Wavelengths (um)")
        pdf.savefig(fig); plt.close(fig)

        fig, ax = plt.subplots()
        im = ax.imshow(wavemap, norm=norm, cmap="viridis")
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Wavelength (um)")
        ax.set_xlabel("X [px]")
        ax.set_ylabel("OUTPUT")
        ax.set_title("Fitted Wavemap (um)")
        pdf.savefig(fig); plt.close(fig)

        fig, ax = plt.subplots()
        x_coords = np.repeat(np.arange(nx)[None, :], ny, axis=0)
        y_coords = np.repeat(np.arange(ny)[:, None], nx, axis=1)

        full_design_matrix = dmat_arr(x_coords, y_coords)
        _wavemap = np.einsum("i,iyx->yx", wavemap_coeffs, full_design_matrix) * 1e3
        dwavemap = _wavemap - _wavemap.mean(axis=0)[None, :]
        ddwavemap = dwavemap - dwavemap.mean(axis=1)[:, None]

        norm = plt.Normalize(vmin=-5 * std * 1e3, vmax=5 * std * 1e3)
        im = ax.imshow(ddwavemap, norm=norm, cmap="viridis")
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("dWavelength (nm)")
        ax.set_xlabel("X [px]")
        ax.set_ylabel("Y [px]")
        ax.set_title("Aberration Map:= Fitted Wavemap - (a*X+b*Y)")
        pdf.savefig(fig); plt.close(fig)

        fig, ax = plt.subplots()
        norm = plt.Normalize(vmin=-5e3 * std, vmax=5e3 * std)
        ax.scatter(x_flat, y_flat,
                   c="lime", marker="x", label="All Points",
                   lw=0.3, s=10, alpha=0.5)
        ax.scatter(x_flat[reject], y_flat[reject],
                   ec="r", fc="none", label="Rejected Outliers",
                   lw=0.7, s=30)
        ax.imshow(wave_img, vmax=np.percentile(wave_img, 99.5), vmin=1)
        ax.set_xlabel("X [px]")
        ax.set_ylabel("Y [px]")
        ax.set_title("Wavelength Residuals (nm)")
        ax.legend()
        pdf.savefig(fig); plt.close(fig)

        fig, ax = plt.subplots()
        ax.hist(residuals[~reject] * 1e3,
                histtype="step", bins=50, label="All Points")
        ax.hist(residuals[reject] * 1e3,
                histtype="step", bins=50, label="Rejected Outliers")
        ax.text(0.5, 0.8, f"Std Dev: {1e3 * std:.3f} nm",
                color="C0", transform=ax.transAxes, ha="center")
        ax.set_xlabel("Fitted Residuals (nm)")
        pdf.savefig(fig); plt.close(fig)

        fig, axs = plt.subplots(1, 2)
        ax = axs[0]
        norm = plt.Normalize(vmin=y_flat.min(), vmax=y_flat.max())
        ax.scatter(x_flat, residuals * 1e3, s=15,
                   c=y_flat, cmap="viridis", norm=norm, label="All Points")
        ax.scatter(x_flat[reject], residuals[reject] * 1e3, s=50,
                   ec="r", fc="none", lw=0.7, label="Rejected Outliers")
        ax.set_xlabel("X [px]")
        ax.set_ylabel("Fitted Residuals (nm)")
        cbar = plt.colorbar(ax.collections[0], ax=ax)
        cbar.set_label("Y Position (px)")
        ax.set_ylim(-1e4 * std, 1e4 * std)

        ax = axs[1]
        norm = plt.Normalize(vmin=x_flat.min(), vmax=x_flat.max())
        ax.scatter(y_flat, residuals * 1e3, s=15,
                   c=y_flat, cmap="viridis", norm=norm, label="All Points")
        ax.scatter(y_flat[reject], residuals[reject] * 1e3, s=50,
                   ec="r", fc="none", lw=0.7, label="Rejected Outliers")
        ax.set_xlabel("Y [px]")
        cbar = plt.colorbar(ax.collections[0], ax=ax)
        cbar.set_label("X Position (px)")
        ax.set_ylim(-1e4 * std, 1e4 * std)
        ax.legend()
        pdf.savefig(fig); plt.close(fig)

        for reg in range(ctx.n_reg):
            if reg % 4 == 0:
                fig, axs =\
                genfig(4, 1, xlabel="X [px]", ylabel="Intensity [adu]", sharey=False)

            ax = axs[reg % 4]
            trans = trans_all[reg]
            spec = wave_spec[reg]
            ax.plot(spec, color="k", lw=0.3)
            for _i, _l in enumerate(x_centroids[reg]):
                ax.axvline(_l, color="r", ls="-", alpha=0.5, lw=0.3)
                if _i in outliers[reg]:
                    ax.text(_l, spec.max() * 0.8, "OUTLIER",
                            alpha=0.7, rotation=90, fontsize=6, zorder=10,
                            va="top", ha="center", color="red",
                            bbox=dict(facecolor="white", edgecolor="none", pad=0))

            for _i, _l in enumerate(line_pos):
                # 1. transform form lines to mean_spec
                # 2. transform from mean_spec to individual_spec
                _l_tr1 = _l**2 * pixel_to_wl_mean[0] +\
                         _l**1 * pixel_to_wl_mean[1] +\
                         _l**0 * pixel_to_wl_mean[2]
                _l_tr2 = _l_tr1**2 * trans[0] +\
                         _l_tr1**1 * trans[1] +\
                         _l_tr1**0 * trans[2]
                if _l_tr2 < 0 or _l_tr2 > nx: continue
                if _i in idx_wave:
                    # Matched lines: blue dashed line + label
                    ax.axvline(_l_tr2, color="b", ls="--", alpha=0.7, lw=0.3)
                    ax.text(_l_tr2, spec.max() * 0.8, f"{_l:.4f}",
                            va="top", ha="center", rotation=90,
                            fontsize=6, color="blue",
                            bbox=dict(facecolor="white", edgecolor="none", pad=0))
                else:
                    # Unmatched lines: gray solid line + label
                    ax.axvline(_l_tr2, color="k", ls="-", alpha=0.2, lw=0.3)
                    ax.text(_l_tr2, spec.max() * 0.8, f"{_l:.4f}",
                            alpha=0.5, rotation=90, fontsize=6,
                            va="top", ha="center", color="gray",
                            bbox=dict(facecolor="white", edgecolor="none", pad=0))
            ax.set_xlim(0, nx)
            nmatch = len(matched_ind_all[reg])
            ax.text(0.1, 0.8, f"OUTPUT{reg}: matched {nmatch} lines",
                    transform=ax.transAxes, c="b")
            if (reg + 1) % 4 == 0:
                pdf.savefig(fig); plt.close(fig)

    # --- 6. Build wavemap using Aberration Model ---

    log.info("Building Wavemap using Aberration Model...")
    # Find the matched lines across all regions
    matched_ind_common = set(range(len(line_pos)))
    for matched in matched_ind_all:
        matched_ind_common = matched_ind_common & set(matched[:, 0])
    x_centroids_common = []
    y_centroids_common = []

    for reg, matched in enumerate(matched_ind_all):
        _common = np.isin(matched[:, 0], list(matched_ind_common))
        _common = np.where(_common)[0]
        x_centroids_common.append(x_centroids[reg][_common])
        y_centroids_common.append(y_centroids[reg][_common])
    x_centroids_common = np.array(x_centroids_common)
    y_centroids_common = np.array(y_centroids_common)

    # flatten the arrays for fitting:
    # we have a set of (x, y, wl) points.
    x_flat = np.concatenate(x_centroids_common)
    y_flat = np.concatenate(y_centroids_common)
    wl_grid = line_pos[list(matched_ind_common)]
    wl_flat = np.tile(wl_grid, ctx.n_reg)

    # fit a 1D polynomial to the mean spec
    # to get the pixel-to-wavelength mapping
    # along the dispersion direction
    log.info("Fitting 1D pixel-to-wave mapping for the mean trace...")
    x_centroids_mean = x_centroids_common.mean(axis=0)
    disp_deg = kwargs["disp_deg"]
    pixel_to_wl_coeffs = np.polyfit(x_centroids_mean, wl_grid, deg=disp_deg)
    disp_std = np.std(wl_grid - np.polyval(pixel_to_wl_coeffs, x_centroids_mean))
    px2wl_str = [f"{coef*1e3:.1e}*x{3 - i}" for i, coef in enumerate(pixel_to_wl_coeffs)]
    log.debug("Pixel-to-wave mapping: " + " + ".join(px2wl_str) + " nm")
    log.info(f"Std dev of pixel-to-wave fit: {disp_std * 1e3:.2f} nm")

    # fit a 2D polynomial to the deviations from the mean spec for the aberrations
    log.info("Fitting 2D polynomial f(x,y) to aberration Δx...")
    log.debug("Kernel used for aberration fit: ")
    aber_deg = kwargs["deg"]
    ker = [f"x{d-i}y{i}" for d in range(aber_deg+1) for i in range(d + 1)]
    log.debug(", ".join(ker))
    aber_design_matrix = dmat(x_flat, y_flat, max_deg=aber_deg)
    dx = (x_centroids_common - x_centroids_mean[None, :]).flatten()
    aber_coeffs, *_ = np.linalg.lstsq(aber_design_matrix, dx, rcond=None)

    # Residuals for the Aberration fit
    dx_fit = aber_design_matrix @ aber_coeffs
    aber_std = np.std(dx - dx_fit)
    log.info(f"Std dev of aberration fit: {aber_std:.3f} px")
    x_centroids_fit = x_flat - dx_fit
    wl_flat_fit = np.polyval(pixel_to_wl_coeffs, x_centroids_fit)
    residuals = wl_flat - wl_flat_fit
    std = np.std(residuals)
    log.info(f"Std dev of wavemap fit: {std * 1e3:.2f} nm")

    # Apply the Aberration model to create Aberration map
    full_aber_design_matrix = dmat_arr(det_x_coords, det_y_coords, max_deg=aber_deg)
    aber_offset = np.einsum("i,iyx->yx", aber_coeffs, full_aber_design_matrix)

    # Apply the 1D wl solution to get the final wavemap with aberration correction
    wavemap_aber = np.polyval(pixel_to_wl_coeffs, det_x_coords - aber_offset)
    ctx.save_product("wave_aberr", wavemap=wavemap_aber)
    log.info("Saved aberration wavemap")


    log.info("--- Step: WAVE [DONE] ---")
    log.info("")

    if ctx.conf.get("plot_to_pdf", False):
        fig, ax = plt.subplots()
        norm = plt.Normalize(vmin=wl_flat.min(), vmax=wl_flat.max())
        ax.scatter(x_flat, y_flat,
                   c=wl_flat, cmap="viridis", s=15, norm=norm)
        ax.set_xlabel("X [px]")
        ax.set_ylabel("Y [px]")
        pdf.savefig(fig); plt.close(fig)

        fig, ax = plt.subplots()
        ax.hist(residuals * 1e3, histtype="step", bins=50, label="Fit Residuals")
        ax.set_xlabel("Wavelength Error (nm)")
        pdf.savefig(fig); plt.close(fig)

        fig, ax = plt.subplots()
        norm = plt.Normalize(vmin=y_flat.min(), vmax=y_flat.max())
        ax.scatter(x_flat, residuals * 1e3,
                   c=y_flat, cmap="viridis", s=15, norm=norm)
        ax.set_xlabel("X [px]")
        ax.set_ylabel("Wavelength Residual (nm)")
        ax.set_ylim(-1e4 * std, 1e4 * std)
        pdf.savefig(fig); plt.close(fig)

        fig, ax = plt.subplots()
        norm = plt.Normalize(vmin=x_flat.min(), vmax=x_flat.max())
        ax.scatter(y_flat, residuals * 1e3,
                   c=x_flat, cmap="viridis", s=15, norm=norm)
        ax.set_xlabel("Y [px]")
        ax.set_ylabel("Wavelength Residual (nm)")
        ax.set_ylim(-1e4 * std, 1e4 * std)
        pdf.savefig(fig); plt.close(fig)

        x_coords = np.repeat(np.arange(nx)[None, :], ny, axis=0)
        y_coords = np.repeat(np.arange(ny)[:, None], nx, axis=1)
        full_aber_design_matrix = dmat_arr(x_coords, y_coords, max_deg=aber_deg)
        dx_fit = np.einsum("i,iyx->yx", aber_coeffs, full_aber_design_matrix)
        _wavemap_aber =\
        np.polyval(pixel_to_wl_coeffs, x_coords - dx_fit) * 1e3

        dwavemap_aber = _wavemap_aber - _wavemap_aber.mean(axis=0)[None, :]
        ddwavemap_aber = dwavemap_aber - dwavemap_aber.mean(axis=1)[:, None]

        fig, ax = plt.subplots()
        norm = plt.Normalize(vmin=-5 * std0 * 1e3, vmax=5 * std0 * 1e3)
        im = ax.imshow(ddwavemap_aber, norm=norm, cmap="viridis")
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("dWavelength (nm)")
        ax.set_xlabel("X [px]")
        ax.set_ylabel("Y [px]")
        pdf.savefig(fig); plt.close(fig)

        fig, ax = plt.subplots()
        norm = plt.Normalize(vmin=-0.3, vmax=0.3)
        im = ax.imshow(_wavemap - _wavemap_aber, norm=norm, cmap="coolwarm")
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Wavelength Difference (nm)")
        ax.set_xlabel("X [px]")
        ax.set_ylabel("Y [px]")
        ax.set_title("Wavemap(method1) - Wavemap(method2)")
        pdf.savefig(fig); plt.close(fig)

    if ctx.conf.get("plot_to_pdf", False):
        pdf.close()

    return


def match_lines(
    theoretical_lines: List[float],
    obs_peaks: List[float],
    tol: float = 2.0,
    min_inliers: int = 5,
    n_iter: int = 10000,
) -> Tuple[np.ndarray, Tuple[float, ...]]:

    # 1. Setup Data
    obs = np.sort(np.asarray(obs_peaks, dtype=np.float64))
    mod = np.sort(np.asarray(theoretical_lines, dtype=np.float64))

    n_obs, n_mod = len(obs), len(mod)
    if n_obs < 3 or n_mod < 3: raise RuntimeError("Not enough points to match.")

    # 2. Run the ransac Kernel
    best_coeffs = _ransac_kernel(obs, mod, n_iter, tol, n_mod, n_obs)

    # 3. Reconstruct the Matches
    a, b, c = best_coeffs
    pred_pixels = a * mod**2 + b * mod + c

    final_inliers = []
    obs_ptr = 0
    for m_idx in range(n_mod):
        target = pred_pixels[m_idx]
        while obs_ptr < n_obs and obs[obs_ptr] < target - tol:
            obs_ptr += 1
        if obs_ptr < n_obs and abs(obs[obs_ptr] - target) < tol:
            final_inliers.append((m_idx, obs_ptr))
            obs_ptr += 1

    if len(final_inliers) < min_inliers:
        msg = f"line matching failed: only {len(final_inliers)} matches found."
        raise RuntimeError(msg)

    # 4. Final Fit
    # Refine the quadratic using ALL inliers (Least Squares)
    final_mi = [x[0] for x in final_inliers]
    final_oi = [x[1] for x in final_inliers]

    final_coeffs_refined = np.polyfit(mod[final_mi], obs[final_oi], 2)

    return np.array(final_inliers), tuple(final_coeffs_refined)


def match_lines_grid(
    theoretical_lines: List[float],
    obs_peaks: List[float],
    tol: float = 1,
    min_inliers: int = 6,
    offset_limit: float = 20,
    slope_bounds: Tuple[float, float] = (0.98, 1.02),
    curve_limit: float = 1e-3,
    steps_per_unit: float = 2.0,  # Density of search. Higher = more robust but slower.
) -> Tuple[np.ndarray, Tuple[float, ...]]:
    # 1. Prepare Data
    obs = np.sort(np.ascontiguousarray(obs_peaks, dtype=np.float64))
    mod = np.sort(np.ascontiguousarray(theoretical_lines, dtype=np.float64))

    # 2. Auto-Calculate Grid Steps
    # We need step sizes small enough so we don't "step over" a valid match.
    # Rule of thumb: Step size should cause a shift of ~0.5 * tol

    # Offset (c) steps: 0.5 pixels
    c_step = tol / steps_per_unit
    c_grid = np.arange(-offset_limit, offset_limit + c_step, c_step)

    # Slope (b) steps:
    # A change in slope d_b shifts the last pixel by d_b * max_wave.
    # We want d_b * max_wave ~ 0.5 * tol  =>  d_b = 0.5 * tol / max_wave
    max_wave = np.max(np.abs(mod)) if len(mod) > 0 else 1000.0
    b_step = (tol / 2.0) / max_wave
    b_grid = np.arange(slope_bounds[0], slope_bounds[1] + b_step, b_step)

    # Curvature (a) steps:
    # Usually we can search just 3 points: [-limit, 0, +limit] or coarse steps
    # If a is very small, we might skip it in the grid and solve it in the final fit.
    # Let's do a coarse grid for robustness.
    a_step = curve_limit / 2.0  # 5 steps total
    a_grid = np.arange(-curve_limit, curve_limit + a_step / 10.0, a_step)  # small epsilon

    # Sanity check on grid size
    total_iters = len(a_grid) * len(b_grid) * len(c_grid)
    if total_iters > 2e6:
        print(f"Warning: Grid search is large ({total_iters} iterations).")

    # 3. Run Grid Search
    # This finds the "Best Approximate" parameters
    a_best, b_best, c_best =\
    _grid_search_kernel(obs, mod, tol, a_grid, b_grid, c_grid)

    # 4. Final Refinement (The "Polish" Step)
    # Use the approximate grid parameters to identify inliers, then run Least Squares
    pred = a_best * mod**2 + b_best * mod + c_best
    matches = []
    ptr = 0
    for i, target in enumerate(pred):
        while ptr < len(obs) and obs[ptr] < target - tol:
            ptr += 1
        # Check if match
        if ptr < len(obs) and abs(obs[ptr] - target) < tol:
            matches.append((i, ptr))

    if len(matches) < min_inliers:
        raise RuntimeError(f"Grid search failed: found {len(matches)} matches.")

    final_idx = np.array(matches)

    # 5. Polyfit on the identified inliers for sub-pixel accuracy
    final_coeffs = np.polyfit(mod[final_idx[:, 0]], obs[final_idx[:, 1]], 2)

    return final_idx, tuple(final_coeffs)


@njit(fastmath=True, cache=True)
def _ransac_kernel(obs: np.ndarray, mod: np.ndarray,
                   n_iter: int, tol: float, n_mod: int,
                   n_obs: int) -> Tuple[float, float, float]:
    """
    The heavy lifting kernel compiled to machine code.
    Returns the best coefficients found.
    """
    best_count = -1
    best_coeffs = (0.0, 0.0, 0.0)

    # Pre-calculate min/max for monotonicity check optimization
    mod_min = mod[0]
    mod_max = mod[-1]
    np.random.seed(0)  # For reproducibility in Numba (uses global state)

    for _ in range(n_iter):
        # 1. Fast Random Selection of 3 unique indices
        # (Faster than numpy.choice with replace=False inside a loop)
        i1 = np.random.randint(0, n_mod)
        i2 = np.random.randint(0, n_mod)
        while i1 == i2:
            i2 = np.random.randint(0, n_mod)
        i3 = np.random.randint(0, n_mod)
        while i3 == i1 or i3 == i2:
            i3 = np.random.randint(0, n_mod)

        # Sort indices to ensure stability
        # (Though not strictly necessary for solving, good for consistency)
        if i1 > i2: i1, i2 = i2, i1
        if i2 > i3: i2, i3 = i3, i2
        if i1 > i2: i1, i2 = i2, i1

        # Correct sampling based on original logic:
        # Sample 3 unique MOD indices
        idxs_m = np.array([i1, i2, i3])
        m_samp = mod[idxs_m]

        # Sample 3 unique OBS indices
        o1 = np.random.randint(0, n_obs)
        o2 = np.random.randint(0, n_obs)
        while o1 == o2: o2 = np.random.randint(0, n_obs)
        o3 = np.random.randint(0, n_obs)
        while o3 == o1 or o3 == o2: o3 = np.random.randint(0, n_obs)
        idxs_o = np.array([o1, o2, o3])
        # Sort the OBS samples to match the sorted MOD samples (assuming monotonic mapping)
        # The original code did `sort(rng.choice)`,
        # implying it matches lowest-to-lowest, etc.
        idxs_o_sorted = np.sort(idxs_o)
        p_samp = obs[idxs_o_sorted]

        # 2. Solve Quadratic Equation (Vandermonde)
        # We solve: A * c = y  ->  c = inv(A) * y
        # Hardcoded 3x3 solver is faster, but linalg.solve is acceptable in Numba
        x1, x2, x3 = m_samp[0], m_samp[1], m_samp[2]
        y1, y2, y3 = p_samp[0], p_samp[1], p_samp[2]

        # Denominator for Cramer's rule / Lagrange (avoids matrix overhead)
        denom = (x1 - x2) * (x1 - x3) * (x2 - x3)
        if denom == 0: continue  # Should not happen with unique floats

        # Lagrange Interpolation / Analytic quadratic fit
        # A bit verbose but very fast (no matrix alloc)
        a = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / denom
        b = (x3 * x3 * (y1 - y2) + x2 * x2 * (y3 - y1) + x1 * x1 * (y2 - y3)) / denom
        c = (x2 * x3 * (x2 - x3) * y1 +\
             x3 * x1 * (x3 - x1) * y2 +\
             x1 * x2 * (x1 - x2) * y3) / denom

        # 3. Monotonicity Check Optimization
        # Derivative d(Pix)/d(Wave) = 2*a*x + b
        # Since this is linear, we only check the ENDPOINTS of the range.
        # If it's > 0 at min and max, it's > 0 everywhere in between.
        if (2 * a * mod_min + b <= 0) or (2 * a * mod_max + b <= 0):
            continue

        # 4. Inlier Counting (Zero Allocation)
        # Instead of building a list, we just count. Memory allocation is slow.
        current_inliers = 0
        obs_ptr = 0

        for m_idx in range(n_mod):
            # Calculate prediction on the fly (no array allocation)
            mx = mod[m_idx]
            target = a * mx**2 + b * mx + c

            # Fast scan (Two pointers)
            while obs_ptr < n_obs and obs[obs_ptr] < target - tol:
                obs_ptr += 1

            if obs_ptr < n_obs:
                diff = obs[obs_ptr] - target
                # abs(diff) < tol
                if diff < tol and diff > -tol:
                    current_inliers += 1
                    obs_ptr += 1

        if current_inliers > best_count:
            best_count = current_inliers
            best_coeffs = (a, b, c)

    return best_coeffs


@njit(fastmath=True, cache=True)
def _grid_search_kernel(
    obs: np.ndarray ,
    mod: np.ndarray,
    tol: int ,
    a_grid: np.ndarray,
    b_grid: np.ndarray,
    c_grid: np.ndarray
) -> Tuple[float, float, float]:
    """
    Scans the parameter grid and returns the best (a,b,c) parameters.
    """
    n_obs = len(obs)
    n_mod = len(mod)

    best_score = -1
    best_params = (0.0, 1.0, 0.0)

    # Iterate over the grid
    for a in a_grid:
        for b in b_grid:
            for c in c_grid:
                # 2. Inlier Counting
                score = 0
                ptr = 0

                for m_idx in range(n_mod):
                    pred = a * mod[m_idx] ** 2 + b * mod[m_idx] + c
                    while ptr < n_obs and obs[ptr] < pred - tol: ptr += 1
                    if ptr < n_obs and abs(obs[ptr] - pred) <= tol: score += 1

                if score > best_score:
                    best_score = score
                    best_params = (a, b, c)

                    # Early exit:
                    if best_score == n_mod:
                        return best_params
    return best_params
