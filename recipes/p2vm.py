import numpy as np

from typing import Any, Dict, List, Optional, Tuple
from scipy.optimize import minimize

from . import PipelineContext, command, arg, log
from .reduce import compute_gdelay
from .visualize import plt, genfig


@command("p2vm", "Calculate P2VM matrix.", 
         requires=["flat", "wave", ("preproc", "p2vm")],
         produces=["p2vm"])
@arg("--phase_correction", type=int, default=1, help="Apply phase correction using WAVE,SC data.")
def run_p2vm(ctx: PipelineContext, **kwargs: Any) -> None:
    """
    Computes the Pixel-to-Visibility Matrix.
    """
    log.info("--- Step: P2VM ---")

    # Load Data
    with ctx.load_product(("preproc", "p2vm")) as d:
        spec_tel = d["spec_tel"]
        spec_bsl = d["spec_bsl"]
        spec_wavesc = d["spec_wavesc"]
        bsl_to_reg = d["bsl_to_reg"][()]
        bsl_to_tel = d["bsl_to_tel"][()]
        spec_flat = d["spec_flat"]
        wl_grid = d["wl_grid"]

    reg_to_bsl = {reg: bsl for bsl, regs in bsl_to_reg.items() for reg in regs}

    n_frame = spec_bsl[0].shape[1]
    n_wave = spec_bsl[0].shape[2]

    # Initialize matrices
    # v2pm: Visibility to Pixel Matrix
    v2pm = np.zeros((ctx.n_reg, ctx.n_data, n_wave)) 
    cmat = np.zeros((ctx.n_reg, ctx.n_bsl, n_wave))
    ellipse_results = []

    if ctx.conf.get("plot_to_pdf", False):
        pdf = ctx.plot_ctx("p2vm.pdf")

    # --- A. Process Baselines (Interference Terms) ---
    for bsl in range(ctx.n_bsl):
        log.info(f"Processing baseline {bsl+1}/{ctx.n_bsl}...")

        spec = spec_bsl[bsl]
        regs = bsl_to_reg[bsl]
        envs = np.ones((n_frame, n_wave), dtype=float)

        # 1. Fit Ellipses to extract raw Phase & Visibility
        log.debug("    Fitting ellipses for phase extraction...")
        phase, visamp = fit_ellipse_linear(spec[regs], envs)
        _visdata = visamp * np.exp(1j * phase)
        ellipse_results.append((phase, visamp))

        # use the central 50% of the wavelength range for OPD and GD estimation
        w_start = n_wave // 4
        w_end = (n_wave * 3) // 4

        opd = np.mean((phase * wl_grid)[:, w_start:w_end] / 2 / np.pi, axis=1)
        gd = compute_gdelay(_visdata[:, w_start:w_end], wl_grid[w_start:w_end])

        # Assume OPD=0 <-> GD=0; remove OPD zero-point
        slope, opd0 = np.polyfit(gd, opd, 1)
        opd -= opd0
        log.debug(f"    Removing zero-point {opd0=:.2f} um...")

        if ctx.conf.get("plot_to_pdf", False):
            fig, axs = genfig(2, 3, 
                              xlabel="C-A (transformed)", ylabel="D-B (transformed)")
            for i, ax in enumerate(axs):
                iw = n_wave // 6 * i
                visdata = visamp[:, iw] * np.exp(1j * phase[:, iw])
                ax.scatter(visdata.real, visdata.imag, s=5, alpha=0.5)
                circ = plt.Circle((0, 0), 1, color='r', fill=False)
                ax.add_artist(circ)
                ax.set_xlim(-1.2, 1.2)
                ax.set_ylim(-1.2, 1.2)
                ax.text(0.5, 0.5, f"Wavelength Index {iw}", 
                        transform=ax.transAxes, va="center", ha="center")
            fig.suptitle(f"Baseline {ctx.baselines[bsl]} Ellipse Fits")
            pdf.savefig(fig); plt.close(fig)

            fig, axs = genfig(2, 3, 
                              xlabel="C-A (transformed)", ylabel="D-B (transformed)")
            for i, ax in enumerate(axs):
                id = n_frame // 6 * i
                c = np.linspace(0, 1, n_wave)
                visdata = visamp[id, :] * np.exp(1j * phase[id, :])
                ax.scatter(visdata.real, visdata.imag, 
                           s=5, c=c, cmap="rainbow", alpha=0.5)
                circ = plt.Circle((0, 0), 1, color='k', fill=False)
                ax.add_artist(circ)
                ax.set_xlim(-1.2, 1.2)
                ax.set_ylim(-1.2, 1.2)
                ax.text(0.5, 0.5, f"#Frame {id}", 
                        transform=ax.transAxes, va="center", ha="center")
            fig.suptitle(f"Baseline {ctx.baselines[bsl]} Ellipse Fits")
            pdf.savefig(fig); plt.close(fig)

        # 2. Fit P2VM Coefficients
        # Reconstruct phase for fitting
        phi = 2 * np.pi * opd[:, None] / wl_grid[None, :]

        # Solve Linear System: spec = a*cos(phi) + b*cos(phi) + c
        mat = np.ones((len(phi), 3))
        log.debug("    Fitting P2VM coefficients...")
        for iw in range(n_wave):
            # Construct design matrix for this wavelength
            mat[:, 1] = np.sin(phi[:, iw])
            mat[:, 2] = np.cos(phi[:, iw])
            mat_inv = np.linalg.pinv(mat)

            # Fit coefficients
            coeff = np.einsum("if,of->oi", mat_inv, spec[..., iw])
            a, b, c = coeff[:, 2], coeff[:, 1], coeff[:, 0]

            # Store in V2PM
            v2pm[:, ctx.sl_real, iw][:, bsl] = np.sqrt(a**2 + b**2) # Coherence Amp
            v2pm[:, ctx.sl_imag, iw][:, bsl] = np.arctan2(b, a) # Phase
            cmat[:, bsl, iw] = c

            if ctx.conf.get("plot_to_pdf", False):
                if iw % (n_wave // 6): continue
                fig, axs = genfig(2, 4, xlabel="#Frame", ylabel="Flux [adu]", 
                                  sharey=False, sharex=False)
                regs = bsl_to_reg[bsl]
                labels = ["A", "C", "B", "D"]
                for i in range(4):
                    axs[i].plot(spec[regs[i], :, iw] / spec_flat[regs[i], :, iw])
                    axs[i].plot(mat @ coeff[regs[i]] / spec_flat[regs[i], :, iw], ls="--")
                    axs[i].set_title(f"OUTPUT {regs[i]}-{labels[i]}")
                    axs[i].sharey(axs[0])

                for i in range(4):
                    res = spec[regs[i], :, iw] / spec_flat[regs[i], :, iw] - \
                          mat @ coeff[regs[i]] / spec_flat[regs[i], :, iw]
                    axs[4+i].plot(res, "o-", markersize=3, alpha=0.5, c="gray")
                    axs[4+i].set_ylim(-1, 1)
                axs[4].set_yticks([-1, 0, 1])
                axs[4].set_ylabel("Residual [adu]")
                fig.suptitle(f"Baseline {ctx.baselines[bsl]} Wavelength Index {iw}; P2VM Fit")
                pdf.savefig(fig); plt.close(fig)

                if False:
                    # --- refining the wavelength ---
                    fig, axs = genfig(2, 2, xlabel="#Frame", ylabel="Flux [adu]")
                    regs = bsl_to_reg[bsl]
                    labels = ["A", "C", "B", "D"]
                    for i in range(4):

                        def model(a, b, c, f):
                            return c + a * np.cos(phi[:, iw] * f) + b * np.sin(phi[:, iw] * f)
                        def chi2(theta):
                            a, b, c, f = theta
                            mod = model(a, b, c, f)
                            return np.sum((spec[regs[i], :, iw] - mod) ** 2)
                        res = minimize(chi2, x0=[a[i], b[i], c[i], 1.0], method="Powell", 
                                    bounds=[(None, None), (None, None), (None, None), (0.7, 1.3)])
                        _f = res.x[3]

                        _mat = np.ones((len(phi), 3))
                        _mat[:, 1] = np.sin(phi[:, iw] * _f)
                        _mat[:, 2] = np.cos(phi[:, iw] * _f)
                        _coeff = np.einsum("if,of->oi", np.linalg.pinv(_mat), spec[..., iw])

                        axs[i].plot(spec[regs[i], :, iw] / spec_flat[regs[i], :, iw])
                        axs[i].plot(_mat @ _coeff[regs[i]] / spec_flat[regs[i], :, iw], ls="--")
                        axs[i].set_title(f"OUTPUT {regs[i]}-{labels[i]}; f={_f:.3f}")

                    fig.suptitle(f"Baseline {ctx.baselines[bsl]} Wavelength Index {iw}; P2VM Fit")
                    pdf.savefig(fig); plt.close(fig)

        if ctx.conf.get("plot_to_pdf", False):
            # Also plot the raw spectra for this baseline (normalized by flat)
            _spec = spec / spec_flat
            fig, axs = genfig(2, 2, 
                              xlabel="Wavelength Index", ylabel="Normalized Flux")
            regs = bsl_to_reg[bsl]
            colors = plt.cm.viridis(np.linspace(0, 1, n_frame))
            for i in range(4):
                axs[i].set_prop_cycle(color=colors)
                axs[i].plot(_spec[regs[i], :, :].T, alpha=0.3, lw=0.3)
                axs[i].set_title(f"OUTPUT {regs[i]}-{['A','C','B','D'][i]}")
            fig.suptitle(f"Baseline {ctx.baselines[bsl]} Extracted Spectra")
            pdf.savefig(fig); plt.close(fig)

    # --- B. Process Telescopes (Photometric Terms) ---
    for tel in range(ctx.n_tel):
        log.info(f"Processing telescope {tel + 1}/{ctx.n_tel}...")
        # Median transmission map
        trans_map = np.median(spec_tel[tel], axis=1)
        # Normalize sum to 1
        v2pm[:, tel, :] = trans_map / trans_map.sum()

    # --- C. Normalization & Inversion ---
    log.info("P2VM Normalization...")
    log.info("    Phase normalization (A=0)...")
    for bsl in range(ctx.n_bsl):
        regs = bsl_to_reg[bsl]
        phase_A = v2pm[regs[0], ctx.sl_imag, :]
        v2pm[regs, ctx.sl_imag, :] -= phase_A[None, :, :]

    log.info("    Transmission normalization (sum of 24 outputs -> 1)...")
    v2pm[:, :ctx.n_tel, :] /= v2pm[:, :ctx.n_tel, :].mean(axis=0)[None, :, :]

    log.info("    Coherence normalization...")
    for bsl in range(ctx.n_bsl):
        t1, t2 = bsl_to_tel[bsl]
        for iw in range(n_wave):
            t_mat = v2pm[:, :ctx.n_tel, iw]
            c = cmat[:, bsl, iw]

            # Solve for intensities contributing to the DC term
            imat = np.linalg.pinv(t_mat) @ c
            f1f2 = np.sqrt(imat[t1] * imat[t2])
            v2pm[:, ctx.n_tel + bsl, iw] /= f1f2

    # Convert Polar (Amp, Phase) to Cartesian (Cos, Sin) for final matrix
    coh = v2pm[:, ctx.sl_real, :]
    phase = v2pm[:, ctx.sl_imag, :]

    def phase_corr(phase: np.ndarray, visdata: np.ndarray) -> np.ndarray:
        phase = phase.copy()
        opl = np.zeros((len(visdata), 4))
        wave0 = len(wl_grid) // 2

        for base in [1, 2, 3]:
            _gd = compute_gdelay(visdata[:, base - 1, :], wl_grid)
            _phasor = np.exp(-1j * 2 * np.pi * _gd[:, None] / wl_grid)
            opl[:, base] = np.angle(visdata[:, base - 1, wave0] * _phasor[:, wave0]) +\
                           2 * np.pi * _gd / wl_grid[wave0]
        opl -= opl.mean(axis=0)[None, :]

        visphase = np.zeros((ctx.n_bsl, n_wave), dtype=complex)
        for bsl in range(ctx.n_bsl):
            tel1, tel2 = bsl_to_tel[bsl]
            x = opl[:, tel2] - opl[:, tel1]
            visphase[bsl, :] = (
                visdata[:, bsl, :] * np.exp(-1j * x[:, None] * wl_grid[wave0] / wl_grid)
            ).sum(axis=0)

        ref = np.ones((4, n_wave), dtype=complex)
        gdelay = compute_gdelay(visphase, wl_grid)

        for base in [0, 1, 2]:
            ref[base + 1] = np.exp(2j * np.pi * gdelay[base] / wl_grid)

        for bsl in range(ctx.n_bsl):
            tel1, tel2 = bsl_to_tel[bsl]
            visphase[bsl, :] *= ref[tel1] * ref[tel2].conj()

        for reg in range(ctx.n_reg):
            bsl = reg_to_bsl[reg]
            phi = np.exp(1j * phase[reg, bsl, :]) * visphase[bsl, :].conj()
            phase[reg, bsl, :] = np.angle(phi)
        return phase

    _v2pm = v2pm.copy()
    _v2pm[:, ctx.sl_real, :] = coh * np.cos(phase)
    _v2pm[:, ctx.sl_imag, :] = coh * np.sin(phase)
    _v2pm /= ctx.n_reg

    log.info("V2PM -> P2VM") 
    p2vm = np.zeros((ctx.n_data, ctx.n_reg, n_wave))
    for iw in range(n_wave):
        p2vm[:, :, iw] = np.linalg.pinv(_v2pm[:, :, iw])


    if spec_wavesc is not None and kwargs["phase_correction"]:
        log.info("Applying phase correction...")
        p2vmred_wave = np.einsum("dow,ofw->fdw", p2vm, spec_wavesc)
        vis_ft = p2vmred_wave[:, ctx.sl_real, :] + 1j * p2vmred_wave[:, ctx.sl_imag, :]
        phase = phase_corr(phase, vis_ft)

        _v2pm = v2pm.copy()
        _v2pm[:, ctx.sl_real, :] = coh * np.cos(phase)
        _v2pm[:, ctx.sl_imag, :] = coh * np.sin(phase)
        _v2pm /= ctx.n_reg

        p2vm = np.zeros((ctx.n_data, ctx.n_reg, n_wave))
        for iw in range(n_wave):
            p2vm[:, :, iw] = np.linalg.pinv(_v2pm[:, :, iw])

    to_save = ["p2vm", "v2pm", "opd", "wl_grid", 
               "bsl_to_reg", "bsl_to_tel", "ellipse_results"]
    ctx.save_product("p2vm", **{k: locals()[k] for k in to_save})
    log.info("--- Step: P2VM [DONE] ---")
    log.info("")

    if ctx.conf.get("plot_to_pdf", False):
        fig, axs = genfig(2, 3, xlabel="Wavelength index", ylabel="Phase [deg]")
        phase = v2pm[:, ctx.sl_imag, :]
        for bsl in range(6):
            regs = bsl_to_reg[bsl]
            phase[regs, bsl, :] -= phase[regs[0], bsl, :]
        phase = np.unwrap(phase, period=2 * np.pi)
        phase *= 180 / np.pi
        for bsl in range(6):
            regs = bsl_to_reg[bsl]
            for reg, lb in zip(regs, ["A", "C", "B", "D"]):
                axs[bsl].plot(phase[reg, bsl, :], label=lb)
            axs[bsl].set_title(f"Baseline {ctx.baselines[bsl]}")
        fig.suptitle("V2PM Phase [deg]")
        pdf.savefig(fig); plt.close(fig)


        fig, axs = genfig(2, 3, xlabel="Wavelength index", ylabel="Coherence")
        for bsl in range(6):
            regs = bsl_to_reg[bsl]
            tels = bsl_to_tel[bsl]
            t1t2 = 2 * abs(v2pm[:, tels[0], :] * v2pm[:, tels[1], :]) ** 0.5
            for reg, lb in zip(regs, ["A", "C", "B", "D"]):
                axs[bsl].plot(coh[reg, bsl, :] / t1t2[reg], label=lb)
            axs[bsl].set_title(f"Baseline {ctx.baselines[bsl]}")
        fig.suptitle("V2PM Coherence")
        pdf.savefig(fig); plt.close(fig)
        pdf.close()
    return


# @njit(parallel=True, fastmath=True, cache=True)
def fit_ellipse_linear(
    spec: np.ndarray, envs: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts phase and visibility from ABCD beam combiner data by fitting ellipses.

    Args:
        spec (np.ndarray): ABCD flux (4, n_frames, n_wave).
        envs (np.ndarray): Envelope coherence factors (n_frames, n_wave).

    Returns:
        tuple:
            - phase (np.ndarray): Extracted phase in radians (n_frames, n_wave).
            - visamp (np.ndarray): Visibility amplitude (n_frames, n_wave).
    """
    _, n_frames, n_wave = spec.shape

    phase = np.zeros((n_frames, n_wave))
    visamp = np.zeros((n_frames, n_wave))

    for iwl in range(n_wave):
        A, C, B, D = spec[:, :, iwl]

        X, Y = C - A, D - B

        # For numerical stability, center and scale the data
        X = X - np.mean(X)
        Y = Y - np.mean(Y)
        std_x, std_y = np.std(X), np.std(Y)
        if std_x > 1e-9:
            X /= std_x
        if std_y > 1e-9:
            Y /= std_y

        # Design matrix for: c0*X^2 + c1*XY + c2*X + c3*Y + c4*Y^2 = 1
        coeffs = np.empty((5, n_frames))
        coeffs[0, :] = X ** 2
        coeffs[1, :] = X * Y
        coeffs[2, :] = X
        coeffs[3, :] = Y
        coeffs[4, :] = Y ** 2

        # Linear solve via Pseudo-Inverse
        w = np.linalg.pinv(coeffs.T) @ np.ascontiguousarray(envs[:, iwl])

        b_sq = w[4]
        if b_sq <= 0:
            b_sq = 1e-6

        d = w[1] / (2 * b_sq)
        a_sq = w[0] - b_sq * (d**2)
        if a_sq <= 0:
            a_sq = 1e-6

        c2 = w[3] / (2 * b_sq)
        c1 = (w[2] - 2 * b_sq * d * c2) / (2 * a_sq)

        a, b = np.sqrt(a_sq), np.sqrt(b_sq)
        fac = np.sqrt(1 + (a * c1) ** 2 + (b * c2) ** 2)

        # Transform to unit circle
        X_fit = a * (X + c1) / fac
        Y_fit = b * (Y + d * X + c2) / fac

        phase[:, iwl] = np.unwrap(np.arctan2(Y_fit, X_fit))
        visamp[:, iwl] = np.sqrt(X_fit**2 + Y_fit**2)

    return phase, visamp
