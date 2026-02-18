import numpy as np

from . import PipelineContext, arg, command, log
from .preproc import run_preproc
from .visualize import summary_plot, colored_text


@command("reduce", "Reduce science target data.", 
         requires=["flat", "wave", "p2vm"], 
         produces=["preproc", "reduced"])
@arg("--object", type=str, default=None, help="target to reduce (object name)")
def run_reduce(ctx: PipelineContext, **kwargs):
    obj = kwargs.get("object", None)
    if not obj: return # fmt: skip

    # --- 1. Load P2VM ---
    with ctx.load_product("p2vm") as d:
        p2vm_map = d["p2vm"]
        wl_grid = d["wl_grid"]
        bsl_to_reg = d["bsl_to_reg"][()]
        bsl_to_tel = d["bsl_to_tel"][()]

    def reduce_single(object=None, gd_range=(-100, 100)):

        spec = ctx.load_product(("preproc", object))["spec"]
        cobj = colored_text(object, color="green", bold=True)

        log.info(f"Applying P2VM to extract Visibility for {cobj}...")
        p2vmred = np.einsum("box,ofx->fbx", p2vm_map, spec)

        # (n_frame, 16, n_wave)
        # (n_frame, 4, n_wave) -> flux
        # 4:10 -> real
        # 10:16 -> imag
        visdata = p2vmred[:, ctx.sl_real, :] + 1j * p2vmred[:, ctx.sl_imag, :]

        gdelay = np.zeros((len(visdata), ctx.n_bsl))
        log.info(f"Computing Group Delay for {cobj}...")
        for bsl in range(ctx.n_bsl):
            gdelay[:, bsl] = compute_gdelay(visdata[:, bsl], wl_grid, 
                                            search_range=gd_range)

        visphi = visdata * np.exp(-2j * np.pi * gdelay[:, :, None] / wl_grid[None, :])
        visphi *= visphi.mean(axis=2)[:, :, None].conj()
        visphi = np.angle(visphi, deg=True)

        visphi -= visphi.mean(axis=2)[:, :, None]
        visphi = (visphi + 180) % 360 - 180

        to_save = ["visdata", "p2vmred", "wl_grid", "gdelay", "visphi"]
        product = {k: locals()[k] for k in to_save} \
               | dict(bsl_to_reg=bsl_to_reg, bsl_to_tel=bsl_to_tel)

        ctx.save_product(("reduced", object), **product)
        log.info(f"Reduction for {cobj} completed.")
        return

    log.info("--- Step: REDUCE ---")
    obj_list = ctx.conf["object"] if obj == "all" else [obj]
    for obj in obj_list:
        run_preproc(ctx, object=obj)
        reduce_single(object=obj, gd_range=(-100, 100))
        if ctx.conf.get("plot_to_pdf", False):
            summary_plot(ctx, object=obj)
    log.info("--- Step: REDUCE [DONE] ---")
    log.info("")


def compute_gdelay(visdata, wl, search_range=(-30, 30), n_newton=5):
    """Computes Group Delay for a batch of visibilities."""

    # Constants
    k = -2j * np.pi / wl
    w = -2 * np.pi / wl

    gd_grid = np.arange(search_range[0], search_range[1], 0.01)
    phasors_grid = np.exp(gd_grid[:, None] * k[None, :])
    coherence = np.abs(np.tensordot(visdata, phasors_grid, axes=([-1], [-1])))

    idx_best = np.argmax(coherence, axis=-1)
    gd_current = gd_grid[idx_best]  # Shape: (N_frames,)

    # maximize f(gd) = |sum(V * exp(i*w*gd))|^2
    for _ in range(n_newton):
        gd_exp = gd_current[:, None]
        rot_term = np.exp(1j * w * gd_exp)
        V_rot = visdata * rot_term
        S0 = np.sum(V_rot, axis=-1)
        S1 = np.sum(V_rot * (1j * w), axis=-1)
        S2 = np.sum(V_rot * (-(w**2)), axis=-1)
        S0_conj = np.conj(S0)
        grad = 2 * np.real(S1 * S0_conj)
        hess = 2 * np.real(S2 * S0_conj + S1 * np.conj(S1))
        diff = np.zeros_like(grad)
        mask = hess != 0

        diff[mask] = grad[mask] / hess[mask]
        gd_current = gd_current - diff
    return gd_current
