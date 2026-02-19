"""
Science data reduction module for the PLANETES pipeline.

This module handles the reduction of science target data, including
visibility extraction, group delay computation, and phase calculation.
"""

from typing import Any, Dict, List, Tuple

import numpy as np

from . import PipelineContext, arg, command, log
from .preproc import run_preproc
from .visualize import colored_text, summary_plot


@command(
    "reduce",
    "Reduce science target data.",
    requires=["flat", "wave", "p2vm"],
    produces=["preproc", "reduced"]
)
@arg("--object", type=str, default=None, help="Target to reduce (object name)")
def run_reduce(ctx: PipelineContext, **kwargs: Any) -> None:
    """
    Reduce science target data.

    Args:
        ctx: Pipeline context with configuration
        **kwargs: Additional keyword arguments passed from CLI
    """
    obj = kwargs.get("object", None)
    if not obj:
        log.error("No object specified for reduction")
        return

    log.info("--- Step: REDUCE ---")

    # --- 1. Load P2VM ---
    with ctx.load_product("p2vm") as d:
        p2vm_map = d["p2vm"]
        wl_grid = d["wl_grid"]
        bsl_to_reg = d["bsl_to_reg"][()]
        bsl_to_tel = d["bsl_to_tel"][()]

    def reduce_single(object_name: str, gd_range: Tuple[float, float] = (-100, 100)) -> None:
        """
        Reduce data for a single object.

        Args:
            object_name: Name of the object to reduce
            gd_range: Search range for group delay computation
        """
        # Load preprocessed spectra
        spec = ctx.load_product(("preproc", object_name))["spec"]
        cobj = colored_text(object_name, color="green", bold=True)

        log.info(f"Applying P2VM to extract visibility for {cobj}...")
        p2vmred = np.einsum("box,ofx->fbx", p2vm_map, spec)

        # Extract visibility data
        visdata = p2vmred[:, ctx.sl_real, :] + 1j * p2vmred[:, ctx.sl_imag, :]

        # Compute group delay for each baseline
        gdelay = np.zeros((len(visdata), ctx.n_bsl))
        log.info(f"Computing group delay for {cobj}...")
        for bsl in range(ctx.n_bsl):
            gdelay[:, bsl] = compute_gdelay(
                visdata[:, bsl], wl_grid, search_range=gd_range
            )

        # Compute visibility phase
        visphi = visdata * np.exp(-2j * np.pi * gdelay[:, :, None] / wl_grid[None, :])
        visphi *= visphi.mean(axis=2)[:, :, None].conj()
        visphi = np.angle(visphi, deg=True)

        # Normalize phase
        visphi -= visphi.mean(axis=2)[:, :, None]
        visphi = (visphi + 180) % 360 - 180

        # Save results
        product_data: Dict[str, Any] = {
            "visdata": visdata,
            "p2vmred": p2vmred,
            "wl_grid": wl_grid,
            "gdelay": gdelay,
            "visphi": visphi,
            "bsl_to_reg": bsl_to_reg,
            "bsl_to_tel": bsl_to_tel,
        }

        ctx.save_product(("reduced", object_name), **product_data)
        log.info(f"Reduction for {cobj} completed.")

    # Process objects
    obj_list: List[str] = ctx.conf["object"] if obj == "all" else [obj]

    for obj_name in obj_list:
        log.info(f"Processing object: {obj_name}")

        # Run preprocessing if needed
        run_preproc(ctx, object=obj_name)

        # Perform reduction
        reduce_single(obj_name, gd_range=(-100, 100))

        # Generate plots if requested
        if ctx.conf.get("plot_to_pdf", False):
            summary_plot(ctx, object=obj_name)

    log.info("--- Step: REDUCE [DONE] ---")
    log.info("")


def compute_gdelay(
    visdata: np.ndarray,
    wl: np.ndarray,
    search_range: Tuple[float, float] = (-30, 30),
    n_newton: int = 5
) -> np.ndarray:
    """
    Compute group delay for a batch of visibilities.

    Args:
        visdata: Visibility data array
        wl: Wavelength array
        search_range: Range for initial grid search
        n_newton: Number of Newton iterations for refinement

    Returns:
        Group delay values
    """
    # Constants
    k = -2j * np.pi / wl
    w = -2 * np.pi / wl

    # Initial grid search
    gd_grid = np.arange(search_range[0], search_range[1], 0.01)
    phasors_grid = np.exp(gd_grid[:, None] * k[None, :])
    coherence = np.abs(np.tensordot(visdata, phasors_grid, axes=([-1], [-1])))

    idx_best = np.argmax(coherence, axis=-1)
    gd_current = gd_grid[idx_best]  # Shape: (N_frames,)

    # Newton-Raphson refinement
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
