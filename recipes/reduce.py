"""
Science data reduction module for the PLANETES pipeline.

This module handles the reduction of science target data, including
visibility extraction, group delay computation, and phase calculation.
"""

from collections.abc import Mapping
from typing import Any, Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray

from . import PipelineContext, arg, command, log
from .products import P2VM_PRODUCT, preproc_product, reduced_product
from .preproc import run_preproc
from .visualize import colored_text, summary_plot


@command(
    "reduce",
    "Reduce science target data.",
    requires=["flat", "wave", "p2vm"],
    produces=["preproc", "reduced"]
)
@arg("--object", type=str, required=True, help="Target to reduce (object name)")
def run_reduce(ctx: PipelineContext, **kwargs: Any) -> None:
    """
    Reduce science target data.

    Args:
        ctx: Pipeline context with configuration
        **kwargs: Additional keyword arguments passed from CLI
    """
    obj = kwargs.get("object", None)
    if not obj:
        raise ValueError("An object name is required for reduction")
    if obj != "all" and obj not in ctx.conf["object"]:
        available = ", ".join(ctx.conf["object"])
        raise ValueError(f"Unknown object {obj!r}; available objects: {available}")

    log.info("--- Step: REDUCE ---")

    # --- 1. Load P2VM ---
    with ctx.load_product(P2VM_PRODUCT) as d:
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
        spec = ctx.load_product(preproc_product(object_name))["spec"]
        cobj = colored_text(object_name, color="green", bold=True)

        log.info(f"Applying P2VM to extract visibility for {cobj}...")
        p2vmred = np.einsum("box,ofx->fbx", p2vm_map, spec)

        fluxdata = p2vmred[:, :ctx.n_tel, :]
        # Extract visibility data
        coherent_flux = (
            p2vmred[:, ctx.sl_real, :] + 1j * p2vmred[:, ctx.sl_imag, :]
        )
        visamp, flag = normalize_visibility(coherent_flux, fluxdata, bsl_to_tel)

        # Compute group delay for each baseline
        gdelay = np.zeros((len(coherent_flux), ctx.n_bsl))
        log.info(f"Computing group delay for {cobj}...")
        for bsl in range(ctx.n_bsl):
            gdelay[:, bsl] = compute_gdelay(
                coherent_flux[:, bsl], wl_grid, search_range=gd_range
            )

        # Compute visibility phase
        visphi = coherent_flux * np.exp(
            -2j * np.pi * gdelay[:, :, None] / wl_grid[None, :]
        )
        visphi *= visphi.mean(axis=2)[:, :, None].conj()
        visphi = np.angle(visphi, deg=True)

        # Normalize phase
        visphi -= visphi.mean(axis=2)[:, :, None]
        visphi = (visphi + 180) % 360 - 180

        # Save results
        product_data: Dict[str, Any] = {
            "fluxdata": fluxdata,
            "coherent_flux": coherent_flux,
            "visamp": visamp,
            "flag": flag,
            "p2vmred": p2vmred,
            "wl_grid": wl_grid,
            "gdelay": gdelay,
            "visphi": visphi,
            "bsl_to_reg": bsl_to_reg,
            "bsl_to_tel": bsl_to_tel,
        }

        ctx.save_product(reduced_product(object_name), **product_data)
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


def normalize_visibility(
    coherent_flux: NDArray,
    flux: NDArray,
    baseline_telescopes: Mapping[int, tuple[int, int]],
) -> tuple[NDArray, NDArray]:
    """Normalize coherent flux and flag non-positive or non-finite denominators."""
    visamp = np.full(coherent_flux.shape, np.nan, dtype=float)
    flag = np.ones(coherent_flux.shape, dtype=bool)

    for baseline, (telescope_1, telescope_2) in baseline_telescopes.items():
        flux_product = flux[:, telescope_1, :] * flux[:, telescope_2, :]
        valid = (
            (flux_product > 0)
            & np.isfinite(flux_product)
            & np.isfinite(coherent_flux[:, baseline, :])
        )
        visamp[:, baseline, :][valid] = (
            np.abs(coherent_flux[:, baseline, :][valid])
            / np.sqrt(flux_product[valid])
        )
        flag[:, baseline, :] = ~valid

    return visamp, flag


def compute_gdelay(
    visdata: NDArray,
    wl: NDArray,
    search_range: Tuple[float, float] = (-30, 30),
    n_newton: int = 5
) -> NDArray:
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
