"""
Calibration sequence for the PLANETES pipeline.

This module orchestrates the full calibration sequence:
flat field, wavelength calibration, preprocessing, and P2VM calculation.
"""

from typing import Any

from . import PipelineContext, command, log
from .flat import run_flat
from .p2vm import run_p2vm
from .preproc import run_preproc
from .wave import run_wave


@command(
    "calib",
    "Run the full calibration sequence: flat, wave, preproc, p2vm",
    requires=[],
    produces=["flat", "wave", "preproc", "p2vm"]
)
def run_calib(ctx: PipelineContext, **kwargs: Any) -> None:
    """
    Execute the full calibration sequence.

    Args:
        ctx: Pipeline context with configuration
        **kwargs: Additional keyword arguments passed from CLI
    """
    log.info("Starting full calibration sequence")

    CALIB_STEPS = [run_flat, run_wave, run_preproc, run_p2vm]

    for i, step in enumerate(CALIB_STEPS, 1):
        step_name = step.__name__.replace("run_", "")
        log.info(f"Calibration step {i}/{len(CALIB_STEPS)}: {step_name}")
        step(ctx, **kwargs)

    log.info("Calibration sequence completed successfully")
