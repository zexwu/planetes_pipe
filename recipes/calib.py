from . import PipelineContext, arg, command, log
from .flat import run_flat
from .wave import run_wave
from .preproc import run_preproc
from .p2vm import run_p2vm


@command("calib", "Run the full calibration sequence: flat, wave, preproc, p2vm", 
         requires=[],
         produces=["flat", "wave", "preproc", "p2vm"])
def run_calib(ctx: PipelineContext, **kwargs):
    CALIB_STEPS = [run_flat, run_wave, run_preproc, run_p2vm]
    for step in CALIB_STEPS:
        step(ctx, **kwargs)
    return 
