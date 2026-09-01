"""Linear calibration recipe for the PLANETES pipeline."""

from .. import log
from ..config import PipelineConfig
from ..products import P2VMProduct
from .flat import run_flat
from .p2vm import run_p2vm
from .preproc import run_preproc_p2vm
from .wavelength import run_wave


def run_calibration(config: PipelineConfig) -> P2VMProduct:
    """Run flat, wavelength, extraction, and P2VM calibration in order."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.save(config.output_dir / "config_used.yaml")
    log.info("Starting full calibration sequence")
    run_flat(config)
    run_wave(config)
    run_preproc_p2vm(config)
    product = run_p2vm(config)
    log.info("Calibration sequence completed successfully")
    return product
