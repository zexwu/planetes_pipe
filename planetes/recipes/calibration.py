"""Linear calibration recipe for the PLANETES pipeline."""

from .. import log
from ..config import PipelineConfig
from ..product import P2VMProduct
from .flat import run_flat
from .p2vm import run_p2vm
from .preproc import preprocess_calibration
from .wavelength import run_wave


def run_calibration(config: PipelineConfig) -> P2VMProduct:
    """Run flat, wavelength, extraction, and P2VM calibration in order."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.copy_used_config(config.output_dir / "config_used.yaml")
    log.info("Starting full calibration sequence")
    flat = run_flat(config)
    wave = run_wave(config, flat)
    spectra = preprocess_calibration(config, flat, wave)
    product = run_p2vm(config, spectra)
    log.info("Calibration sequence completed successfully")
    return product
