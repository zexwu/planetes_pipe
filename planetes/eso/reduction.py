"""PyCPL entry point for instrumental science visibility extraction."""

from typing import Any

from cpl import ui

from planetes import __version__
from planetes.algorithms.reduction import compute_reduce
from planetes.io.products import read_product
from planetes.processing import extract_target_spectra
from planetes.product import FlatProduct, P2VMProduct, WaveProduct
from planetes.eso.common import new_output, save_product, select_inputs


class PlanetesReduce(ui.PyRecipe):
    """Apply detector, wavelength, and P2VM calibration to one science cube."""

    _name = "planetes_reduce"
    _version = __version__
    _author = "PLANETES contributors"
    _email = ""
    _copyright = "PLANETES contributors; ESO interface under development"
    _synopsis = "Extract PLANETES instrumental visibility observables"
    _description = "Inputs: SCIENCE_RAW, PLANETES_FLAT, PLANETES_WAVE, PLANETES_P2VM; optional SCIENCE_DARK. Output: planetes_reduced.fits. This intermediate product has no on-sky transfer-function calibration or propagated science errors."

    def __init__(self) -> None:
        super().__init__()
        self.parameters = ui.ParameterList()

    def run(self, frameset: ui.FrameSet, settings: dict[str, Any]) -> ui.FrameSet:
        """Read explicit calibration products and register the instrumental result."""
        if settings:
            raise ValueError(f"Unknown reduction parameters: {', '.join(settings)}")
        inputs, used = select_inputs(
            frameset,
            ["SCIENCE_RAW", "PLANETES_FLAT", "PLANETES_WAVE", "PLANETES_P2VM"],
            ["SCIENCE_DARK"],
        )
        output = new_output("planetes_reduced.fits")
        flat = read_product(inputs["PLANETES_FLAT"], FlatProduct)
        wave = read_product(inputs["PLANETES_WAVE"], WaveProduct)
        p2vm = read_product(inputs["PLANETES_P2VM"], P2VMProduct)
        spectra = extract_target_spectra(
            inputs["SCIENCE_RAW"], flat, wave, dark=inputs.get("SCIENCE_DARK")
        )
        return save_product(output, compute_reduce(spectra, p2vm), used, self)
