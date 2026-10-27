"""PyCPL P2VM recipe with explicit file tags and standard recipe parameters."""

from typing import Any

from cpl import ui

from planetes import __version__

from planetes.algorithms.p2vm import compute_p2vm
from planetes.instrument import BASELINES, TELESCOPES
from planetes.io.products import read_product
from planetes.eso.common import new_output, save_product, select_inputs
from planetes.product import FlatProduct, WaveProduct
from planetes.processing import extract_calibration_spectra


class PlanetesP2VM(ui.PyRecipe):
    """Construct a P2VM from tagged raw cubes and upstream calibration FITS."""

    _name = "planetes_p2vm"
    _version = __version__
    _author = "PLANETES contributors"
    _email = ""
    _copyright = "PLANETES contributors; ESO interface under development"
    _synopsis = "Build the PLANETES pixel-to-visibility calibration"
    _description = (
        "4T/6B/24-output P2VM calibration. Required tags: PLANETES_FLAT, "
        "PLANETES_WAVE, TEL_FLAT_4/3/2/1, P2VM_RAW_43/42/41/32/31/21. "
        "Optional tags: PHASE_REFERENCE and PHASE_DARK. Every tag accepts one file. "
        "The instrument mapping, full-cube extraction, wavelength grid, and numerical "
        "defaults are unchanged. Output: planetes_p2vm.fits. Product categories and "
        "the payload schema are provisional pending ESO review."
    )

    def __init__(self) -> None:
        super().__init__()
        parameter = ui.ParameterValue(
            name="planetes.p2vm.phase_correction",
            context="planetes.p2vm",
            description="Apply the existing phase-reference correction when a reference is provided",
            default=True,
        )
        parameter.cli_alias = "phase-correction"
        self.parameters = ui.ParameterList([parameter])

    def run(self, frameset: ui.FrameSet, settings: dict[str, Any]) -> ui.FrameSet:
        """Validate tags, read inputs, calculate P2VM, and register one FITS product."""
        for parameter in self.parameters:
            parameter.value = parameter.default
        for name, value in settings.items():
            self.parameters[name].value = value

        required = ["PLANETES_FLAT", "PLANETES_WAVE"]
        required += [f"TEL_FLAT_{name}" for name in TELESCOPES]
        required += [f"P2VM_RAW_{name}" for name in BASELINES]
        selected, used = select_inputs(
            frameset, required, ["PHASE_REFERENCE", "PHASE_DARK"]
        )
        if "PHASE_DARK" in selected and "PHASE_REFERENCE" not in selected:
            raise ValueError("PHASE_DARK requires PHASE_REFERENCE")
        output = new_output("planetes_p2vm.fits")

        flat = read_product(selected["PLANETES_FLAT"], FlatProduct)
        wave = read_product(selected["PLANETES_WAVE"], WaveProduct)
        spectra = extract_calibration_spectra(
            flat,
            wave,
            telescope_flats={name: selected[f"TEL_FLAT_{name}"] for name in TELESCOPES},
            baseline_cubes={name: selected[f"P2VM_RAW_{name}"] for name in BASELINES},
            phase_reference=(
                None
                if "PHASE_REFERENCE" not in selected
                else selected["PHASE_REFERENCE"]
            ),
            phase_dark=(
                None if "PHASE_DARK" not in selected else selected["PHASE_DARK"]
            ),
        )
        enabled = self.parameters["planetes.p2vm.phase_correction"].value
        product = compute_p2vm(spectra, phase_correction=enabled)

        return save_product(
            output,
            product,
            used,
            self,
            phase_reference_applied=enabled and "PHASE_REFERENCE" in selected,
        )
