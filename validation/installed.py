"""Verify an installed distribution from outside the checkout, including default wave reference."""

from pathlib import Path
import json
import os
import sys

import numpy as np
from astropy.table import Table
import planetes

from planetes.fits import read_product
from planetes.wavelength import load_wavelength_reference
from validation.regression import run_recipe


def validate_installation(
    products: Path, lamp: Path, output: Path, *, environment=None
):
    """Use exactly the input flat's matching wave as the numerical reference."""
    package = Path(planetes.__file__).parent
    lines = Table.read(package / "data/wavelength_lines.ecsv", format="ascii.ecsv")
    reference = load_wavelength_reference()
    assert (
        reference.wavelength_um.tobytes()
        == (np.array(lines["wavelength_nm"]) / 1000).tobytes()
    )
    assert reference.origin == tuple(lines["origin"])
    run_recipe(
        "wave",
        {
            "DARK_CALIB": products / "dark_wave/planetes_dark.fits",
            "PLANETES_FLAT": products / "flat/planetes_flat.fits",
            "WAVE_RAW": lamp,
        },
        output,
        source=package.parent,
        python=sys.executable,
        environment=environment,
    )
    actual = read_product(output / "planetes_wave.fits")
    expected = read_product(products / "wave/planetes_wave.fits")
    assert actual.provenance.dependencies == expected.provenance.dependencies
    assert actual.wavelength_map_um.tobytes() == expected.wavelength_map_um.tobytes()
    result = {
        "installed_module": str(planetes.__file__),
        "version": planetes.__version__,
        "reference_lines": len(reference.origin),
        "reference_values_and_origins_exact": True,
        "default_reference_wave_executed": True,
        "wave_bitwise_equal_to_matching_parent": True,
        "used_inputs": actual.provenance.inputs,
    }
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    # Set explicit real-data paths and a fresh output; run with installed package first on PYTHONPATH.
    project = Path("/Users/zexwu/zexwu_lib/planetes_pipe")
    work = project / "reduced/regression_new_run"
    assert not Path(planetes.__file__).is_relative_to(project / "planetes")
    print(
        validate_installation(
            work,
            project / "2026_02_05/p2vm2/wave.fits",
            work.parent / "installed_wave_new",
            environment={**os.environ, "NUMBA_CACHE_DIR": str(work / "numba_cache")},
        )
    )
