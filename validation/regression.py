"""Reproduce the complete bench chain in fresh output directories; no raw writes.

Set DATA and OUTPUT in main(), or import run_chain from a research script.
The original v0.2 source is run with run_reference before comparing results.
"""

from pathlib import Path
import json
import os
import subprocess
import sys
from time import perf_counter


def run_recipe(
    stage,
    roles,
    output,
    *,
    source,
    python=sys.executable,
    environment=None,
    parameters=(),
):
    """Execute a real PyEsoRex recipe with explicit roles and a short local log name."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    sof = output / "inputs.sof"
    sof.write_text(
        "".join(f"{Path(path).absolute()} {role}\n" for role, path in roles.items())
    )
    command = [
        str(python),
        "-m",
        "pyesorex",
        f'--recipe-dir={Path(source)/"planetes/eso"}',
        f"planetes_{stage}",
        *parameters,
        str(sof),
    ]
    before = perf_counter()
    with (output / "execution.log").open("w") as log:
        subprocess.run(
            command,
            cwd=output,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    return {"stage": stage, "seconds": perf_counter() - before, "command": command}


def run_chain(
    data: Path, output: Path, *, source=None, python=sys.executable, environment=None
):
    """Run the five scientific recipe types, including three independently selected darks."""
    data, output = Path(data), Path(output)
    source = Path(__file__).resolve().parents[1] if source is None else Path(source)
    raw, science = data / "p2vm2", data / "sci2"
    steps = []
    for name, path in [
        ("main", raw / "dark.fits"),
        ("wave", raw / "wave_dark.fits"),
        ("science", science / "sci_dark.fits"),
    ]:
        steps.append(
            run_recipe(
                "dark",
                {"DARK_RAW": path},
                output / f"dark_{name}",
                source=source,
                python=python,
                environment=environment,
            )
        )
    dark = {
        name: output / f"dark_{name}/planetes_dark.fits"
        for name in ("main", "wave", "science")
    }
    flats = {
        f"TEL_FLAT_{name}": raw / f"tel{name}.fits" for name in ("4", "3", "2", "1")
    }
    flat, wave, p2vm = (
        output / f"{name}/planetes_{name}.fits" for name in ("flat", "wave", "p2vm")
    )
    jobs = [
        ("flat", {"DARK_CALIB": dark["main"], **flats}, "flat"),
        (
            "wave",
            {
                "DARK_CALIB": dark["wave"],
                "PLANETES_FLAT": flat,
                "WAVE_RAW": raw / "wave.fits",
            },
            "wave",
        ),
        (
            "p2vm",
            {
                "DARK_CALIB": dark["main"],
                "PLANETES_FLAT": flat,
                "PLANETES_WAVE": wave,
                **flats,
                **{
                    f"P2VM_RAW_{name}": raw / f"bsl{name}.fits"
                    for name in ("43", "42", "41", "32", "31", "21")
                },
                "PHASE_REFERENCE": science / "sci.fits",
                "PHASE_DARK_CALIB": dark["science"],
            },
            "p2vm",
        ),
        (
            "reduce",
            {
                "DARK_CALIB": dark["science"],
                "SCIENCE_RAW": science / "sci.fits",
                "PLANETES_FLAT": flat,
                "PLANETES_WAVE": wave,
                "PLANETES_P2VM": p2vm,
            },
            "science",
        ),
    ]
    for stage, roles, folder in jobs:
        steps.append(
            run_recipe(
                stage,
                roles,
                output / folder,
                source=source,
                python=python,
                environment=environment,
            )
        )
        print(stage, steps[-1]["seconds"], flush=True)
    (output / "execution.json").write_text(json.dumps(steps, indent=2) + "\n")
    return steps


def run_reference(
    data: Path, output: Path, source: Path, *, python=sys.executable, cache=None
):
    """Run frozen v0.2 under its own import path and export arrays with its decoder."""
    data, output, source = Path(data), Path(output), Path(source)
    raw, science = data / "p2vm2", data / "sci2"
    environment = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(source), str(Path(__file__).parent)]),
    }
    if cache is not None:
        environment["NUMBA_CACHE_DIR"] = str(cache)
    flats = {f"TEL_FLAT_{n}": raw / f"tel{n}.fits" for n in ("4", "3", "2", "1")}
    flat, wave, p2vm = (
        output / f"{name}/planetes_{name}.fits" for name in ("flat", "wave", "p2vm")
    )
    jobs = [
        ("flat", {"DARK_RAW": raw / "dark.fits", **flats}, "flat"),
        (
            "wave",
            {
                "WAVE_RAW": raw / "wave.fits",
                "WAVE_DARK": raw / "wave_dark.fits",
                "PLANETES_FLAT": flat,
            },
            "wave",
        ),
        (
            "p2vm",
            {
                "PLANETES_FLAT": flat,
                "PLANETES_WAVE": wave,
                **flats,
                **{
                    f"P2VM_RAW_{n}": raw / f"bsl{n}.fits"
                    for n in ("43", "42", "41", "32", "31", "21")
                },
                "PHASE_REFERENCE": science / "sci.fits",
                "PHASE_DARK": science / "sci_dark.fits",
            },
            "p2vm",
        ),
        (
            "reduce",
            {
                "SCIENCE_RAW": science / "sci.fits",
                "SCIENCE_DARK": science / "sci_dark.fits",
                "PLANETES_FLAT": flat,
                "PLANETES_WAVE": wave,
                "PLANETES_P2VM": p2vm,
            },
            "science",
        ),
    ]
    steps = [
        run_recipe(
            stage,
            roles,
            output / folder,
            source=source,
            python=python,
            environment=environment,
        )
        for stage, roles, folder in jobs
    ]
    subprocess.run(
        [
            str(python),
            "-c",
            "from pathlib import Path; from reference_export import export; import sys; export(Path(sys.argv[1]))",
            str(output),
        ],
        cwd=output,
        env=environment,
        check=True,
    )
    (output / "execution.json").write_text(json.dumps(steps, indent=2) + "\n")
    return steps


def run_python_chain(data: Path, output: Path):
    """Run the same complete tasks directly from Python and explicitly save each result."""
    from planetes.dark import calibrate_dark_file
    from planetes.flat import calibrate_flat_file
    from planetes.wavelength import calibrate_wavelength_file
    from planetes.p2vm import calibrate_p2vm_file
    from planetes.reduction import reduce_file
    from planetes.fits import write_product

    data, output = Path(data), Path(output)
    raw, science = data / "p2vm2", data / "sci2"
    darks = {
        name: calibrate_dark_file(path)
        for name, path in [
            ("main", raw / "dark.fits"),
            ("wave", raw / "wave_dark.fits"),
            ("science", science / "sci_dark.fits"),
        ]
    }
    for name, product in darks.items():
        write_product(output / f"dark_{name}/planetes_dark.fits", product)
    flats = {name: raw / f"tel{name}.fits" for name in ("4", "3", "2", "1")}
    flat = calibrate_flat_file(flats, darks["main"])
    write_product(output / "flat/planetes_flat.fits", flat)
    wave = calibrate_wavelength_file(raw / "wave.fits", darks["wave"], flat)
    write_product(output / "wave/planetes_wave.fits", wave)
    p2vm = calibrate_p2vm_file(
        flat,
        wave,
        darks["main"],
        telescope_flats=flats,
        baseline_cubes={
            name: raw / f"bsl{name}.fits"
            for name in ("43", "42", "41", "32", "31", "21")
        },
        phase_reference=science / "sci.fits",
        phase_dark=darks["science"],
    )
    write_product(output / "p2vm/planetes_p2vm.fits", p2vm)
    reduced = reduce_file(science / "sci.fits", darks["science"], flat, wave, p2vm)
    write_product(output / "science/planetes_reduced.fits", reduced)
    return reduced


def main():
    """Explicit local settings; choose a new OUTPUT for every numerical comparison."""
    project = Path(__file__).resolve().parents[1]
    data = project / "2026_02_05"
    output = project / "reduced/regression_new_run"
    environment = {**os.environ, "NUMBA_CACHE_DIR": str(output / "numba_cache")}
    run_chain(data, output, environment=environment)


if __name__ == "__main__":
    main()
