#!/usr/bin/env python
"""Explicit command-line entry point for the PLANETES pipeline."""

import argparse

from planetes import configure_logging
from planetes.calib import run_calibration
from planetes.config import PipelineConfig
from planetes.flat import run_flat
from planetes.p2vm import run_p2vm
from planetes.preproc import run_preproc_p2vm, run_preproc_target
from planetes.reduce import run_reduce
from planetes.visualize import summary_plot
from planetes.wave import run_wave


def build_parser() -> argparse.ArgumentParser:
    """Build the small, explicit pipeline CLI."""
    parser = argparse.ArgumentParser(
        description="PLANETES Bench Data Reduction Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--conf", default="conf.yaml", help="configuration YAML")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("calib", help="run the full calibration sequence")
    commands.add_parser("flat", help="build detector maps and traces")
    commands.add_parser("wave", help="build wavelength maps")
    commands.add_parser("p2vm", help="build the pixel-to-visibility matrix")

    preproc = commands.add_parser("preproc", help="extract calibration or target cubes")
    preproc.add_argument("target", nargs="?", default="p2vm", help="p2vm, all, or target name")

    reduce = commands.add_parser("reduce", help="reduce one target or all targets")
    reduce.add_argument("target", help="target name or all")

    plot = commands.add_parser("plot", help="plot one reduced target or all targets")
    plot.add_argument("target", help="target name or all")
    return parser


def main() -> None:
    """Load one configuration and dispatch one explicit recipe."""
    parser = build_parser()
    args = parser.parse_args()
    try:
        config = PipelineConfig.from_yaml(args.conf)
    except (FileNotFoundError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    configure_logging(config.log_level)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.save(config.output_dir / "config_used.yaml")

    match args.command:
        case "calib":
            run_calibration(config)
        case "flat":
            run_flat(config)
        case "wave":
            run_wave(config)
        case "preproc":
            if args.target == "p2vm":
                run_preproc_p2vm(config)
            else:
                for target in _targets(config, args.target):
                    run_preproc_target(config, target)
        case "p2vm":
            run_p2vm(config)
        case "reduce":
            for target in _targets(config, args.target):
                run_reduce(config, target)
        case "plot":
            for target in _targets(config, args.target):
                summary_plot(config, target)


def _targets(config: PipelineConfig, selection: str) -> tuple[str, ...]:
    if selection == "all":
        return tuple(config.targets)
    if selection not in config.targets:
        available = ", ".join(config.targets)
        raise ValueError(f"Unknown target {selection!r}; available targets: {available}")
    return (selection,)


if __name__ == "__main__":
    main()
