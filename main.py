#!/usr/bin/env python
"""Explicit command-line entry point for the PLANETES pipeline."""

import argparse

from planetes import configure_logging
from planetes.config import PipelineConfig
from planetes.recipes.calibration import run_calibration
from planetes.recipes.plotting import summary_plot
from planetes.recipes.reduction import run_reduce


def build_parser() -> argparse.ArgumentParser:
    """Build the small, explicit pipeline CLI."""
    parser = argparse.ArgumentParser(
        description="PLANETES data reduction pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--conf", default="conf.yaml", help="configuration YAML")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("calib", help="run the full calibration sequence")

    reduce = commands.add_parser("reduce", help="reduce one target or all targets")
    reduce.add_argument("target", help="target name or all")

    plot = commands.add_parser("plot", help="plot one reduced target or all targets")
    plot.add_argument("target", help="target name or all")
    plot.add_argument(
        "--full",
        action="store_true",
        help="include frame- and wavelength-sampled diagnostic pages",
    )
    return parser


def main() -> None:
    """Load one configuration and dispatch one explicit recipe."""
    parser = build_parser()
    args = parser.parse_args()
    try:
        config = PipelineConfig.from_yaml(args.conf)
    except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    configure_logging()

    match args.command:
        case "calib":
            run_calibration(config)
        case "reduce":
            for target in _selected_targets(parser, config, args.target):
                run_reduce(config, target)
        case "plot":
            for target in _selected_targets(parser, config, args.target):
                summary_plot(config, target, full=args.full)


def _selected_targets(
    parser: argparse.ArgumentParser,
    config: PipelineConfig,
    selection: str,
) -> tuple[str, ...]:
    if selection == "all":
        return tuple(config.targets)
    try:
        config.target(selection)
    except ValueError as exc:
        parser.error(str(exc))
    return (selection,)


if __name__ == "__main__":
    main()
