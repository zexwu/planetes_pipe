#!/usr/bin/env python
"""
PLANETES Bench Data Reduction Pipeline - Main Entry Point.

This script provides the command-line interface for the data reduction pipeline,
parsing arguments and dispatching to the appropriate recipe commands.
"""

import sys
import argparse

import planetes


def main() -> None:
    """
    Main entry point for the pipeline CLI.

    Parses command-line arguments, validates required inputs, and executes
    the requested pipeline command.
    """
    parser = argparse.ArgumentParser(
        description="PLANETES Bench Data Reduction Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--conf",
        default="conf.yaml",
        help="Path to configuration YAML file"
    )

    # Create subparsers for each registered command
    subs = parser.add_subparsers(dest="command", required=True)

    for name, func in planetes.COMMANDS.items():
        # Create the subparser for this command
        sub = subs.add_parser(name, help=func.meta["help"])

        # Add the specific arguments defined by @arg decorator
        for args_list, kwargs_dict in reversed(func.meta.get("args", [])):
            arg_kwargs = dict(kwargs_dict)
            arg_kwargs["default"] = argparse.SUPPRESS
            sub.add_argument(*args_list, **arg_kwargs)
        sub.set_defaults(func=func)

    # Parse arguments
    args = parser.parse_args()

    # Initialize pipeline context
    try:
        ctx = planetes.PipelineContext(args.conf)
    except (FileNotFoundError, TypeError, ValueError) as exc:
        parser.error(str(exc))

    # Configure logging level from config
    log_level = ctx.conf.get("log_level", "INFO")
    planetes.log.setLevel(log_level)

    # Check for required input products
    missing = []
    for req in args.func.meta["requires"]:
        if not ctx.product_exists(req):
            missing.append(req)

    if missing:
        planetes.log.error(
            f"Failed to run {args.func.meta['name']}\n"
            f"Missing required inputs: {missing}"
        )
        sys.exit(1)

    # Execute the command with filtered arguments
    args.func(ctx, **vars(args))


if __name__ == "__main__":
    main()
