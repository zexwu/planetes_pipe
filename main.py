#!/usr/bin/env python
import sys

import argparse
import recipes

def main():
    parser = argparse.ArgumentParser(description="PLANETES Bench Data Reduction Pipeline")
    parser.add_argument("--conf", default="conf.yaml", help="Path to configuration YAML")

    subs = parser.add_subparsers(dest="command", required=True)
    for name, func in recipes.COMMANDS.items():
        # Create the subparser for this command
        sub = subs.add_parser(name, help=func.meta["help"])

        # Add the specific arguments defined by @arg
        for args_list, kwargs_dict in reversed(func.meta.get("args", [])):
            sub.add_argument(*args_list, **kwargs_dict)
        sub.set_defaults(func=func)

    args = parser.parse_args()
    ctx = recipes.PipelineContext(args.conf)
    recipes.log.setLevel(ctx.conf.get("log_level", "DEBUG"))
    missing = []
    for req in args.func.meta["requires"]:
        if not ctx.product_exists(req):
            missing.append(req)
    if missing:
        recipes.log.error(f"Failed to run {args.func.meta["name"]}\n"
                          f"Missing required inputs: {missing}")
        sys.exit(1)
    args.func(ctx, **vars(args))

if __name__ == "__main__":
    main()
