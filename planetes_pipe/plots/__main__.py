"""Write the diagnostic PDF of each product file given: python -m planetes_pipe.plots FILE..."""
import sys

from planetes_pipe.plots import report

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    for name in sys.argv[1:]:
        try:
            print(report(name))
        except (ValueError, FileExistsError) as error:
            print(f"skipped {name}: {error}")
