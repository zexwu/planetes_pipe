"""PLANETES reduction pipeline.

    steps/   dark, flat, wave, p2vm, science: product, algorithm, QC, FITS, run()
    io/      the file format: keywords, raw files, product files
    eso/     optional PyEsoRex recipes, which call the steps' run() functions
"""

import logging

__version__ = "0.40.1"
log = logging.getLogger("planetes_pipe")

from planetes_pipe.logs import configure_logging  # noqa: E402

__all__ = ["configure_logging", "log"]
