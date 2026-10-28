"""PLANETES scientific reduction and optional ESO recipe adapters."""

import logging

__version__ = "0.8.1"
log = logging.getLogger("planetes")


def configure_logging(level: str = "INFO") -> None:
    """Configure console logging explicitly at an application entry point."""
    if not log.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("[%(name)s:%(lineno)d %(levelname)s] %(message)s")
        )
        log.addHandler(handler)
    log.setLevel(level.upper())
    log.propagate = False
