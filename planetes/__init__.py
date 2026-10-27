"""PLANETES data reduction."""

import logging

import colorlog

__version__ = "0.2.0"


log = logging.getLogger("planetes")
if not log.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        colorlog.ColoredFormatter(
            "[%(filename)-12s:%(lineno)-4s %(log_color)s%(levelname)5s%(reset)s] "
            "%(message)s"
        )
    )
    log.addHandler(handler)
log.propagate = False


def configure_logging(level: str = "INFO") -> None:
    """Set the package log level for the current process."""
    log.setLevel(level.upper())
