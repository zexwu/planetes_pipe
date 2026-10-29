"""PLANETES scientific reduction and optional ESO recipe adapters."""

import logging

__version__ = "0.15.0"
log = logging.getLogger("planetes_pipe")


def configure_logging(level: str = "INFO") -> None:
    """Enable console messages for a standalone Python application.

    Parameters
    ----------
    level : str, optional
        Standard Python log level name; default 'INFO'. Existing handlers are
        reused. Otherwise a console handler is added; propagation is disabled.

    Notes
    -----
    Call explicitly from a script if progress messages are wanted. ESO recipes
    use their scoped CPL adapter so PyEsoRex controls terminal and logfile levels.
    This function returns None and does not configure the root logger.
    """
    if not log.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("[%(name)s:%(lineno)d %(levelname)s] %(message)s")
        )
        log.addHandler(handler)
    log.setLevel(level.upper())
    log.propagate = False
