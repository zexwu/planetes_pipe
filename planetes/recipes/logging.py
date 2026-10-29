"""Route scientific Python log messages through the active CPL executor."""

import logging
from collections.abc import Iterator
from contextlib import contextmanager

from cpl.core import Msg

from planetes import log


class _CPLHandler(logging.Handler):
    """Forward records without applying a second terminal or file verbosity filter."""

    def emit(self, record: logging.LogRecord) -> None:
        if record.levelno >= logging.ERROR:
            send = Msg.error
        elif record.levelno >= logging.WARNING:
            send = Msg.warning
        elif record.levelno >= logging.INFO:
            send = Msg.info
        else:
            send = Msg.debug
        send(record.funcName, self.format(record))


@contextmanager
def recipe_logging() -> Iterator[None]:
    """Use CPL logging for one recipe invocation, restoring Python logging on exit.

    CPL owns the independent terminal and logfile levels set by PyEsoRex.
    Forward DEBUG and above so its logfile can be more verbose than its terminal.
    Existing Python handlers, level and propagation are restored even on failure.
    Use as a context manager or as a decorator on a recipe's ``run`` method.
    """
    handlers, level, propagate = log.handlers, log.level, log.propagate
    handler = _CPLHandler()
    log.handlers = [handler]
    log.setLevel(logging.DEBUG)
    log.propagate = False
    try:
        yield
    finally:
        log.handlers = handlers
        log.setLevel(level)
        log.propagate = propagate
        handler.close()
