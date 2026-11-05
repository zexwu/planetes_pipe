"""Pipeline logging: one logger per module, every message tagged with its step.

Every line names the step running (DARK, FLAT, WAVE, P2VM, SCIENCE), the module
and the function that wrote it:

    16:02:11 INFO    [WAVE] wave.make_wave: lines matched per output: 14-18 of 30

Each step's run() is wrapped by @step: it logs the inputs, the products written
and the time taken, or which step failed and why. Summaries are INFO; per-trace,
per-output and per-iteration details are DEBUG.

The library writes nothing until a script calls configure_logging(); PyEsoRex
recipes route the same records through CPL instead (eso.recipes.recipe_logging).
"""
import functools
import inspect
import logging
import time
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

ROOT = "planetes_pipe"
_STEP: ContextVar[str] = ContextVar("planetes_step", default="")
CONSOLE_FORMAT = "%(asctime)s %(levelname)-7s [%(step)s] %(where)s: %(message)s"
FILE_FORMAT = "%(asctime)s.%(msecs)03d " + CONSOLE_FORMAT.removeprefix("%(asctime)s ")


class StepFormatter(logging.Formatter):
    """A Formatter that also knows %(step)s, the step running when the record was made,
    and %(where)s, module.function that wrote it."""

    def format(self, record: logging.LogRecord) -> str:
        if not hasattr(record, "step"):
            record.step = _STEP.get() or "-"
        if not hasattr(record, "where"):
            record.where = f"{record.module}.{record.funcName}"
        return super().format(record)


def current_step() -> str:
    return _STEP.get()


@contextmanager
def running(name: str, where: str = "pipeline"):
    """Tag every record made inside with step `name`; log its time, or that it failed.

    where names the caller in those two records, e.g. 'wave.run'.
    """
    log = logging.getLogger(f"{ROOT}.steps")
    token = _STEP.set(name)
    start = time.perf_counter()
    try:
        yield
    except Exception as error:
        log.error("failed after %.1f s: %s: %s", time.perf_counter() - start,
                  type(error).__name__, error, extra={"where": where})
        raise
    else:
        log.info("done in %.1f s", time.perf_counter() - start, extra={"where": where})
    finally:
        _STEP.reset(token)


def _describe(value) -> str | None:
    """A short form of a file argument: its name, or the names of a list of files."""
    if value is None:
        return None
    if isinstance(value, (str, Path)):
        return Path(value).name
    if isinstance(value, (list, tuple)):
        names = [Path(v).name for v in value]
        return f"{len(names)} files: " + ", ".join(names) if len(names) > 1 else ", ".join(names)
    if isinstance(value, dict):
        return ", ".join(f"{k}={Path(v).name}" for k, v in value.items())
    return None


def step(name: str):
    """Decorate a step's run(): tag its records with `name`, log inputs, products and time."""
    def decorate(run):
        signature = inspect.signature(run)
        log = logging.getLogger(run.__module__)
        where = {"where": f"{run.__module__.rsplit('.', 1)[-1]}.{run.__name__}"}

        @functools.wraps(run)
        def wrapper(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            with running(name, where["where"]):
                for argument, value in bound.arguments.items():
                    if argument.endswith(("_file", "_files")):
                        described = _describe(value)
                        if described:
                            label = (argument.removesuffix("_files") + "s"
                                     if argument.endswith("_files")
                                     else argument.removesuffix("_file"))
                            log.info("input %s: %s", label, described, extra=where)
                written = run(*args, **kwargs)
                for category, path in written.items():
                    log.info("wrote %s: %s", category, Path(path).name, extra=where)
                return written
        return wrapper
    return decorate


def configure_logging(level: str = "INFO", file=None, file_level: str = "DEBUG") -> None:
    """Show pipeline messages on the console, and optionally write them to a file.

    Parameters
    ----------
    level : str, optional
        Console level name; default 'INFO'. 'DEBUG' adds per-trace, per-output
        and per-iteration details.
    file : path, optional
        Also append every record at file_level (default DEBUG) to this file,
        with milliseconds.

    Notes
    -----
    Call it from a script if messages are wanted; the library is otherwise silent
    (apart from Python's last-resort output of warnings). Calling it again replaces
    the handlers it added. The root logger is not touched.
    """
    log = logging.getLogger(ROOT)
    for handler in [h for h in log.handlers if getattr(h, "_planetes", False)]:
        log.removeHandler(handler)
        handler.close()
    console = logging.StreamHandler()
    console.setLevel(level.upper())
    console.setFormatter(StepFormatter(CONSOLE_FORMAT, datefmt="%H:%M:%S"))
    handlers = [console]
    if file is not None:
        Path(file).parent.mkdir(parents=True, exist_ok=True)
        to_file = logging.FileHandler(file, encoding="utf-8")
        to_file.setLevel(file_level.upper())
        to_file.setFormatter(StepFormatter(FILE_FORMAT, datefmt="%Y-%m-%d %H:%M:%S"))
        handlers.append(to_file)
    for handler in handlers:
        handler._planetes = True
        log.addHandler(handler)
    log.setLevel(min(h.level for h in handlers))
    log.propagate = False
