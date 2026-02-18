#!/usr/bin/env python
import importlib
import logging
import pkgutil
import functools
from contextlib import nullcontext
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import colorlog
import numpy as np
import yaml
from astropy.io import fits
from matplotlib.backends.backend_pdf import PdfPages

# --- 0. Setup Logging ---
log = logging.getLogger(__name__)
handler = logging.StreamHandler()
FMT = "[%(filename)-12s:%(lineno)-4s %(log_color)s%(levelname)5s%(reset)s] %(message)s"
formatter = colorlog.ColoredFormatter(FMT)
handler.setFormatter(formatter)
log.addHandler(handler)


# --- 1. Configuration & Context ---
@dataclass
class PipelineContext:
    """
    Manages configuration, IO, and derived dimensions.
    """

    conf_path: Union[str, Path]
    data_dir: Path = field(default_factory=Path)
    output_dir: Path = field(default=Path("output"))

    # Internal configuration state
    conf: Dict[str, Any] = field(init=False, repr=False)
    baselines: List[str] = field(init=False)
    telescopes: List[str] = field(init=False)

    def __post_init__(self):
        """Load config and setup paths."""
        self.conf_path = Path(self.conf_path)
        if not self.conf_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.conf_path}")

        with open(self.conf_path, "r") as f:
            self.conf = yaml.safe_load(f)

        # Allow config to override output dir
        if "output_dir" in self.conf:
            self.output_dir = Path(self.conf["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir = Path(self.conf["data_dir"])

        # Populate core lists
        self.baselines = self.conf.get("baselines", [])
        self.telescopes = self.conf.get("telescopes", [])

    # --- Fixed Dimensions ---
    @property
    def n_bsl(self) -> int:
        return len(self.baselines)

    @property
    def n_tel(self) -> int:
        return len(self.telescopes)

    @property
    def n_reg(self) -> int:
        return 4 * self.n_bsl  # assuming ABCD encoding

    @property
    def n_data(self) -> int:
        # Total data : Photometry + Real + Imag
        return (self.n_tel + 2 * self.n_bsl)  

    # --- Slices (Derived) ---
    @cached_property
    def sl_data(self) -> slice | Any:
        """Standard extraction slice [Frames, Y, X]."""
        fr = self.conf.get("frames", [0, None])
        xr = self.conf.get("xrange", [0, None])
        yr = self.conf.get("yrange", [0, None])
        return np.s_[fr[0] : fr[1], yr[0] : yr[1], xr[0] : xr[1]]

    @property
    def sl_real(self) -> slice:
        """Slice for Real part of visibility."""
        return np.s_[self.n_tel : self.n_tel + self.n_bsl]

    @property
    def sl_imag(self) -> slice:
        """Slice for Imaginary part of visibility."""
        return np.s_[self.n_tel + self.n_bsl : self.n_tel + 2 * self.n_bsl]

    # --- IO Helpers ---
    def _get_filename(self, product_id: str | Tuple[str, str]) -> str:
        """Look up filename for a product ID, with fallback to {product_id}.npz."""
        fn_dict = self.conf.get("products", {})
        if isinstance(product_id, tuple):
            template = fn_dict.get(product_id[0], f"{product_id}.npz")
            return template.format(product_id[1])
        return fn_dict.get(product_id, f"{product_id}.npz")

    def product_exists(self, product_id: str) -> bool:
        """Check if a product file exists based on the config mapping."""
        path = self.output_dir / self._get_filename(product_id)
        return path.exists()

    def load_fits(self, filename: str, **kwargs):
        """Load FITS and apply slice."""
        path = self.data_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing: {path}")
        return fits.getdata(path, **kwargs)[self.sl_data].astype(np.float32)
        return fits.getdata(path, "IMAGING_DATA_SC")[self.sl_data]

    def save_product(self, product_id: str, **kwargs) -> None:
        filename = self._get_filename(product_id)
        np.savez(self.output_dir / filename, **kwargs)
        return

    def load_product(self, product_id: str, **kwargs) -> Dict[str, Any]:
        filename = self._get_filename(product_id)
        return np.load(self.output_dir / filename, allow_pickle=True)

    def plot_ctx(self, filename: str):
        if self.conf.get("plot_to_pdf"):
            return PdfPages(self.output_dir / filename)
        return nullcontext()


# --- 2. Registry & CLI ---
COMMANDS = {}


def _infer_dest(opt_strings, kw):
    if "dest" in kw:
        return kw["dest"]
    opts = [s for s in opt_strings if isinstance(s, str) and s.startswith("-")]
    if not opts:
        return None
    opt = max(opts, key=len)  # prefer --long-opt
    return opt.lstrip("-").replace("-", "_")


def arg(*opt_strings, **kwargs):
    """
    Decorator to add argparse arguments to a specific command.
    Usage: @arg("--threshold", type=float, default=5.0, help="...")
    """

    def deco(func):
        meta = getattr(func, "meta", None)
        if meta is None:
            meta = {"args": [], "defaults": {}, "dests": set()}
            func.meta = meta
        else:
            meta.setdefault("args", [])
            meta.setdefault("defaults", {})
            meta.setdefault("dests", set())

        meta["args"].append((opt_strings, kwargs))

        dest = _infer_dest(opt_strings, kwargs)
        if dest:
            meta["dests"].add(dest)
            meta["defaults"][dest] = kwargs["default"]

        return func

    return deco



def command(name: str, help_msg: str, requires: List[str] = None, produces: List[str] = None):
    """Decorator to register pipeline steps."""

    def deco(func):
        meta = getattr(func, "meta", None)
        if meta is None:
            meta = {"args": [], "defaults": {}, "dests": set()}
            func.meta = meta

        meta.update({
            "name": name,
            "help": help_msg,
            "requires": requires or [],
            "produces": produces or [],
        })

        @functools.wraps(func)
        def runner(ctx, **cli_kwargs):
            m = getattr(func, "meta", {})
            defaults = m.get("defaults", {})
            dests = m.get("dests", set())

            merged = {**defaults, **cli_kwargs}
            filtered = {k: v for k, v in merged.items() if k in dests}

            return func(ctx, **filtered)

        runner.meta = meta
        COMMANDS[name] = runner
        return runner

    return deco


# --- 3. Load Registered Recipes ---
for loader, recipe_name, is_pkg in pkgutil.walk_packages(__path__):
    if recipe_name == "__init__":
        continue
    try:
        # dynamically import the module using its name
        importlib.import_module(f".{recipe_name}", package=__name__)
        log.info(f"Load recipe: {recipe_name}")
    except Exception as e:
        log.warning(f"Failed to load recipe '{recipe_name}': {e}")
