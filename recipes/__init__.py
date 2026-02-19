#!/usr/bin/env python
"""
Pipeline framework for PLANETES Bench Data Reduction.

This module provides the core infrastructure for the data reduction pipeline,
including configuration management, command registration, and logging setup.
"""

import functools
import importlib
import logging
import pkgutil
from contextlib import nullcontext
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

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
    Manages configuration, IO, and derived dimensions for the pipeline.

    Attributes:
        conf_path: Path to the configuration YAML file
        data_dir: Directory containing input data files
        output_dir: Directory for output products
        conf: Loaded configuration dictionary
        baselines: List of baseline identifiers
        telescopes: List of telescope identifiers
    """

    conf_path: Union[str, Path]
    data_dir: Path = field(default_factory=Path)
    output_dir: Path = field(default=Path("output"))

    # Internal configuration state
    conf: Dict[str, Any] = field(init=False, repr=False)
    baselines: List[str] = field(init=False)
    telescopes: List[str] = field(init=False)

    def __post_init__(self) -> None:
        """Load configuration and setup paths."""
        self.conf_path = Path(self.conf_path)
        if not self.conf_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.conf_path}")

        with open(self.conf_path, "r") as f:
            self.conf = yaml.safe_load(f)

        # Allow config to override output directory
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
        """Number of baselines."""
        return len(self.baselines)

    @property
    def n_tel(self) -> int:
        """Number of telescopes."""
        return len(self.telescopes)

    @property
    def n_reg(self) -> int:
        """Number of detector regions (assuming ABCD encoding)."""
        return 4 * self.n_bsl

    @property
    def n_data(self) -> int:
        """Total data dimensions: Photometry + Real + Imaginary parts."""
        return self.n_tel + 2 * self.n_bsl

    # --- Slices (Derived) ---
    @cached_property
    def sl_data(self) -> Tuple[slice, ...]:
        """
        Standard extraction slice [Frames, Y, X].

        Returns:
            Slice object for data extraction
        """
        fr = self.conf.get("frames", [0, None])
        xr = self.conf.get("xrange", [0, None])
        yr = self.conf.get("yrange", [0, None])
        return np.s_[fr[0] : fr[1], yr[0] : yr[1], xr[0] : xr[1]]

    @property
    def sl_real(self) -> slice:
        """Slice for the real part of visibility."""
        return np.s_[self.n_tel : self.n_tel + self.n_bsl]

    @property
    def sl_imag(self) -> slice:
        """Slice for the imaginary part of visibility."""
        return np.s_[self.n_tel + self.n_bsl : self.n_tel + 2 * self.n_bsl]

    # --- IO Helpers ---
    def _get_filename(self, product_id: Union[str, Tuple[str, str]]) -> str:
        """
        Look up filename for a product ID.

        Args:
            product_id: Product identifier (string or tuple)

        Returns:
            Filename for the product
        """
        fn_dict = self.conf.get("products", {})
        if isinstance(product_id, tuple):
            template = fn_dict.get(product_id[0], f"{product_id}.npz")
            return template.format(product_id[1])
        return fn_dict.get(product_id, f"{product_id}.npz")

    def product_exists(self, product_id: str) -> bool:
        """
        Check if a product file exists.

        Args:
            product_id: Product identifier

        Returns:
            True if the product file exists
        """
        path = self.output_dir / self._get_filename(product_id)
        return path.exists()

    def load_fits(self, filename: str, **kwargs) -> np.ndarray:
        """
        Load FITS file and apply data slice.

        Args:
            filename: Name of the FITS file
            **kwargs: Additional arguments to pass to fits.getdata

        Returns:
            Numpy array containing the data
        """
        path = self.data_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"FITS file not found: {path}")

        data = fits.getdata(path, **kwargs)
        return data[self.sl_data].astype(np.float32)

    def save_product(self, product_id: str, **kwargs) -> None:
        """
        Save pipeline product to disk.

        Args:
            product_id: Product identifier
            **kwargs: Data to save (key-value pairs)
        """
        filename = self._get_filename(product_id)
        np.savez(self.output_dir / filename, **kwargs)

    def load_product(self, product_id: str, **kwargs) -> Dict[str, Any]:
        """
        Load pipeline product from disk.

        Args:
            product_id: Product identifier
            **kwargs: Additional arguments to pass to np.load

        Returns:
            Dictionary containing the loaded data
        """
        filename = self._get_filename(product_id)
        return np.load(self.output_dir / filename, allow_pickle=True, **kwargs)

    def plot_ctx(self, filename: str) -> Union[PdfPages, nullcontext]:
        """
        Create a plotting context for PDF output.

        Args:
            filename: Name of the PDF file

        Returns:
            PDF context manager or null context
        """
        if self.conf.get("plot_to_pdf", False):
            return PdfPages(self.output_dir / filename)
        return nullcontext()


# --- 2. Registry & CLI ---
COMMANDS: Dict[str, callable] = {}


def _infer_dest(opt_strings: Tuple[str, ...], kw: Dict[str, Any]) -> Optional[str]:
    """
    Infer destination name from option strings.

    Args:
        opt_strings: Tuple of option strings (e.g., ("--threshold", "-t"))
        kw: Keyword arguments passed to add_argument

    Returns:
        Destination name or None
    """
    if "dest" in kw:
        return kw["dest"]

    opts = [s for s in opt_strings if isinstance(s, str) and s.startswith("-")]
    if not opts:
        return None

    # Prefer the longest option (usually the long form)
    opt = max(opts, key=len)
    return opt.lstrip("-").replace("-", "_")


def arg(*opt_strings: str, **kwargs: Any) -> callable:
    """
    Decorator to add argparse arguments to a specific command.

    Usage:
        @arg("--threshold", type=float, default=5.0, help="Threshold value")
        def my_command(ctx, threshold):
            ...

    Args:
        *opt_strings: Option strings for argparse
        **kwargs: Keyword arguments for argparse.add_argument

    Returns:
        Decorator function
    """

    def decorator(func: callable) -> callable:
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
            meta["defaults"][dest] = kwargs.get("default")

        return func

    return decorator


def command(name: str, help_msg: str,
            requires: Optional[List[str]] = None,
            produces: Optional[List[str]] = None) -> callable:
    """
    Decorator to register pipeline steps.

    Args:
        name: Command name
        help_msg: Help message for the command
        requires: List of required input products
        produces: List of produced output products

    Returns:
        Decorator function
    """

    def decorator(func: callable) -> callable:
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
        def runner(ctx: PipelineContext, **cli_kwargs: Any) -> Any:
            """
            Wrapper function that filters CLI arguments before calling the command.
            """
            m = getattr(func, "meta", {})
            defaults = m.get("defaults", {})
            dests = m.get("dests", set())

            merged = {**defaults, **cli_kwargs}
            filtered = {k: v for k, v in merged.items() if k in dests}

            return func(ctx, **filtered)

        runner.meta = meta
        COMMANDS[name] = runner
        return runner

    return decorator


# --- 3. Load Registered Recipes ---
for loader, recipe_name, is_pkg in pkgutil.walk_packages(__path__):
    if recipe_name == "__init__":
        continue
    try:
        # Dynamically import the module using its name
        importlib.import_module(f".{recipe_name}", package=__name__)
        log.info(f"Loaded recipe: {recipe_name}")
    except Exception as e:
        log.warning(f"Failed to load recipe '{recipe_name}': {e}")
