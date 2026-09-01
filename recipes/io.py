"""Low-level atomic file writing helpers."""

import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


def write_npz_atomic(path: Path, values: dict[str, Any]) -> None:
    """Write an NPZ product atomically in its destination directory."""
    fd, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez(stream, **values)
            stream.flush()
            os.fsync(stream.fileno())
        temporary_path.replace(path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
