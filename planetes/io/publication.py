"""Verified non-overwriting publication with rollback of owned links."""

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def destinations(paths, *, verify):
    """Stage/verify all outputs, publish without overwrite, rollback only owned links."""
    paths = [Path(p).absolute() for p in paths]
    if len(set(paths)) != len(paths):
        raise ValueError("Duplicate output destination")
    for p in paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        if os.path.lexists(p):
            raise FileExistsError(f"Refusing to overwrite {p}")
    if len({p.parent for p in paths}) != 1:
        raise ValueError("Related outputs require one output directory")
    published = []
    with tempfile.TemporaryDirectory(
        prefix=".planetes-", dir=paths[0].parent
    ) as directory:
        temporary = [Path(directory) / p.name for p in paths]
        try:
            yield temporary
            for p in temporary:
                verify(p)
            for source, target in zip(temporary, paths, strict=True):
                os.link(source, target)
                published.append((target, source.stat().st_dev, source.stat().st_ino))
        except BaseException:
            for target, device, inode in reversed(published):
                try:
                    stat = target.lstat()
                    if (stat.st_dev, stat.st_ino) == (device, inode):
                        target.unlink()
                except FileNotFoundError:
                    pass
            raise
