"""Small numerical helpers shared across scientific processing modules."""

from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray


def frame_mean(chunks: Iterable[NDArray]) -> NDArray:
    """Compute the equally weighted temporal mean of a sequence of frame chunks.

    Parameters
    ----------
    chunks : iterable of ndarray
        Nonempty sequence of (frame, y, x) arrays in ADU, all on the same detector
        grid. Supply float64 calculation arrays; raw readers do this conversion.

    Returns
    -------
    mean_adu : ndarray, shape (y, x)
        Mean over all frames, including a shorter final chunk.
    """
    total = None
    count = 0
    for chunk in chunks:
        if total is None:
            total = np.zeros(chunk.shape[1:])
        total += chunk.sum(axis=0)
        count += len(chunk)
    if total is None or count == 0:
        raise ValueError("Cube contains no frames")
    return total / count
