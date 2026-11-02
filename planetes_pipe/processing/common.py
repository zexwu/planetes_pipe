"""Small numerical helpers shared across scientific processing modules."""

from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray
from planetes_pipe.model.raw import Raw, shutter_identity


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


def order_raws(raws: list[Raw], identities: tuple[str, ...]) -> list[Raw]:
    """Order a calibration sequence by recorded shutters, rejecting missing/duplicate inputs."""
    grouped = {}
    for raw in raws:
        identity = shutter_identity(raw.shutters)
        if identity in grouped:
            raise ValueError(f"Duplicate calibration shutter state: {identity}")
        grouped[identity] = raw
    if set(grouped) != set(identities):
        raise ValueError(f"Expected shutter configurations {identities}; found {tuple(grouped)}")
    return [grouped[key] for key in identities]
