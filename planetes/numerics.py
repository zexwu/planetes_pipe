"""Small reusable numerical reductions."""

from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray


def frame_mean(
    chunks: Iterable[NDArray[np.float32]],
) -> NDArray[np.float32]:
    """Compute a temporal mean from frame chunks."""
    total = None
    count = 0
    for chunk in chunks:
        if total is None:
            total = np.zeros(chunk.shape[1:], dtype=np.float64)
        total += chunk.sum(axis=0, dtype=np.float64)
        count += len(chunk)
    if total is None or count == 0:
        raise ValueError("Selected cube contains no frames")
    return (total / count).astype(np.float32)


def frame_mean_std(
    chunks: Iterable[NDArray[np.float32]],
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Compute temporal mean and population RMS from frame chunks."""
    total = None
    total_squared = None
    count = 0
    for chunk in chunks:
        if total is None:
            total = np.zeros(chunk.shape[1:], dtype=np.float64)
            total_squared = np.zeros(chunk.shape[1:], dtype=np.float64)
        total += chunk.sum(axis=0, dtype=np.float64)
        total_squared += np.square(chunk, dtype=np.float64).sum(
            axis=0, dtype=np.float64
        )
        count += len(chunk)
    if total is None or total_squared is None or count == 0:
        raise ValueError("Selected cube contains no frames")

    mean = total / count
    variance = np.maximum(total_squared / count - mean**2, 0.0)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


def compute_gdelay(
    visdata: NDArray,
    wavelength: NDArray,
    search_range: tuple[float, float] = (-30, 30),
    n_newton: int = 5,
) -> NDArray:
    """Estimate group delay by grid search followed by Newton refinement."""
    k = -2j * np.pi / wavelength
    w = -2 * np.pi / wavelength

    gd_grid = np.arange(search_range[0], search_range[1], 0.01)
    phasors_grid = np.exp(gd_grid[:, None] * k[None, :])
    coherence = np.abs(np.tensordot(visdata, phasors_grid, axes=([-1], [-1])))

    best = np.argmax(coherence, axis=-1)
    gd_current = gd_grid[best]

    for _ in range(n_newton):
        gd_exp = gd_current[:, None]
        rotated = visdata * np.exp(1j * w * gd_exp)
        sum_0 = np.sum(rotated, axis=-1)
        sum_1 = np.sum(rotated * (1j * w), axis=-1)
        sum_2 = np.sum(rotated * (-(w**2)), axis=-1)

        sum_0_conjugate = np.conj(sum_0)
        gradient = 2 * np.real(sum_1 * sum_0_conjugate)
        hessian = 2 * np.real(
            sum_2 * sum_0_conjugate + sum_1 * np.conj(sum_1)
        )
        difference = np.zeros_like(gradient)
        valid = hessian != 0
        difference[valid] = gradient[valid] / hessian[valid]
        gd_current = gd_current - difference

    return gd_current
