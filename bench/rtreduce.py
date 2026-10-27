"""Standalone benchmark and plot for real-time PLANETES P2VM reduction."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rtp2vm import P2VMReducer

from planetes.cube import iter_cube
from planetes.numerics import frame_mean


def main() -> None:
    """Benchmark and display visibility amplitudes for the fake science cube."""
    project_dir = Path(__file__).resolve().parents[1]
    reducer = P2VMReducer(project_dir / "reduced")

    # Set explicit bench files and calibration directory here before benchmarking.
    science = project_dir / "2026_02_05/sci2/sci.fits"
    science_dark = project_dir / "2026_02_05/sci2/sci_dark.fits"
    cube = np.concatenate(list(iter_cube(science)), axis=0)
    dark = frame_mean(iter_cube(science_dark))
    cube -= dark[None, ...]

    import timeit

    t = timeit.timeit(lambda: reducer.reduce(cube[0]), number=10000)
    print(f"Time per DIT: {t / 10000 * 1e3:.3f} ms")

    n_frame = len(cube)
    visamps = np.zeros((n_frame, reducer.n_baselines, len(reducer.wl_grid)))
    for i in range(n_frame):
        visamps[i] = reducer.reduce(cube[i]).visamp

    fig, axs = plt.subplots(2, 3, figsize=(8, 6), sharey=True, sharex=True)
    axs = axs.flatten()
    colors = plt.cm.viridis(np.linspace(0, 1, n_frame))
    for bsl in range(reducer.n_baselines):
        axs[bsl].set_prop_cycle(color=colors)
        axs[bsl].plot(reducer.wl_grid, visamps[:, bsl, :].T, alpha=0.05, lw=0.3)
        axs[bsl].set_ylabel("VISAMP")
        axs[bsl].set_xlabel("Wavelength [um]")
        axs[bsl].label_outer()
    plt.show()


if __name__ == "__main__":
    main()
