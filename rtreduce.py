"""Standalone benchmark and plot for real-time PLANETES P2VM reduction."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from . import PipelineContext
    from .rtp2vm import P2VMReducer, extract_spec_sparse_single, twopx_interp_single
except ImportError:
    from recipes import PipelineContext
    from rtp2vm import P2VMReducer, extract_spec_sparse_single, twopx_interp_single


def main() -> None:
    """Benchmark and display visibility amplitudes for the fake science cube."""
    base_dir = Path(__file__).resolve().parent
    reducer = P2VMReducer(base_dir / "reduced")

    ctx = PipelineContext(base_dir / "conf.yaml")
    cube = ctx.load_fits(Path("./2026_02_05/sci2/sci.fits"))
    dark = ctx.load_fits(Path("./2026_02_05/sci2/sci_dark.fits"))
    cube -= dark.mean(axis=0)[None, ...]

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
