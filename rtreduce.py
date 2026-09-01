"""Standalone benchmark and plot for real-time PLANETES P2VM reduction."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from .rtp2vm import P2VMReducer, extract_spec_sparse_single, twopx_interp_single
except ImportError:
    from rtp2vm import P2VMReducer, extract_spec_sparse_single, twopx_interp_single

from planetes.config import PipelineConfig
from planetes.cube import CubeSelection, iter_cube
from planetes.numerics import frame_mean


def main() -> None:
    """Benchmark and display visibility amplitudes for the fake science cube."""
    base_dir = Path(__file__).resolve().parent
    reducer = P2VMReducer(base_dir / "reduced")

    config = PipelineConfig.from_yaml(base_dir / "conf.yaml")
    target = config.targets["sci"]
    selection = CubeSelection(
        target.frames.slice,
        config.crop.y.slice,
        config.crop.x.slice,
    )
    cube = np.concatenate(
        list(
            iter_cube(
                target.data,
                selection,
                config.chunk_frames,
                hdu=config.input_hdu,
            )
        ),
        axis=0,
    )
    if target.dark is not None:
        dark_selection = CubeSelection(
            target.dark_frames.slice,
            config.crop.y.slice,
            config.crop.x.slice,
        )
        dark = frame_mean(
            iter_cube(
                target.dark,
                dark_selection,
                config.chunk_frames,
                hdu=config.input_hdu,
            )
        )
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
