"""DARK and BAD sections: detector maps, pixel distributions, bad pixels by kind."""
import numpy as np

from planetes_pipe.plots.pages import Section
from planetes_pipe.steps.dark import Bad, Dark

BAD_KINDS = (("hot", 1, "red"), ("dead", 2, "deepskyblue"), ("noisy", 4, "gold"))


def sections(dark: Dark | None = None, bad: Bad | None = None) -> list[Section]:
    """Sections for a DARK, a BAD, or both."""
    result = []
    if dark is not None:
        maps = [("Temporal mean", dark.mean_adu)]
        if dark.temporal_rms_adu is not None:
            maps.append(("Temporal RMS", dark.temporal_rms_adu))

        def draw_maps(fig):
            for ax, (title, image) in zip(np.atleast_1d(fig.subplots(1, len(maps))), maps):
                low, high = np.percentile(image, [1, 99])
                shown = ax.imshow(image, origin="lower", aspect="auto", cmap="inferno",
                                  vmin=low, vmax=high)
                fig.colorbar(shown, ax=ax, label="ADU", shrink=0.8)
                ax.set(title=title, xlabel="Detector x [pixel]", ylabel="Detector y [pixel]")

        def draw_histograms(fig):
            for ax, (title, image) in zip(np.atleast_1d(fig.subplots(1, len(maps))), maps):
                median = float(np.median(image))
                ax.hist(image.ravel(), bins=300, log=True, color="gray")
                ax.axvline(median, color="C3", label=f"median {median:.2f} ADU")
                ax.set(title=title, xlabel="ADU", ylabel="Pixels")
                ax.legend()

        frames = f"{dark.n_frames} frames; " if dark.n_frames else ""
        result += [Section("DARK detector maps", draw_maps, 3.4,
                           f"{frames}colour range 1–99 % of the pixels."),
                   Section("DARK pixel distributions", draw_histograms, 2.4,
                           "Histograms on a logarithmic scale; red: median.")]
    if bad is not None:
        flags = bad.bad_pixel_flags
        counts = {name: int(np.count_nonzero(flags & bit)) for name, bit, _ in BAD_KINDS}

        def draw_bad(fig):
            ax = fig.subplots()
            for name, bit, color in BAD_KINDS:
                ys, xs = np.nonzero(flags & bit)
                ax.scatter(xs, ys, s=1.5, color=color, label=f"{name}: {len(xs)}", linewidths=0)
            ax.set(xlim=(0, flags.shape[1]), ylim=(0, flags.shape[0]), facecolor="black",
                   xlabel="Detector x [pixel]", ylabel="Detector y [pixel]")
            ax.legend(loc="upper right", markerscale=5)

        result.append(Section(
            f"BAD pixels: {np.count_nonzero(flags)} flagged", draw_bad, 5.0,
            ", ".join(f"{n} {name}" for name, n in counts.items())
            + ". Hot and dead from the temporal mean, noisy from the temporal RMS."))
    return result
