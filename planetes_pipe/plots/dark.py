"""DARK and BAD sections: dark images, pixel distributions, bad pixel statistics and image.

Titles follow GRAVITY's gravi_visual_dark and gravi_visual_badpix where the plot is the same.
"""
import numpy as np

from planetes_pipe.plots.pages import ABCD_COLORS, FLAG, Section, name, panels, table, table_height
from planetes_pipe.steps.dark import Bad, Dark

BAD_KINDS = (("hot", 1, "#FF2C00"), ("dead", 2, "#00B4FF"), ("noisy", 4, "#FFD000"))
MAX_BAD_PIXELS = 10000  # GRAVITY's limit of acceptable bad pixels; above it the count is shaded


def sections(dark: Dark | None = None, bad: Bad | None = None) -> list[Section]:
    """Sections for a DARK, a BAD, or both."""
    result = []
    if dark is not None:
        maps = [("SC dark", dark.mean_adu)]
        if dark.temporal_rms_adu is not None:
            maps.append(("SC dark standard deviation", dark.temporal_rms_adu))

        def draw_maps(fig):
            axes = panels(fig, 1, len(maps), "Detector x [pixel]", "Detector y [pixel]", sharey=True)
            for ax, (title, image) in zip(axes, maps):
                low, high = np.percentile(image, [1, 99])
                shown = ax.imshow(image, origin="lower", aspect="auto", vmin=low, vmax=high)
                fig.colorbar(shown, ax=ax, label=f"{title} [ADU/DIT]", shrink=0.9, pad=0.02)
                ax.minorticks_off()

        def draw_histograms(fig):
            axes = panels(fig, 1, len(maps), "ADU", "Pixels", sharex=False)
            for ax, (title, image) in zip(axes, maps):
                median = float(np.median(image))
                ax.hist(image.ravel(), bins=300, log=True, color="0.45", histtype="stepfilled")
                ax.axvline(median, color=ABCD_COLORS[3], lw=0.7)
                name(ax, f"{title}\nmedian {median:.2f} ADU", "upper right")

        frames = f"{dark.n_frames} frames; " if dark.n_frames else ""
        result += [Section("SC dark image and standard deviation (ADU/DIT)", draw_maps, 3.0,
                           f"Per-pixel mean and standard deviation over the frames; {frames}colour range 1–99 % of the pixels."),
                   Section("DARK pixel distributions", draw_histograms, 2.0,
                           "Histograms on a logarithmic scale; red line: median.")]
    if bad is not None:
        flags = bad.bad_pixel_flags
        counts = {kind: int(np.count_nonzero(flags & bit)) for kind, bit, _ in BAD_KINDS}

        def draw_bad(fig):
            ax = fig.subplots()
            for kind, bit, color in BAD_KINDS:
                ys, xs = np.nonzero(flags & bit)
                ax.scatter(xs, ys, s=1.5, color=color, label=f"{kind}: {len(xs)}", linewidths=0)
            ax.set(xlim=(0, flags.shape[1]), ylim=(0, flags.shape[0]), facecolor="black",
                   xlabel="Detector x [pixel]", ylabel="Detector y [pixel]")
            fig.legend(loc="outside upper center", ncol=len(BAD_KINDS), markerscale=4)

        total = int(np.count_nonzero(flags))
        statistics = [[str(total)], [f"{100 * total / flags.size:.4f} %"]]
        result.append(Section(
            "Statistics", lambda fig: table(
                fig, statistics, None, ["Number of SC bad pixels", "Fraction of bad pixels in SC"],
                shade=[((0, 0), FLAG)] if total > MAX_BAD_PIXELS else []),
            table_height(1), f"Counts above GRAVITY's limit of {MAX_BAD_PIXELS} are shaded."))
        result.append(Section(
            "SC bad pixel image", draw_bad, 4.6,
            ", ".join(f"{n} {kind}" for kind, n in counts.items())
            + ". With $\\tilde m,\\tilde s$ the medians of the temporal mean and RMS: hot $> \\tilde m + 5\\tilde s$, "
            "dead $< \\tilde m - 5\\tilde s$, noisy RMS $> 5\\tilde s$ (default threshold 5)."))
    return result
