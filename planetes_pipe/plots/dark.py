"""DARK and BAD pages: detector maps, pixel distributions, bad pixels by kind."""
import numpy as np

from planetes_pipe.plots.pages import finish, page
from planetes_pipe.steps.dark import Bad, Dark

BAD_KINDS = (("hot", 1, "red"), ("dead", 2, "deepskyblue"), ("noisy", 4, "gold"))


def pages(dark: Dark | None = None, bad: Bad | None = None):
    """Pages for a DARK, a BAD, or both."""
    if dark is not None:
        maps = [("Temporal mean", dark.mean_adu)]
        if dark.temporal_rms_adu is not None:
            maps.append(("Temporal RMS", dark.temporal_rms_adu))
        fig = page()
        for ax, (title, image) in zip(np.atleast_1d(fig.subplots(1, len(maps))), maps):
            low, high = np.percentile(image, [1, 99])
            shown = ax.imshow(image, origin="lower", aspect="auto", cmap="inferno",
                              vmin=low, vmax=high)
            fig.colorbar(shown, ax=ax, label="ADU", shrink=0.7)
            ax.set(title=f"{title} (colour range: 1–99 %)", xlabel="Detector x [pixel]",
                   ylabel="Detector y [pixel]")
        frames = f", {dark.n_frames} frames" if dark.n_frames else ""
        yield finish(fig, f"DARK — detector maps{frames}")

        fig = page()
        for ax, (title, image) in zip(np.atleast_1d(fig.subplots(1, len(maps))), maps):
            median = float(np.median(image))
            ax.hist(image.ravel(), bins=300, log=True, color="gray")
            ax.axvline(median, color="C3", label=f"median {median:.2f} ADU")
            ax.set(title=title, xlabel="ADU", ylabel="Pixels")
            ax.legend()
        yield finish(fig, "DARK — pixel distributions (log scale)")

    if bad is not None:
        flags = bad.bad_pixel_flags
        fig = page()
        ax = fig.subplots()
        counts = []
        for name, bit, color in BAD_KINDS:
            ys, xs = np.nonzero(flags & bit)
            ax.scatter(xs, ys, s=2, color=color, label=f"{name}: {len(xs)}", linewidths=0)
            counts.append(f"{len(xs)} {name}")
        ax.set(xlim=(0, flags.shape[1]), ylim=(0, flags.shape[0]), xlabel="Detector x [pixel]",
               ylabel="Detector y [pixel]", facecolor="black")
        ax.legend(loc="upper right", markerscale=5)
        yield finish(fig, f"BAD — {np.count_nonzero(flags)} flagged pixels ({', '.join(counts)})")
