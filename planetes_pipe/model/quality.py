"""QC values and their FITS keyword metadata; measurements live in processing."""

from dataclasses import dataclass, field


@dataclass(slots=True)
class DarkQC:
    """DARK quality measurements; None means not measured."""

    median_adu: float | None = field(default=None, metadata={
        "keyword": "ESO QC MEDIANDARK SC",
        "comment": "[ADU] Spatial median of temporal mean",
    })
    rms_median_adu: float | None = field(default=None, metadata={
        "keyword": "ESO QC DARKRMS SC",
        "comment": "[ADU] Spatial median of temporal population RMS",
    })


@dataclass(slots=True)
class BadQC:
    """BAD quality measurements; None means not measured."""

    pixel_count: int | None = field(default=None, metadata={
        "keyword": "ESO QC BADPIX SC",
        "comment": "Number of flagged detector pixels",
    })
    dark_pixel_count: int | None = field(default=None, metadata={
        "keyword": "ESO QC BADPIX_DARK SC",
        "comment": "Number with hot or dead bits",
    })
    rms_pixel_count: int | None = field(default=None, metadata={
        "keyword": "ESO QC BADPIX_RMS SC",
        "comment": "Number with noisy bit",
    })


@dataclass(slots=True)
class FlatQC:
    """FLAT quality measurements; None means not measured."""

    flux_t4_adu: float | None = field(default=None, metadata={
        "keyword": "ESO QC FLATFLUX SC1",
        "comment": "[ADU/DIT] Mean-frame total flux, T4",
    })
    flux_t3_adu: float | None = field(default=None, metadata={
        "keyword": "ESO QC FLATFLUX SC2",
        "comment": "[ADU/DIT] Mean-frame total flux, T3",
    })
    flux_t2_adu: float | None = field(default=None, metadata={
        "keyword": "ESO QC FLATFLUX SC3",
        "comment": "[ADU/DIT] Mean-frame total flux, T2",
    })
    flux_t1_adu: float | None = field(default=None, metadata={
        "keyword": "ESO QC FLATFLUX SC4",
        "comment": "[ADU/DIT] Mean-frame total flux, T1",
    })


@dataclass(slots=True)
class WaveQC:
    """WAVE quality measurements; None means not measured."""

    min_m: float | None = field(default=None, metadata={
        "keyword": "ESO QC MINWAVE SC",
        "comment": "[m] Coverage shared by every detector output",
        "um_keyword": "ESO QC MINWAVE UM SC",
    })
    max_m: float | None = field(default=None, metadata={
        "keyword": "ESO QC MAXWAVE SC",
        "comment": "[m] Coverage shared by every detector output",
        "um_keyword": "ESO QC MAXWAVE UM SC",
    })
    residual_std_m: float | None = field(default=None, metadata={
        "keyword": "ESO QC RMSWAVE SC",
        "comment": "[m] Sample stdev of accepted lamp fit residuals; N={n}",
        "um_keyword": "ESO QC RMSWAVE UM SC",
    })
    residual_count: int = 0  # Number of accepted lamp residuals, for the FITS comment.


@dataclass(slots=True)
class P2VMQC:
    """P2VM quality measurements; None means not measured."""

    coherence_mean_b43: float | None = field(default=None, metadata={
        "keyword": "ESO QC P2VM_COHERENCE_SC43",
        "comment": "Mean normalized coherence over four outputs and wavelengths",
    })
    coherence_mean_b42: float | None = field(default=None, metadata={
        "keyword": "ESO QC P2VM_COHERENCE_SC42",
        "comment": "Mean normalized coherence over four outputs and wavelengths",
    })
    coherence_mean_b41: float | None = field(default=None, metadata={
        "keyword": "ESO QC P2VM_COHERENCE_SC41",
        "comment": "Mean normalized coherence over four outputs and wavelengths",
    })
    coherence_mean_b32: float | None = field(default=None, metadata={
        "keyword": "ESO QC P2VM_COHERENCE_SC32",
        "comment": "Mean normalized coherence over four outputs and wavelengths",
    })
    coherence_mean_b31: float | None = field(default=None, metadata={
        "keyword": "ESO QC P2VM_COHERENCE_SC31",
        "comment": "Mean normalized coherence over four outputs and wavelengths",
    })
    coherence_mean_b21: float | None = field(default=None, metadata={
        "keyword": "ESO QC P2VM_COHERENCE_SC21",
        "comment": "Mean normalized coherence over four outputs and wavelengths",
    })


Quality = DarkQC | BadQC | FlatQC | WaveQC | P2VMQC
