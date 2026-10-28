"""GRAVITY 1.9.6 categories/extension names and explicit PLANETES additions.

Sources: gravi_dfs.h:43-99, gravi_pfits.h:41-95; detector keywords are
ESO-044156 v8 sections 4.4.2.1/4.4.2.7. Shutter keys are a PLANETES
adaptation of gravi_utils.h:131-140, not GRAVITY's hardware numbering.
"""

from ..instrument import BASELINES, TELESCOPES
from ..products import (
    BadPixelMap,
    Dark,
    Flat,
    P2VM,
    P2VMRed,
    Wave,
    WavelengthReference,
)

RAW_TYPES = {
    "DARK_RAW": ("CALIB", "DARK", "IMAGE", 0),
    "FLAT_RAW": ("CALIB", "FLAT", "SPECTRUM", 1),
    "P2VM_RAW": ("CALIB", "P2VM", "INTERFEROMETRY", 2),
    "WAVELAMP_RAW": ("CALIB", "WAVE,LAMP", "SPECTRUM", 4),
    "SINGLE_SCI_RAW": ("SCIENCE", "OBJECT,SINGLE", "INTERFEROMETRY", 4),
}
RAW_IMAGE = "IMAGING_DATA_SC"
DARK_RMS = "IMAGING_ERR_SC"
PROFILE_PARAMS = "PROFILE_PARAMS"
WAVE_DATA = "WAVE_DATA_SC"
P2VM_DATA = "P2VM_SC"
DETECTOR_DATA = "IMAGING_DETECTOR_SC"
DIT = "ESO DET WIN1 DIT1"
UIT = "ESO DET WIN1 UIT1"
NDIT = "ESO DET WIN1 NDIT"
ROI_KEYS = {
    "origin_x": "ESO DET WIN1 STRX",
    "origin_y": "ESO DET WIN1 STRY",
    "width": "ESO DET WIN1 NX",
    "height": "ESO DET WIN1 NY",
    "binning_x": "ESO DET WIN1 BINX",
    "binning_y": "ESO DET WIN1 BINY",
}
# These are explicitly private; they are not ESO dictionary/QC definitions.
META_EXT = "PLANETES_META"
FORMAT_KEY = "PLANETES FORMAT"
SHUTTER_KEYS = tuple(f"PLANETES INS SHUT{n} ST" for n in range(1, 5))
RAW_FORMAT = 3
PRODUCT_FORMAT = 5


CATEGORIES = {
    Dark: "DARK",
    BadPixelMap: "BAD",
    Flat: "FLAT",
    Wave: "WAVE",
    P2VM: "P2VM",
    P2VMRed: "SINGLE_SCI_P2VMRED",
    WavelengthReference: "LINE_LIST",
}


# Numerical QC uses physical names. Only this mapping knows FITS keyword spelling.
QC_KEYWORDS = {
    "dark_median_adu": "ESO QC MEDIANDARK SC",
    "dark_rms_median_adu": "ESO QC DARKRMS SC",
    "bad_pixel_count": "ESO QC BADPIX SC",
    "bad_dark_pixel_count": "ESO QC BADPIX_DARK SC",
    "bad_rms_pixel_count": "ESO QC BADPIX_RMS SC",
    "p2vm_coherence_mean": "ESO QC P2VM_COHERENCE_AVG_SC",
    "p2vm_coherence_spectral_std": "ESO QC P2VM_COHERENCERMS_AVG_SC",
    "p2vm_phase_spectral_std_rad": "ESO QC P2VM_PHASERMS_AVG_SC",
    "wavelength_residual_std_m": "ESO QC RMSWAVE SC",
    "wavelength_min_m": "ESO QC MINWAVE SC",
    "wavelength_max_m": "ESO QC MAXWAVE SC",
    "p2vm_transmission_mean": "ESO QC P2VM_TRANS_AVG_SC",
    "p2vm_transmission_min": "ESO QC P2VM_TRANS_MIN_SC",
    "p2vm_transmission_max": "ESO QC P2VM_TRANS_MAX_SC",
}
for i, telescope in enumerate(TELESCOPES, start=1):
    QC_KEYWORDS[f"flat_flux_t{telescope}_adu"] = f"ESO QC FLATFLUX SC{i}"
for region in (1, 13):
    for suffix in ("med", "left", "right"):
        QC_KEYWORDS[f"profile_center_region{region}_{suffix}_pixels"] = (
            f"ESO QC PROFILE_CENTER SC{region} {suffix.upper()}"
        )
    QC_KEYWORDS[f"profile_sigma_region{region}_pixels"] = (
        f"ESO QC PROFILE_WIDTH SC{region} MED"
    )
for baseline in BASELINES:
    for name, field in (
        ("mean", "COHERENCE"),
        ("min", "MINCOHERENCE"),
        ("max", "MAXCOHERENCE"),
    ):
        QC_KEYWORDS[f"p2vm_coherence_{name}_b{baseline}"] = (
            f"ESO QC P2VM_{field}_SC{baseline}"
        )

QC_STATUS_KEYS = {
    "readout_noise": "RON",
    "bad_flat_pixel_count": "ESO QC BADPIX_FLAT SC",
    "flat_flux": "FLATFLUX",
    "wavelength_residual_std": "RMSWAVE",
    "reference_wavelength": "REFWAVE",
    "p2vm_chi_square": "P2VM CHI2",
    "p2vm_group_delay_rejection": "P2VM GD MAX",
    "averaged_visibility": "VISAMP/VISPHI AVG",
    "rejected_frame_ratio": "REJECTED RATIO",
    "snr_and_flux_rate": "SNR/FLUX RATE",
}
for region in (1, 13):
    QC_STATUS_KEYS[f"profile_sigma_region{region}_pixels"] = f"PROFILE_WIDTH SC{region}"
for baseline in BASELINES:
    QC_STATUS_KEYS[f"p2vm_coherence_mean_b{baseline}"] = f"P2VM_COHERENCE_SC{baseline}"
FLAT_IMAGE = "PLANETES_FLAT_IMAGE"
SPARSE_PROFILE = "PLANETES_PROFILE"
FLAT_FLUX = "PLANETES_FLAT_FLUX"
WAVE_FIT = "PLANETES_WAVE_FIT"
LINE_LIST = "LINE_LIST"
WAVELENGTH = "PLANETES_WAVELENGTH"
P2VM_INVERSE = "PLANETES_P2VM_INVERSE"
OPD = "PLANETES_OPD"
GROUP_DELAY = "PLANETES_GD"
ELLIPSE_PHASE = "PLANETES_ELLIPSE_PHASE"
ELLIPSE_AMPLITUDE = "PLANETES_ELLIPSE_AMPLITUDE"
P2VM_FIT_RMS = "PLANETES_P2VM_FIT_RMS"
SCIENCE_FLUX = "PLANETES_FLUX"
SCIENCE_VISIBILITY = "PLANETES_VIS"
PRO_CATEGORY = "ESO PRO CATG"
PRO_TYPE = "ESO PRO TYPE"
PRO_SCIENCE = "ESO PRO SCIENCE"
DPR_CATEGORY = "ESO DPR CATG"
DPR_TYPE = "ESO DPR TYPE"
DPR_TECHNIQUE = "ESO DPR TECH"
BAD_BITS = "PLANETES BAD BITS"
