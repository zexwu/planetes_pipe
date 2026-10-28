"""FITS header keywords, HDU names, raw definitions, product categories and QC.

GRAVITY 1.9.6: gravi_dfs.h:43-99, gravi_pfits.h:41-95.
Detector header fields: ESO-044156 v8, sections 4.4.2.1/4.4.2.7.
PLANETES-specific names are grouped below. Shutter numbering is the physical
1..4 order, adapted from gravi_utils.h:131-140 rather than GRAVITY hardware IDs.
"""

from ..products import Bad, Dark, Flat, P2VM, P2VMRed, Wave, WavelengthReference


# Keep these lookup tables aligned for comparison by column.
# fmt: off

# ESO raw (DPR) and product (PRO) header keywords.
DPR_CATEGORY  = "ESO DPR CATG"
DPR_TYPE      = "ESO DPR TYPE"
DPR_TECHNIQUE = "ESO DPR TECH"
PRO_CATEGORY  = "ESO PRO CATG"
PRO_TYPE      = "ESO PRO TYPE"
PRO_SCIENCE   = "ESO PRO SCIENCE"

# Detector timing and window. Python ROI origins are zero-based; FITS is one-based.
DIT  = "ESO DET WIN1 DIT1"
UIT  = "ESO DET WIN1 UIT1"
NDIT = "ESO DET WIN1 NDIT"

ROI_KEYS = {
    "origin_x":  "ESO DET WIN1 STRX",
    "origin_y":  "ESO DET WIN1 STRY",
    "width":     "ESO DET WIN1 NX",
    "height":    "ESO DET WIN1 NY",
    "binning_x": "ESO DET WIN1 BINX",
    "binning_y": "ESO DET WIN1 BINY",
}

# GRAVITY-derived HDU names. These are extension names, not header keywords.
RAW_IMAGE      = "IMAGING_DATA_SC"
DARK_RMS       = "IMAGING_ERR_SC"
DETECTOR_DATA  = "IMAGING_DETECTOR_SC"
PROFILE_PARAMS = "PROFILE_PARAMS"
WAVE_DATA      = "WAVE_DATA_SC"
P2VM_DATA      = "P2VM_SC"
LINE_LIST      = "LINE_LIST"

# PLANETES HDU names.
FLAT_IMAGE         = "PLANETES_FLAT_IMAGE"
SPARSE_PROFILE     = "PLANETES_PROFILE"
FLAT_FLUX          = "PLANETES_FLAT_FLUX"
WAVE_FIT           = "PLANETES_WAVE_FIT"
WAVELENGTH         = "PLANETES_WAVELENGTH"
P2VM_INVERSE       = "PLANETES_P2VM_INVERSE"
OPD                = "PLANETES_OPD"
GROUP_DELAY        = "PLANETES_GD"
ELLIPSE_PHASE      = "PLANETES_ELLIPSE_PHASE"
ELLIPSE_AMPLITUDE  = "PLANETES_ELLIPSE_AMPLITUDE"
P2VM_FIT_RMS       = "PLANETES_P2VM_FIT_RMS"
SCIENCE_FLUX       = "PLANETES_FLUX"
SCIENCE_VISIBILITY = "PLANETES_VIS"
META_EXT           = "PLANETES_META"

# Private header keywords and versions, outside the ESO dictionary.
FORMAT_KEY   = "PLANETES FORMAT"
BAD_BITS     = "PLANETES BAD BITS"
SHUTTER_KEYS = tuple(f"PLANETES INS SHUT{n} ST" for n in range(1, 5))

RAW_FORMAT     = 3
PRODUCT_FORMAT = 6

# Raw exposure definitions: DPR values plus the number of open physical shutters.
RAW_TYPES = {
    "DARK_RAW": {
        "dpr_category":  "CALIB",
        "dpr_type":      "DARK",
        "dpr_technique": "IMAGE",
        "open_shutters": 0,
    },
    "FLAT_RAW": {
        "dpr_category":  "CALIB",
        "dpr_type":      "FLAT",
        "dpr_technique": "SPECTRUM",
        "open_shutters": 1,
    },
    "P2VM_RAW": {
        "dpr_category":  "CALIB",
        "dpr_type":      "P2VM",
        "dpr_technique": "INTERFEROMETRY",
        "open_shutters": 2,
    },
    "WAVELAMP_RAW": {
        "dpr_category":  "CALIB",
        "dpr_type":      "WAVE,LAMP",
        "dpr_technique": "SPECTRUM",
        "open_shutters": 4,
    },
    "SINGLE_SCI_RAW": {
        "dpr_category":  "SCIENCE",
        "dpr_type":      "OBJECT,SINGLE",
        "dpr_technique": "INTERFEROMETRY",
        "open_shutters": 4,
    },
}

# Scientific product class -> DFS product category.
CATEGORIES = {
    Dark:                "DARK",
    Bad:                 "BAD",
    Flat:                "FLAT",
    Wave:                "WAVE",
    P2VM:                "P2VM",
    P2VMRed:             "SINGLE_SCI_P2VMRED",
    WavelengthReference: "LINE_LIST",
}

# Scientific metric -> FITS keyword. Fixed physical identities are explicit.
QC_KEYWORDS = {
    # Dark and bad pixels.
    "dark_median_adu":      "ESO QC MEDIANDARK SC",
    "dark_rms_median_adu":  "ESO QC DARKRMS SC",
    "bad_pixel_count":      "ESO QC BADPIX SC",
    "bad_dark_pixel_count": "ESO QC BADPIX_DARK SC",
    "bad_rms_pixel_count":  "ESO QC BADPIX_RMS SC",
    # Telescope flats: SC1..4 correspond to T4,T3,T2,T1.
    "flat_flux_t4_adu": "ESO QC FLATFLUX SC1",
    "flat_flux_t3_adu": "ESO QC FLATFLUX SC2",
    "flat_flux_t2_adu": "ESO QC FLATFLUX SC3",
    "flat_flux_t1_adu": "ESO QC FLATFLUX SC4",
    # Wavelength solution.
    "wavelength_residual_std_m": "ESO QC RMSWAVE SC",
    "wavelength_min_m":          "ESO QC MINWAVE SC",
    "wavelength_max_m":          "ESO QC MAXWAVE SC",
    # P2VM coherence, in the six physical baselines.
    "p2vm_coherence_mean_b43": "ESO QC P2VM_COHERENCE_SC43",
    "p2vm_coherence_mean_b42": "ESO QC P2VM_COHERENCE_SC42",
    "p2vm_coherence_mean_b41": "ESO QC P2VM_COHERENCE_SC41",
    "p2vm_coherence_mean_b32": "ESO QC P2VM_COHERENCE_SC32",
    "p2vm_coherence_mean_b31": "ESO QC P2VM_COHERENCE_SC31",
    "p2vm_coherence_mean_b21": "ESO QC P2VM_COHERENCE_SC21",
}

# fmt: on
