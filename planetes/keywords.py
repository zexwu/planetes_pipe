"""GRAVITY 1.9.6 categories/extension names and explicit PLANETES additions.

Sources: gravi_dfs.h:43-99, gravi_pfits.h:41-95; detector keywords are
ESO-044156 v8 sections 4.4.2.1/4.4.2.7. Shutter keys are a PLANETES
adaptation of gravi_utils.h:131-140, not GRAVITY's hardware numbering.
"""

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
HASH_KEY = "PLANETES CONTENT SHA256"
SHUTTER_KEYS = tuple(f"PLANETES INS SHUT{n} ST" for n in range(1, 5))
SCHEMA_VERSION = 3
