"""Dictionary of every FITS keyword, category and recipe option the pipeline uses.

Add a keyword here before writing it. Header writers take units and comments
from these tables, and tests/test_keywords.py rejects product QC keywords that
are not declared. The QC dictionary delivered to ESO is this file.
"""

# Raw tag -> ESO DPR TYPE. Tags ending in _RAW are raw inputs in the SOF.
RAW_TYPES = {
    "DARK_RAW": "DARK",
    "FLAT_RAW": "FLAT",
    "P2VM_RAW": "P2VM",
    "WAVELAMP_RAW": "WAVE,LAMP",
    "SINGLE_SCI_RAW": "OBJECT,SINGLE",
}
# Raw input that is not a *_RAW tag.
EXTRA_RAW_TAGS = ("PHASE_REFERENCE",)

# Keywords read from raw files (instrument data interface dictionary).
RAW = {
    "ESO DPR CATG": "Observation category",
    "ESO DPR TYPE": "Observation type",
    "ESO DPR TECH": "Observation technique",
    "ESO DET ID": "Detector identifier",
    "ESO DET WIN1 STRX": "Window start x, one-based",
    "ESO DET WIN1 STRY": "Window start y, one-based",
    "ESO DET WIN1 NX": "Window width [pixel]",
    "ESO DET WIN1 NY": "Window height [pixel]",
    "ESO DET WIN1 BINX": "Binning in x",
    "ESO DET WIN1 BINY": "Binning in y",
    "ESO DET WIN1 DIT1": "[s] Integration time",
    "ESO DET WIN1 UIT1": "[s] User integration time",
    **{f"ESO INS SHUT{n} ST": f"Shutter T{n} open" for n in range(1, 5)},
}

# QC keyword -> (unit, description). Units appear as "[unit]" in the comment.
QC = {
    "ESO QC MEDIANDARK SC": ("ADU", "Spatial median of temporal mean"),
    "ESO QC DARKRMS SC": ("ADU", "Spatial median of temporal population RMS"),
    "ESO QC BADPIX SC": ("", "Number of flagged detector pixels"),
    "ESO QC BADPIX_DARK SC": ("", "Number with hot or dead bits"),
    "ESO QC BADPIX_RMS SC": ("", "Number with noisy bit"),
    **{f"ESO QC FLATFLUX SC{i}": ("ADU/DIT", f"Mean-frame total flux, T{t}")
       for i, t in enumerate("4321", 1)},
    "ESO QC MINWAVE SC": ("m", "Shortest wavelength covered by every output"),
    "ESO QC MAXWAVE SC": ("m", "Longest wavelength covered by every output"),
    "ESO QC RMSWAVE SC": ("m", "Sample stdev of accepted lamp fit residuals"),
    "ESO QC NWAVE SC": ("", "Number of accepted lamp fit residuals"),
    **{f"ESO QC P2VM_COHERENCE_SC{b}": ("", "Mean normalized coherence over outputs "
                                         "and wavelengths")
       for b in ("43", "42", "41", "32", "31", "21")},
}

# Recipe option: Python keyword -> (option name, help). The CPL parameter is
# planetes.<calib|reduce>.<option name>; the default is the step's signature.
PARAMETERS = {
    "bad_pixel_threshold": ("bad-dark-threshold",
                            "Positive multiplier in the retained dark bad-pixel tests"),
    "aperture_width_pixels": ("profile-width", "Full extraction aperture in detector pixels"),
    "metrology_trace": ("metrology-trace",
                        "Detect 25 traces and omit the middle metrology trace"),
    "keep_diagnostics": ("keep-diagnostics",
                         "Retain P2VM scan and fit diagnostics in the calibration product"),
    "gd_search_min_m": ("gd-search-min-m",
                        "Coarse GD lower initialization [m]; Newton remains unconstrained"),
    "gd_search_max_m": ("gd-search-max-m",
                        "Coarse GD upper initialization [m]; Newton remains unconstrained"),
    "dl_gain": ("dl-gain", "Delay-line mm to OPD m gain per baseline, e.g. "
                "43:-1,42:-1,41:-1,32:1,31:-1,21:-1; empty disables the DL_SCAN OPD guide"),
}


def is_raw(tag: str) -> bool:
    """RAW (not calibration) input in the DICD and CPL frame-group sense."""
    return tag.endswith("_RAW") or tag in EXTRA_RAW_TAGS
