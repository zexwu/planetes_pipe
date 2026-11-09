"""FITS keywords, each defined once with its unit and comment, as in an ESO dictionary.

The ESO Data Interface Control Document (ESO-044156 v8, §7.3) defines a keyword by
its name, unit and comment. A Keyword is such an entry, named in ESO's short form:
"DET.WIN1.DIT1" is the card HIERARCH ESO DET WIN1 DIT1, and a name without dots is
a standard FITS keyword. Headers are read and written only through Keyword objects,
so the spelling, unit and comment of a keyword exist in one place.

    DIT.get(header)          # -> 1.0 or None
    DIT.set(header, 1.0)     # HIERARCH ESO DET WIN1 DIT1 = 1.0 / [s] Integration time
    SHUTTER.at(4)            # INS.SHUT4.ST, for indexed entries

QC keywords are defined next to the qc() function of each step module.
"""

from dataclasses import dataclass, replace

from astropy.io import fits


@dataclass(frozen=True, slots=True)
class Keyword:
    """One dictionary entry: ESO short name, comment and unit (DICD §7.3)."""

    name: str
    comment: str
    unit: str = ""

    @property
    def card(self) -> str:
        """The keyword as written in FITS, e.g. 'HIERARCH ESO DET WIN1 DIT1'."""
        if "." not in self.name:
            return self.name
        return "HIERARCH ESO " + self.name.replace(".", " ")

    def at(self, *index) -> "Keyword":
        """An indexed entry: SHUTTER.at(4) is INS.SHUT4.ST, 'Shutter S4 open'.

        The same arguments fill the name and the comment; with more than one, use
        positional fields ("QC.FLATFLUX.SC{0}", "flux of S{1}").
        """
        return replace(self, name=self.name.format(*index), comment=self.comment.format(*index))

    def get(self, header: fits.Header, default=None):
        return header.get(self.card, default)

    def set(self, header: fits.Header, value) -> None:
        """Write value with the comment '[unit] comment', cut to fit one 80-column card."""
        comment = f"[{self.unit}] {self.comment}" if self.unit else self.comment
        header[self.card] = value
        room = 80 - len(header.cards[self.card].image.rstrip()) - len(" / ")
        header[self.card] = (value, comment[: max(room, 0)])


# Primary FITS keywords (DICD §4.1, §4.7.1).
INSTRUME = Keyword("INSTRUME", "Instrument used")
DATE = Keyword("DATE", "Date this file was written (UTC)")
DATE_OBS = Keyword("DATE-OBS", "Observing date (UTC)")
MJD_OBS = Keyword("MJD-OBS", "Observation start", "d")
PIPEFILE = Keyword("PIPEFILE", "Filename of data product")
EXPTIME = Keyword("EXPTIME", "Sum of integration times", "s")
NFRAMES = Keyword("NFRAMES", "Number of frames averaged")
INHERIT = Keyword("INHERIT", "Primary header keywords apply to this extension")

# Raw data classification (DICD §4.4.2.1); raw files only.
DPR_CATG = Keyword("DPR.CATG", "Observation category")
DPR_TYPE = Keyword("DPR.TYPE", "Observation type")
DPR_TECH = Keyword("DPR.TECH", "Observation technique")

# Template: one run of a capture script (DICD §4.4.2.4); TPL.START groups its files.
TPL_ID = Keyword("TPL.ID", "Template signature ID")
TPL_START = Keyword("TPL.START", "TPL start time")

# Detector.
DET_ID = Keyword("DET.ID", "Detector identifier")
WIN_STRX = Keyword("DET.WIN1.STRX", "Lower left pixel in x")
WIN_STRY = Keyword("DET.WIN1.STRY", "Lower left pixel in y")
WIN_NX = Keyword("DET.WIN1.NX", "Window width", "pixel")
WIN_NY = Keyword("DET.WIN1.NY", "Window height", "pixel")
WIN_BINX = Keyword("DET.WIN1.BINX", "Binning factor in x")
WIN_BINY = Keyword("DET.WIN1.BINY", "Binning factor in y")
DIT = Keyword("DET.WIN1.DIT1", "Integration time", "s")
NDIT = Keyword("DET.WIN1.NDIT", "Number of frames")

# Detector window (DICD §4.4.2.7) a calibration must share with the raw data it corrects.
WINDOW = (DET_ID, WIN_STRX, WIN_STRY, WIN_NX, WIN_NY, WIN_BINX, WIN_BINY)

# Instrument.
SHUTTER = Keyword("INS.SHUT{}.ST", "Shutter S{} open")

# Processing (DICD §4.7.1): mandatory in every product.
PRO_CATG = Keyword("PRO.CATG", "Category of pipeline product frame")
PRO_TYPE = Keyword("PRO.TYPE", "Product type")
PRO_TECH = Keyword("PRO.TECH", "Observation technique")
PRO_SCIENCE = Keyword("PRO.SCIENCE", "Scientific product if T")
PRO_DID = Keyword("PRO.DID", "Data dictionary for PRO")
PRO_DATANCOM = Keyword("PRO.DATANCOM", "Number of combined frames")
PRO_RECIPE = Keyword("PRO.REC1.ID", "Pipeline recipe (unique) identifier")
PRO_DRS = Keyword("PRO.REC1.DRS.ID", "Data Reduction System identifier")
PRO_PIPELINE = Keyword("PRO.REC1.PIPE.ID", "Pipeline (unique) identifier")
PRO_RAW_NAME = Keyword("PRO.REC1.RAW{}.NAME", "File name of raw frame")
PRO_RAW_CATG = Keyword("PRO.REC1.RAW{}.CATG", "Category of raw frame")
PRO_CAL_NAME = Keyword("PRO.REC1.CAL{}.NAME", "File name of calibration frame")
PRO_CAL_CATG = Keyword("PRO.REC1.CAL{}.CATG", "Category of calibration frame")
PRO_PARAM_NAME = Keyword("PRO.REC1.PARAM{}.NAME", "Recipe parameter")
PRO_PARAM_VALUE = Keyword("PRO.REC1.PARAM{}.VALUE", "Value of the recipe parameter")


# Quality control shared by every step that reads raw data (io/raw.record_saturation).
QC_SATURATED = Keyword("QC.NSAT.{0}", "Raw samples at the saturation level, {0}")


# Recommended order of hierarchical categories after the primary keywords (DICD §3.1.1.3).
CATEGORY_ORDER = ("DPR", "OBS", "TPL", "GEN", "TEL", "ADA", "INS", "DET")


def sorted_by_category(header: fits.Header) -> fits.Header:
    """Primary keywords first, then ESO keywords by category; order within a group is kept."""

    def rank(card: fits.Card) -> int:
        words = card.keyword.split()
        if len(words) < 2 or words[0] != "ESO":
            return 0
        if words[1] in CATEGORY_ORDER:
            return 1 + CATEGORY_ORDER.index(words[1])
        return 1 + len(CATEGORY_ORDER)

    return fits.Header(sorted(header.cards, key=rank))
