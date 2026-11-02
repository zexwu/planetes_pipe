"""Read raw FITS headers and stream detector frames from an explicit image HDU."""

from contextlib import contextmanager
from dataclasses import replace
from planetes_pipe.model.raw import Raw, RawMetadata
from pathlib import Path

from astropy.io import fits as afits

from planetes_pipe.model.instrument import DetectorSetup
from .common import as_compute_array, read_setup, read_shutters

RAW_TYPES = {
    "DARK_RAW": {"dpr_type": "DARK", "open_shutters": 0},
    "FLAT_RAW": {"dpr_type": "FLAT", "open_shutters": 1},
    "P2VM_RAW": {"dpr_type": "P2VM", "open_shutters": 2},
    "WAVELAMP_RAW": {"dpr_type": "WAVE,LAMP", "open_shutters": 4},
    "SINGLE_SCI_RAW": {"dpr_type": "OBJECT,SINGLE", "open_shutters": 4},
}

CHUNK_FRAMES = 16


def read_raw_metadata(
    path: Path, *, category: str, hdu: int | str = "IMAGING_DATA_SC"
) -> RawMetadata:
    """Read physical detector and shutter metadata for an explicitly classified raw file.

    Parameters
    ----------
    path : Path
        Raw FITS file containing a (frame, y, x) detector cube in ADU.
    category : str
        Expected role: DARK_RAW, FLAT_RAW, WAVELAMP_RAW, P2VM_RAW or SINGLE_SCI_RAW.
        Input role selected by the caller; recorded shutters are passed through.
    hdu : int or str, optional
        Selected image HDU; default IMAGING_DATA_SC. No automatic HDU search.

    Returns
    -------
    metadata : RawMetadata
        Category, shutter states in physical T1..T4 order, and DetectorSetup.
        Missing physical fields remain None; width/height come from cube dimensions.
        FITS one-based ROI origins are converted to zero-based Python coordinates.

    Notes
    -----
    Metadata are decoded as recorded. Callers select the input role; processing
    checks that selected calibrations match the detector setup. Raw data are read only.
    """
    with afits.open(path, memmap=True) as hdus:
        return _metadata(hdus, category, hdu)


def _metadata(hdus, category, hdu):
    header = hdus[0].header
    image = hdus[hdu]
    setup = replace(
        read_setup(header) or DetectorSetup(),
        height=image.shape[1],
        width=image.shape[2],
        exposure_s=header.get("ESO DET WIN1 DIT1", header.get("ESO DET WIN1 UIT1")),
    )
    return RawMetadata(category, read_shutters(header), setup)


def iter_raw_cube(path: Path, *, hdu: int | str = "IMAGING_DATA_SC"):
    """Yield detector frames from one explicitly selected raw FITS image HDU.

    Parameters
    ----------
    path : Path
        Raw FITS cube with nonempty (frame, y, x) dimensions, in ADU.
    hdu : int or str, optional
        Image HDU name/index; default IMAGING_DATA_SC, or 0 for primary-HDU data.

    Yields
    ------
    chunk : ndarray, shape (up to 16 frames, y, x)
        Native float64 detector samples in original frame order. All frames,
        including the final partial chunk, are yielded without dark subtraction.

    Notes
    -----
    The FITS handle stays open during iteration; samples are decoded as ADU.
    """
    with afits.open(path, memmap=True) as hdus:
        data = hdus[hdu].data
        for start in range(0, len(data), CHUNK_FRAMES):
            yield as_compute_array(data[start : start + CHUNK_FRAMES])


@contextmanager
def read(path: Path, *, category: str, hdu: int | str = "IMAGING_DATA_SC"):
    """Open a raw cube; consume Raw.frames inside this context in 16-frame chunks."""
    with afits.open(path, memmap=True) as hdus:
        metadata = _metadata(hdus, category, hdu)
        image = hdus[hdu]

        def chunks():
            for start in range(0, image.shape[0], CHUNK_FRAMES):
                yield as_compute_array(image.data[start:start + CHUNK_FRAMES])

        frames = chunks()
        try:
            yield Raw(frames, metadata.category, metadata.shutters, metadata.setup)
        finally:
            frames.close()
