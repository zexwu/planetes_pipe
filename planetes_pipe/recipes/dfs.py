"""CPL binding of explicit categories/parameters and product publication."""

from pathlib import Path

from astropy.io import fits as afits
from cpl import core, dfs, ui

from planetes_pipe import __version__
from planetes_pipe.io.fits import encode_product, product_filename
from planetes_pipe.io.keywords import CATEGORIES, PRO_CATEGORY, RAW_TYPES
from planetes_pipe.io.hdu import new_fits_file
from planetes_pipe.io.raw import read_raw_metadata
from planetes_pipe.products import P2VM, P2VMRed, Wave, WavelengthReference
from planetes_pipe.recipes.parameters import STATIC_NAME


def cpl_frame_type(cls):
    """Return the CPL image/table classification for a scientific product class.

    Parameters
    ----------
    cls : type
        Product class, not an instance. Wave, WavelengthReference, P2VM and
        P2VMRed use TABLE; detector Dark/Bad/Flat products use IMAGE.

    Returns
    -------
    frame_type : cpl.ui.Frame.FrameType
        Classification used when registering the generated DFS product.
    """
    return (
        ui.Frame.FrameType.TABLE
        if cls in (WavelengthReference, Wave, P2VM, P2VMRed)
        else ui.Frame.FrameType.IMAGE
    )


def read_sof(frameset, required: dict[str, int], optional: tuple[str, ...] = ()):
    """Check SOF roles/counts and classify input frames for DFS provenance.

    Parameters
    ----------
    frameset : cpl.ui.FrameSet
        Recipe inputs, each with an explicit SOF tag and filename.
    required : dict of str to int
        Required tag -> exact input count. Missing or excess entries fail.
    optional : tuple of str, optional
        Additional allowed tags, each with at most one input. Other tags fail.

    Returns
    -------
    paths : dict of str to list of Path
        Paths grouped by tag. Products are not loaded or substituted into this map.
    used : cpl.ui.FrameSet
        Classified RAW/CALIB frames. A recipe must remove unused optional inputs
        before passing this set to save_products.
    """
    grouped = {}
    for frame in frameset:
        if frame.tag not in required and frame.tag not in optional:
            raise ValueError(f"Unexpected input tag: {frame.tag}")
        grouped.setdefault(frame.tag, []).append(Path(frame.file))
    missing = set(required) - grouped.keys()
    if missing:
        raise ValueError(f"Missing required input tags: {sorted(missing)}")
    for tag, count in required.items():
        if len(grouped[tag]) != count:
            raise ValueError(
                f"Expected {count} {tag} input(s), got {len(grouped[tag])}"
            )
    for tag in optional:
        if len(grouped.get(tag, [])) > 1:
            raise ValueError(f"Expected at most one {tag} input")

    used = ui.FrameSet()
    for tag in (*required, *optional):
        for path in grouped.get(tag, []):
            used.append(
                ui.Frame(
                    file=str(path),
                    tag=tag,
                    group=ui.Frame.FrameGroup.RAW
                    if tag in RAW_TYPES or tag == "PHASE_REFERENCE"
                    else ui.Frame.FrameGroup.CALIB,
                    frameType=ui.Frame.FrameType.TABLE
                    if tag in ("LINE_LIST", "WAVE", "P2VM")
                    else ui.Frame.FrameType.IMAGE,
                )
            )
    return grouped, used


def read_shutter_id(path: Path, category: str) -> str:
    """Identify a telescope/beam pair from physical shutter readback in a raw header.

    Parameters
    ----------
    path : Path
        Raw FITS cube using the standard IMAGING_DATA_SC image extension.
    category : str
        Expected raw role, normally FLAT_RAW or P2VM_RAW.

    Returns
    -------
    identity : str
        Open telescope numbers in descending physical order, e.g. '4' or '43'.

    Notes
    -----
    All four shutter states must be known. Unknown readback raises ValueError;
    bench files with missing metadata require explicit Python identity mappings.
    """
    raw = read_raw_metadata(path, category=category)
    if None in raw.shutters:
        raise ValueError(
            f"Unknown shutters for {path}; use Python with an explicit file mapping"
        )
    return "".join(str(n) for n in range(4, 0, -1) if raw.shutters[n - 1])


def save_products(outputs, used, recipe, *, reference: Path):
    """Write, sign and register the products of one ESO recipe invocation.

    Parameters
    ----------
    outputs : iterable of tuple
        (static_filename, product, quality) for each output. The fixed name is
        used only with static-name=True. quality is the already measured QC
        mapping or None; no statistics are computed by the writer.
    used : cpl.ui.FrameSet
        Actual raw/calibration inputs used by the recipe, excluding disabled inputs.
    recipe : cpl.ui.PyRecipe
        Recipe name and current ParameterList for DFS provenance recording.
    reference : Path
        Raw input used for output naming and DFS header inheritance. Its stem is
        preserved, including any timestamp or UNKNOWN marker. Recipes choose a
        fixed physical input for multi-file calibrations, independently of SOF
        order. Existing observation-time cards are inherited without invention.

    Returns
    -------
    products : cpl.ui.FrameSet
        Published outputs registered as final PRODUCT frames with their categories.

    Notes
    -----
    CPL supplies input and parameter provenance; the shared encoder supplies data
    and scientific units. Checksums and DFS signatures are completed before each
    file is published. Existing paths are never overwritten. A later failure does
    not remove an earlier completed output.
    """
    inherit = next(frame for frame in used if Path(frame.file) == reference)
    # CPL takes primary observation cards from the first RAW even when inherit
    # is supplied. GRAVITY puts its reference first for this same reason.
    used = ui.FrameSet(
        [inherit] + [frame for frame in used if Path(frame.file) != reference]
    )
    products = ui.FrameSet()
    for static_filename, product, quality in outputs:
        output = (
            static_filename
            if recipe.parameters[STATIC_NAME].value
            else product_filename(reference, product)
        )
        with new_fits_file(output) as path:
            properties = core.PropertyList()
            properties.append(
                core.Property(PRO_CATEGORY, core.Type.STRING, CATEGORIES[type(product)])
            )
            dfs.save_propertylist(
                used,
                recipe.parameters,
                used,
                recipe.name,
                properties,
                f"planetes-pipe/{__version__}",
                str(path),
                inherit=inherit,
            )
            with afits.open(path, mode="update") as hdus:
                encoded = encode_product(product, hdus[0].header, quality=quality)
                hdus[0].header = encoded[0].header
                for item in encoded[1:]:
                    hdus.append(item)
                for item in hdus:
                    item.add_checksum()
            frame = ui.Frame(
                file=str(path),
                tag=CATEGORIES[type(product)],
                group=ui.Frame.FrameGroup.PRODUCT,
                level=ui.Frame.FrameLevel.FINAL,
                frameType=cpl_frame_type(type(product)),
            )
            ui.FrameSet([frame]).sign_products()
        frame.file = str(output)
        products.append(frame)
    return products
