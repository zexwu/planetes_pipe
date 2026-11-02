"""Select calibration inputs by recorded type, shutters, exposure and detector."""
from planetes_pipe.io.raw import RAW_TYPES
from pathlib import Path
from astropy.io import fits as afits
from planetes_pipe.model.instrument import layout, check_setup
from .raw import read_raw_metadata


def _one_calibration(paths, label):
    """Require an unambiguous exposure rather than mix separate runs."""
    if len(paths) != 1:
        raise ValueError(f"{label}: expected one file, found {len(paths)}: "
                         + ", ".join(str(p) for p in paths))
    return paths[0]


def find_calibration_inputs(directory: Path):
    """Associate raw types/shutters and match each dark by exposure and detector."""
    groups = {name: {} for name in ("FLAT_RAW", "P2VM_RAW")}
    darks, lamps, metadata = [], [], {}
    categories = {v["dpr_type"]: k for k, v in RAW_TYPES.items()}
    for path in sorted(Path(directory).glob("*.fits")):
        header = afits.getheader(path)
        category = categories.get(header.get("ESO DPR TYPE"))
        if "ESO PRO CATG" in header or category not in (
            *groups,
            "DARK_RAW",
            "WAVELAMP_RAW",
        ):
            continue
        meta = read_raw_metadata(path, category=category)
        if None in meta.shutters:
            raise ValueError(f"Missing shutter header: {path}")
        metadata[path] = meta
        if category == "DARK_RAW":
            darks.append(path)
        elif category == "WAVELAMP_RAW":
            lamps.append(path)
        else:
            identity = "".join(t for t in layout.telescopes if meta.shutters[int(t) - 1])
            groups[category].setdefault(identity, []).append(path)
    flats = {t: _one_calibration(groups["FLAT_RAW"].get(t, []), f"flat S{t}") for t in layout.telescopes}
    scans = {b: _one_calibration(groups["P2VM_RAW"].get(b, []), f"scan S{b}") for b in layout.baselines}
    lamp = _one_calibration(lamps, "wave lamp")

    def matching_dark(path):
        setup = metadata[path].setup
        if setup.exposure_s is None:
            raise ValueError(f"Cannot select dark without exposure header: {path}")
        matches = []
        for dark in darks:
            if metadata[dark].setup.exposure_s is None:
                continue
            try:
                check_setup(setup, metadata[dark].setup, exposure=True, role="DARK")
            except ValueError:
                continue
            matches.append(dark)
        return _one_calibration(matches, f"dark for {setup.exposure_s:g} s")

    main_dark, lamp_dark = matching_dark(flats["4"]), matching_dark(lamp)
    for path in [*flats.values(), *scans.values()]:
        check_setup(metadata[path].setup, metadata[main_dark].setup, exposure=True, role="DARK")
    return main_dark, lamp_dark, lamp, list(flats.values()), list(scans.values())
