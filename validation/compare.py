"""Compare preserved scientific fields across the intentional schema migration."""

from pathlib import Path
import json
import numpy as np

from planetes.fits import read_product


def compare(reference: Path, candidate: Path, *, science_folder="science"):
    """Require exact numerical bytes and unchanged mappings; no tolerance relaxation."""
    arrays = reference / "arrays"
    manifest = json.loads((arrays / "manifest.json").read_text())
    results = []
    for stage, entries in manifest.items():
        folder = science_folder if stage == "reduced" else stage
        product = read_product(candidate / folder / f"planetes_{stage}.fits")
        for name, saved in entries.items():
            if name == "dark_map_adu":
                actual = read_product(
                    candidate / "dark_main/planetes_dark.fits"
                ).mean_adu
            elif name == "bad_pixel_flags":
                actual = read_product(
                    candidate / "dark_main/planetes_dark.fits"
                ).bad_pixel_flags
            else:
                actual = getattr(product, name)
            if "array" in saved:
                expected = np.load(arrays / saved["array"], allow_pickle=False)
                equal = (
                    actual.dtype == expected.dtype
                    and actual.shape == expected.shape
                    and actual.tobytes() == expected.tobytes()
                )
                difference = float(
                    np.nanmax(np.abs(expected.astype(complex) - actual.astype(complex)))
                )
            else:
                equal = json.dumps(actual, sort_keys=True) == json.dumps(
                    saved["value"], sort_keys=True
                )
                difference = None
            results.append(
                {
                    "product": stage,
                    "field": name,
                    "bitwise_equal": equal,
                    "max_abs_difference": difference,
                }
            )
    return {
        "all_equal": all(row["bitwise_equal"] for row in results),
        "fields": results,
    }


def compare_products(reference: Path, candidate: Path):
    """Compare all seven typed products including scientific parameters and parent links."""
    from dataclasses import fields
    from planetes.models import fingerprint

    results = []
    for folder, name in [
        ("dark_main", "dark"),
        ("dark_wave", "dark"),
        ("dark_science", "dark"),
        ("flat", "flat"),
        ("wave", "wave"),
        ("p2vm", "p2vm"),
        ("science", "reduced"),
    ]:
        relative = Path(folder) / f"planetes_{name}.fits"
        before, after = (
            read_product(root / relative) for root in (reference, candidate)
        )
        for item in fields(before):
            if item.name == "provenance":
                continue
            expected, actual = getattr(before, item.name), getattr(after, item.name)
            equal = (
                (
                    expected.dtype == actual.dtype
                    and expected.shape == actual.shape
                    and expected.tobytes() == actual.tobytes()
                )
                if isinstance(expected, np.ndarray)
                else expected == actual
            )
            results.append(
                {"product": folder, "field": item.name, "bitwise_equal": equal}
            )
        results.append(
            {
                "product": folder,
                "field": "scientific_fingerprint",
                "bitwise_equal": fingerprint(before) == fingerprint(after),
            }
        )
    return {
        "all_equal": all(row["bitwise_equal"] for row in results),
        "fields": results,
    }
