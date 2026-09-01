"""Freshness manifests for intermediate pipeline products."""

import hashlib
import json
import logging
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .products import ProductSpec, get_product_id

Product = str | tuple[str, str] | ProductSpec
log = logging.getLogger("recipes")


@dataclass(slots=True)
class ManifestStore:
    """Create and validate sidecar manifests for pipeline products."""

    context: Any

    @staticmethod
    def path_for(path: Path) -> Path:
        return path.with_name(path.name + ".manifest.json")

    @staticmethod
    def _json_hash(value: Any) -> str:
        payload = json.dumps(
            value, sort_keys=True, separators=(",", ":"), default=str
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _code_hash() -> str:
        recipe_dir = Path(__file__).resolve().parent
        names = (
            "__init__.py",
            "flat.py",
            "p2vm.py",
            "preproc.py",
            "products.py",
            "reduce.py",
            "wave.py",
        )
        digest = hashlib.sha256()
        for name in names:
            path = recipe_dir / name
            digest.update(name.encode("utf-8"))
            digest.update(path.read_bytes())
        return digest.hexdigest()

    def _effective_parameters(self, stage: str) -> dict[str, Any]:
        from . import COMMANDS

        command_meta = getattr(COMMANDS.get(stage), "meta", {})
        defaults = command_meta.get("defaults", {})
        configured = self.context.conf.get("parameters", {}).get(stage, {})
        active = self.context.active_parameters.get(stage, {})
        return {**defaults, **configured, **active}

    def _source_path(self, value: str | Path) -> Path:
        path = Path(value).expanduser()
        if path.is_absolute():
            return path.resolve()
        return (self.context.data_dir / path).resolve()

    def _source_state(self, values: Sequence[str | Path]) -> list[dict[str, Any]]:
        states = []
        for value in values:
            path = self._source_path(value)
            try:
                stat = path.stat()
            except FileNotFoundError:
                states.append({"path": str(path), "missing": True})
            else:
                states.append(
                    {
                        "path": str(path),
                        "size": stat.st_size,
                        "mtime_ns": stat.st_mtime_ns,
                    }
                )
        return sorted(states, key=lambda state: state["path"])

    def snapshot(self, product: Product) -> dict[str, Any]:
        """Describe the current inputs, parameters, and numerical implementation."""
        product_id = get_product_id(product)
        calib = self.context.conf["calib"]

        flat_sources = [calib["dark"], *calib["flat"]]
        wave_sources = [*flat_sources, calib["wave"], calib["wave_dark"]]
        p2vm_sources = [*wave_sources, *calib["p2vm"]]
        if "wavesc" in calib:
            p2vm_sources.append(calib["wavesc"])
        if "wavesc_dark" in calib:
            p2vm_sources.append(calib["wavesc_dark"])

        stages: list[str]
        sources: list[str | Path]
        if product_id == "flat":
            stages, sources = ["flat"], flat_sources
        elif product_id in {"wave", "wave_aberr"}:
            stages, sources = ["flat", "wave"], wave_sources
        elif product_id == ("preproc", "p2vm"):
            stages = ["flat", "wave", "preproc"]
            sources = p2vm_sources
        elif product_id == "p2vm":
            stages = ["flat", "wave", "preproc", "p2vm"]
            sources = p2vm_sources
        elif isinstance(product_id, tuple) and product_id[0] in {"preproc", "reduced"}:
            object_name = product_id[1]
            entry = self.context.conf["object"][object_name]
            if isinstance(entry, Mapping):
                object_sources = [entry["data"]]
                if entry.get("dark") is not None:
                    object_sources.append(entry["dark"])
            else:
                object_sources = [entry]
            stages = ["flat", "wave", "preproc"]
            sources = [*wave_sources, *object_sources]
            if product_id[0] == "reduced":
                stages.extend(["p2vm", "reduce"])
                sources = [*p2vm_sources, *object_sources]
        else:
            stages, sources = [], []

        parameters = {stage: self._effective_parameters(stage) for stage in stages}
        if isinstance(product_id, tuple) and product_id[0] in {"preproc", "reduced"}:
            parameters.setdefault("preproc", {})["object"] = product_id[1]
        if isinstance(product_id, tuple) and product_id[0] == "reduced":
            parameters.setdefault("reduce", {})["object"] = product_id[1]

        config = {
            key: self.context.conf.get(key)
            for key in (
                "frames",
                "xrange",
                "yrange",
                "telescopes",
                "baselines",
                "abcd_order",
                "io_chunk_frames",
            )
        }
        return {
            "manifest_version": 1,
            "product_id": product_id,
            "config": config,
            "parameters": parameters,
            "sources": self._source_state(sources),
            "code_sha256": self._code_hash(),
        }

    def is_current(self, product: Product, path: Path) -> bool:
        """Return whether a product and its manifest match the current context."""
        if not path.exists():
            return False

        manifest_path = self.path_for(path)
        if not manifest_path.exists():
            log.warning(f"Legacy product has no freshness manifest: {path}")
            return True

        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            log.warning(f"Cannot read product manifest {manifest_path}: {exc}")
            return False

        if manifest.get("snapshot_sha256") != self._json_hash(self.snapshot(product)):
            log.warning(f"Product is stale for the current inputs/configuration: {path}")
            return False

        stat = path.stat()
        file_state = manifest.get("file", {})
        if (
            file_state.get("size") != stat.st_size
            or file_state.get("mtime_ns") != stat.st_mtime_ns
        ):
            log.warning(f"Product file changed after its manifest was written: {path}")
            return False
        return True

    def write(self, product: Product, path: Path) -> None:
        """Write a sidecar manifest for an existing product file."""
        snapshot = self.snapshot(product)
        stat = path.stat()
        manifest = {
            **snapshot,
            "snapshot_sha256": self._json_hash(snapshot),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "file": {
                "name": path.name,
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            },
        }
        self._write_json_atomic(self.path_for(path), manifest)

    @staticmethod
    def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
        fd, temporary_name = tempfile.mkstemp(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(value, stream, indent=2, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            temporary_path.replace(path)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise
