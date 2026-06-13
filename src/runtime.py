"""Runtime inspection and production-readiness checks."""

from __future__ import annotations

import platform
from pathlib import Path
from typing import Any

import torch

from src.data.manifest import VTONManifest
from src.models import get_model_spec


def inspect_runtime(config: dict[str, Any], project_root: str | Path = ".") -> dict[str, Any]:
    """Inspect compute, manifests, notebook source, and adapter status."""
    root = Path(project_root).resolve()
    model_spec = get_model_spec(config["model"]["key"])
    report: dict[str, Any] = {
        "experiment": config["name"],
        "model": model_spec.display_name,
        "production_adapter_status": model_spec.production_adapter_status,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "source_notebook_exists": (root / model_spec.source_notebook).exists(),
        "manifests": {},
    }

    for split, path in config["data"]["manifests"].items():
        manifest_path = root / path
        record: dict[str, Any] = {"path": str(manifest_path), "exists": manifest_path.exists()}
        if manifest_path.exists():
            pairing_mode = config["data"]["pairing_modes"][split]
            manifest = VTONManifest(manifest_path, split, pairing_mode=pairing_mode)
            record["pairing_mode"] = pairing_mode
            record.update(manifest.validate())
        report["manifests"][split] = record
    return report
