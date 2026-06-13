"""Helpers for executing Model 1 evaluation stages in a notebook namespace."""

from pathlib import Path
from typing import Any


EVALUATION_ROOT = Path(__file__).resolve().parent


def run_stage(relative_path: str, namespace: dict[str, Any]) -> None:
    """Execute one evaluation stage in a shared notebook namespace."""
    stage_path = EVALUATION_ROOT / relative_path
    if not stage_path.exists():
        raise FileNotFoundError(f"Evaluation stage not found: {stage_path}")

    print(f"\n{'=' * 72}")
    print(f"Running Model 1 evaluation stage: {relative_path}")
    print(f"{'=' * 72}")
    source = stage_path.read_text(encoding="utf-8")
    exec(compile(source, str(stage_path), "exec"), namespace)
