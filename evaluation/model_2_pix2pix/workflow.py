"""Execute Model 2 evaluation stages in a shared notebook namespace."""

from pathlib import Path
from typing import Any


EVALUATION_ROOT = Path(__file__).resolve().parent


def run_stage(relative_path: str, namespace: dict[str, Any]) -> None:
    stage_path = EVALUATION_ROOT / relative_path
    if not stage_path.exists():
        raise FileNotFoundError(f"Evaluation stage not found: {stage_path}")
    print(f"\n{'=' * 72}\nRunning Model 2 evaluation stage: {relative_path}\n{'=' * 72}")
    exec(compile(stage_path.read_text(encoding="utf-8"), str(stage_path), "exec"), namespace)
