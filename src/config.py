"""Load and validate hierarchical experiment configurations."""

from pathlib import Path
from typing import Any

import yaml


REFERENCE_KEYS = {
    "data_config": "data",
    "model_config": "model",
    "tracking_config": "tracking",
}


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load one YAML mapping."""
    path = Path(path)
    with path.open(encoding="utf-8") as file:
        value = yaml.safe_load(file) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Configuration must contain a mapping: {path}")
    return value


def load_experiment_config(path: str | Path) -> dict[str, Any]:
    """Resolve an experiment file and its referenced component configs."""
    path = Path(path).resolve()
    experiment = load_yaml(path)
    resolved: dict[str, Any] = {}

    for reference_key, output_key in REFERENCE_KEYS.items():
        reference = experiment.pop(reference_key, None)
        if not reference:
            raise ValueError(f"Missing required key '{reference_key}' in {path}")
        resolved[output_key] = load_yaml((path.parent / reference).resolve())

    resolved.update(experiment)
    validate_experiment_config(resolved)
    return resolved


def validate_experiment_config(config: dict[str, Any]) -> None:
    """Validate the minimum contract required for a tracked experiment."""
    required = ("name", "data", "model", "tracking", "training", "evaluation")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Missing resolved configuration sections: {missing}")

    seed = config["training"].get("seed")
    if not isinstance(seed, int):
        raise ValueError("training.seed must be an integer")

    metrics = config["evaluation"].get("metrics")
    if not isinstance(metrics, list) or not metrics:
        raise ValueError("evaluation.metrics must be a non-empty list")

