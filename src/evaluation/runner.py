"""Model-agnostic paired reconstruction evaluation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
from PIL import Image

from src.metrics import reconstruction_metrics, summarize_metrics


Predictor = Callable[[object], Image.Image]
TargetLoader = Callable[[object], Image.Image]


class PairedEvaluationRunner:
    """Evaluate a predictor against pixel-aligned paired targets."""

    def __init__(self, output_dir: str | Path, lpips_model=None, device: str = "cpu") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.lpips_model = lpips_model
        self.device = device

    def run(
        self,
        samples: Iterable[object],
        predictor: Predictor,
        target_loader: TargetLoader,
    ) -> dict:
        """Write per-sample metrics and return aggregate mean and standard deviation."""
        records: list[dict[str, float | str]] = []
        for index, sample in enumerate(samples):
            prediction = _as_array(predictor(sample))
            target = _as_array(target_loader(sample))
            metrics = reconstruction_metrics(target, prediction, self.lpips_model, self.device)
            records.append({"sample_index": str(index), **metrics})

        _write_csv(self.output_dir / "metrics.csv", records)
        numeric = [{k: float(v) for k, v in row.items() if k != "sample_index"} for row in records]
        summary = {"samples": len(records), "metrics": summarize_metrics(numeric)}
        (self.output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n",
            encoding="utf-8",
        )
        return summary


def _as_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def _write_csv(path: Path, records: list[dict]) -> None:
    if not records:
        raise ValueError("Evaluation produced no records")
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

